"""v3 prompt variants for VLM ablation study.

Root cause of v2 failure on Qwen:
    Every few-shot assistant turn emitted the *full canonical descriptor*
    verbatim as ``reasoning`` (confidence always 0.97).  At temperature=0
    Qwen nearest-neighbours over the 5 canned strings instead of looking
    at the query image — all 3 failure-trace reasoning texts are
    byte-identical copies of the ``no_defect`` descriptor.

    Compounded by ``detail: "low"`` (OpenAI-specific hint, undefined on
    Qwen) potentially erasing fine detail on the 256×1600 strips, and
    15 reference images burying the query at the end of a long context.

Variants
--------
v3a  Generic assistant ack (breaks copy-attractor), ``detail`` dropped,
     3 shots/class.  Isolates the reasoning-leak fix.

v3b  Collage: all 3 refs per class packed into ONE user turn
     (30 messages → 10 user/assistant pairs).  Query image is salient.

v3c  Low-shot, 1 ref/class only.  5 reference images total, no detail.

v3d  Chain-of-thought: system prompt asks the model to describe what it
     sees *before* naming a class; few-shot assistant turns model the
     "observe → conclude" pattern.  Same schema (DefectPrediction).

v3e  Tiled query: the 256×1600 strip is split into 4 non-overlapping
     256×400 tiles and sent as 4 image items in the final user turn.
     Few-shot refs use the v3a layout (3 shots/class, generic ack).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from .prompt import METAL_CLASS_DESCRIPTIONS, METAL_CLASSES, METAL_SYSTEM_PROMPT

# ── patched system prompt (v3c, v3d, v3e) ────────────────────────────────────
# Problem: the base METAL_SYSTEM_PROMPT says "scan for long thin lines (scratch)".
# Steel strips ALWAYS have rolling-direction texture that looks like thin lines,
# so Qwen at temperature=0 defaults to "scratch" for almost every image.
# Fix: require HIGH-CONTRAST for scratch and explicitly warn about mill texture.
METAL_SYSTEM_PROMPT_V3 = METAL_SYSTEM_PROMPT.replace(
    "long thin lines (scratch)",
    "long HIGH-CONTRAST thin lines that clearly STAND OUT from the background texture (scratch)",
).replace(
    "faint roller lines, output 'no_defect'.",
    "faint roller lines, output 'no_defect'. "
    "IMPORTANT: Normal faint mill-texture lines do NOT qualify as 'scratch' — "
    "a scratch must be a high-contrast mark that is clearly visible above or below "
    "the surrounding surface texture.",
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _image_url(b64: str) -> dict:
    """Build an image_url content block WITHOUT the OpenAI-specific detail key.

    ``detail: "low"`` is an OpenAI-only hint that has undefined behaviour on
    Qwen-VL via OpenRouter.  Dropping it lets each upstream use its own native
    resolution strategy.
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }


def _tile_strip(image_path: Path, n_tiles: int = 4) -> list[str]:
    """Split a wide steel strip into ``n_tiles`` equal-width JPEG tiles.

    Severstal strips are 256×1600 (H×W).  Splitting into 4 gives 256×400
    tiles, each shown individually to the model so fine detail (pitting,
    small inclusions) isn't lost to aggressive downsampling.

    Returns a list of base64-encoded JPEG strings.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size  # PIL: (width, height)
    tile_w = w // n_tiles
    tiles = []
    for i in range(n_tiles):
        x0 = i * tile_w
        x1 = x0 + tile_w if i < n_tiles - 1 else w
        crop = img.crop((x0, 0, x1, h))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
        tiles.append(base64.b64encode(buf.getvalue()).decode())
    return tiles


def _crop_black(image_path: Path, threshold: int = 18) -> str:
    """Crop zero-padded black borders from a Severstal steel strip.

    Many Severstal images have large black regions at the left or right edges
    (zero-padding for regions outside the scanned strip width).  These produce
    a sharp steel-to-black boundary that the VLM misidentifies as a scratch.

    Crops the image to the bounding box of columns whose mean brightness
    exceeds ``threshold``.  If less than 5 % of the width would be removed,
    returns the original bytes unchanged.

    Returns a base64-encoded JPEG string.
    """
    gray = np.array(Image.open(image_path).convert("L"))
    col_means = gray.mean(axis=0)          # shape: (W,)
    visible = col_means > threshold
    if not visible.any():
        return _b64(image_path)
    left = int(visible.argmax())
    right = int(len(visible) - visible[::-1].argmax() - 1)
    margin = 8
    left = max(0, left - margin)
    right = min(gray.shape[1] - 1, right + margin)
    removed = left + (gray.shape[1] - 1 - right)
    if removed < gray.shape[1] * 0.05:
        return _b64(image_path)
    img_rgb = Image.open(image_path).convert("RGB")
    cropped = img_rgb.crop((left, 0, right + 1, img_rgb.height))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ── generic assistant acknowledgement (the core fix) ─────────────────────────

def _ack(cls: str) -> str:
    return f'{{"defect_class": "{cls}", "confidence": 0.95, "reasoning": "Confirmed {cls} reference noted."}}'


# ── CoT variant helpers ───────────────────────────────────────────────────────

# Two-sentence "observe → conclude" reasoning per class used in v3d few-shot.
_COT_ASSISTANT_REASONING: dict[str, str] = {
    "no_defect": (
        "The surface shows uniform grey mill texture. There may be faint, barely-visible "
        "lines in the rolling direction — these are normal and NOT defects. "
        "No high-contrast spots, blobs, or marks stand out from the background. "
        "This matches no_defect."
    ),
    "pitting": (
        "I can see several small, dark, roughly circular spots (2-10 px wide) clustered "
        "together, sitting slightly below the surrounding surface. "
        "This matches pitting."
    ),
    "inclusion": (
        "There are irregular, high-contrast dark blobs with jagged edges embedded in "
        "the steel — elongated along the rolling direction but not pencil-thin. "
        "This matches inclusion."
    ),
    "scratch": (
        "A HIGH-CONTRAST long thin mark runs clearly across the strip, much brighter or "
        "darker than the surrounding texture — it clearly stands out from normal mill lines. "
        "Aspect ratio >> 10:1. This matches scratch."
    ),
    "patch": (
        "A large region covering >10 % of the visible strip has noticeably different "
        "brightness or texture from the surrounding metal, with diffuse edges. "
        "This matches patch."
    ),
}

_COT_SYSTEM_SUFFIX = (
    "\n\nCRITICAL DISTINCTION: Faint, barely-visible lines in the rolling direction are "
    "NORMAL mill texture — label them 'no_defect'. A 'scratch' must be a HIGH-CONTRAST "
    "mark that clearly stands out from the surrounding surface. "
    "\n\nIMPORTANT — response format: in your 'reasoning' field, FIRST describe "
    "in 1-2 sentences exactly what you observe in the image (texture, contrast, spots, "
    "lines, blobs, patches), THEN state which class best matches. "
    "Keep total reasoning under 80 words. "
    "This observe-then-conclude approach improves accuracy."
)

METAL_SYSTEM_PROMPT_COT = METAL_SYSTEM_PROMPT_V3 + _COT_SYSTEM_SUFFIX


def _cot_ack(cls: str) -> str:
    reasoning = _COT_ASSISTANT_REASONING[cls].replace('"', "'")
    return (
        f'{{"defect_class": "{cls}", "confidence": 0.95, "reasoning": "{reasoning}"}}'
    )


# ── variant builders ─────────────────────────────────────────────────────────

def _v3a(image_path: Path, seed_dir: Path,
         shots_per_class: int = 3) -> list[dict]:
    """Generic assistant ack, no detail param, 3 shots/class."""
    messages: list[dict] = [{"role": "system", "content": METAL_SYSTEM_PROMPT}]
    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for img in sorted(class_dir.glob("*.jpg"))[:shots_per_class]:
                messages.append({
                    "role": "user",
                    "content": [
                        _image_url(_b64(img)),
                        {"type": "text",
                         "text": f"Reference image. Label: '{cls}'."},
                    ],
                })
                messages.append({"role": "assistant", "content": _ack(cls)})
    messages.append({
        "role": "user",
        "content": [
            _image_url(_b64(image_path)),
            {"type": "text",
             "text": "Classify this steel strip. Respond ONLY in the JSON schema."},
        ],
    })
    return messages


def _v3b(image_path: Path, seed_dir: Path,
         shots_per_class: int = 3) -> list[dict]:
    """Collage: all N refs per class in one user turn → 5 user/assistant pairs."""
    messages: list[dict] = [{"role": "system", "content": METAL_SYSTEM_PROMPT}]
    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            imgs = sorted(class_dir.glob("*.jpg"))[:shots_per_class]
            if not imgs:
                continue
            content: list[dict] = []
            for img in imgs:
                content.append(_image_url(_b64(img)))
            content.append({
                "type": "text",
                "text": (
                    f"These {len(imgs)} images are all confirmed '{cls}' examples. "
                    "Study what they have in common."
                ),
            })
            messages.append({"role": "user", "content": content})
            messages.append({
                "role": "assistant",
                "content": (
                    f'{{"defect_class": "{cls}", "confidence": 0.95, '
                    f'"reasoning": "Understood — {len(imgs)} {cls} reference images noted."}}'
                ),
            })
    messages.append({
        "role": "user",
        "content": [
            _image_url(_b64(image_path)),
            {"type": "text",
             "text": "Classify this steel strip. Respond ONLY in the JSON schema."},
        ],
    })
    return messages


def _v3c(image_path: Path, seed_dir: Path) -> list[dict]:
    """Low-shot: 1 reference image per class only."""
    messages: list[dict] = [{"role": "system", "content": METAL_SYSTEM_PROMPT_V3}]
    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            imgs = sorted(class_dir.glob("*.jpg"))
            if not imgs:
                continue
            messages.append({
                "role": "user",
                "content": [
                    _image_url(_b64(imgs[0])),
                    {"type": "text", "text": f"Reference image. Label: '{cls}'."},
                ],
            })
            messages.append({"role": "assistant", "content": _ack(cls)})
    messages.append({
        "role": "user",
        "content": [
            _image_url(_b64(image_path)),
            {"type": "text",
             "text": "Classify this steel strip. Respond ONLY in the JSON schema."},
        ],
    })
    return messages


def _v3d(image_path: Path, seed_dir: Path,
         shots_per_class: int = 3) -> list[dict]:
    """Chain-of-thought: system prompt + few-shot model the observe→conclude pattern."""
    messages: list[dict] = [{"role": "system", "content": METAL_SYSTEM_PROMPT_COT}]
    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for img in sorted(class_dir.glob("*.jpg"))[:shots_per_class]:
                messages.append({
                    "role": "user",
                    "content": [
                        _image_url(_b64(img)),
                        {"type": "text", "text": f"Reference image. Label: '{cls}'."},
                    ],
                })
                messages.append({"role": "assistant", "content": _cot_ack(cls)})
    messages.append({
        "role": "user",
        "content": [
            _image_url(_b64(image_path)),
            {"type": "text",
             "text": (
                 "Classify this steel strip. "
                 "Remember: describe what you see first, then state the class. "
                 "Respond ONLY in the JSON schema."
             )},
        ],
    })
    return messages


def _v3e(image_path: Path, seed_dir: Path,
         shots_per_class: int = 3, n_tiles: int = 4) -> list[dict]:
    """Tiled query: split the 256×1600 strip into 4 tiles in the final turn."""
    messages: list[dict] = [{"role": "system", "content": METAL_SYSTEM_PROMPT_V3}]
    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for img in sorted(class_dir.glob("*.jpg"))[:shots_per_class]:
                messages.append({
                    "role": "user",
                    "content": [
                        _image_url(_b64(img)),
                        {"type": "text", "text": f"Reference image. Label: '{cls}'."},
                    ],
                })
                messages.append({"role": "assistant", "content": _ack(cls)})

    # Tile the query strip
    tiles = _tile_strip(image_path, n_tiles=n_tiles)
    tile_content: list[dict] = [_image_url(t) for t in tiles]
    tile_content.append({
        "type": "text",
        "text": (
            f"These are {n_tiles} sequential left-to-right tiles from a single "
            "256×1600 steel strip. Classify the strip overall. "
            "Respond ONLY in the JSON schema."
        ),
    })
    messages.append({"role": "user", "content": tile_content})
    return messages


def _v3f(image_path: Path, seed_dir: Path,
         shots_per_class: int = 1) -> list[dict]:
    """v3c + black-padding crop applied to every image before encoding.

    Many Severstal images contain large zero-padded black borders at the
    left/right edges.  The sharp steel-to-black boundary is misidentified
    by the VLM as a high-contrast scratch line.  Cropping the padding before
    encoding eliminates this false signal while preserving the visible steel
    surface.
    """
    messages: list[dict] = [{"role": "system", "content": METAL_SYSTEM_PROMPT_V3}]
    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            for img in sorted(class_dir.glob("*.jpg"))[:shots_per_class]:
                messages.append({
                    "role": "user",
                    "content": [
                        _image_url(_crop_black(img)),
                        {"type": "text", "text": f"Reference image. Label: '{cls}'."},
                    ],
                })
                messages.append({"role": "assistant", "content": _ack(cls)})
    messages.append({
        "role": "user",
        "content": [
            _image_url(_crop_black(image_path)),
            {"type": "text",
             "text": "Classify this steel strip. Respond ONLY in the JSON schema."},
        ],
    })
    return messages


# ── public dispatcher ─────────────────────────────────────────────────────────

VARIANTS = Literal["v3a", "v3b", "v3c", "v3d", "v3e", "v3f"]


def build_messages(
    image_path: Path,
    seed_dir: Path,
    *,
    variant: str,
    domain: str | None = None,  # kept for API compatibility; metal-only in v3
) -> list[dict]:
    """Return the multimodal message list for the requested variant.

    Parameters
    ----------
    image_path : Path
        The query image to classify.
    seed_dir : Path
        Directory of per-class reference images (subdirs named by class).
    variant : str
        One of ``v3a``, ``v3b``, ``v3c``, ``v3d``, ``v3e``.
    domain : str | None
        Ignored (all v3 variants are metal-only).  Kept for a consistent
        call signature with the legacy ``prompt.build_messages``.
    """
    if variant == "v3a":
        return _v3a(image_path, seed_dir)
    if variant == "v3b":
        return _v3b(image_path, seed_dir)
    if variant == "v3c":
        return _v3c(image_path, seed_dir)
    if variant == "v3d":
        return _v3d(image_path, seed_dir)
    if variant == "v3e":
        return _v3e(image_path, seed_dir)
    if variant == "v3f":
        return _v3f(image_path, seed_dir)
    raise ValueError(
        f"Unknown variant '{variant}'. Choose one of: v3a, v3b, v3c, v3d, v3e, v3f."
    )
