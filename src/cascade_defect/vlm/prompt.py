"""Few-shot prompt + structured-output schema for the VLM defect classifier.

Lifted unchanged from the original ``layer3_gpt4o.oracle`` so the cascade's
existing online behaviour is preserved bit-for-bit.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
METAL_CLASSES = [
    "no_defect",
    "pitting",
    "inclusion",
    "scratch",
    "patch",
]

# Detailed visual descriptors used both in the system prompt and as the
# `reasoning` field on the few-shot assistant turns. Keeping them in one place
# means the slide builder can render the exact text shown to the model.
METAL_CLASS_DESCRIPTIONS: dict[str, str] = {
    "no_defect": (
        "Uniform grey surface. May show normal mill texture, faint roller "
        "lines, mild brightness gradient, or thin horizontal banding from "
        "the rolling process. NO dark spots, NO embedded particles, NO long "
        "linear marks, NO discoloured patches. Mill texture is NOT a defect."
    ),
    "pitting": (
        "One or more small, dark, roughly circular spots or shallow holes in "
        "the surface. Spots are typically 2-15 px wide, often clustered, and "
        "sit BELOW the surface plane (darker than the surrounding metal)."
    ),
    "inclusion": (
        "Foreign material embedded IN the steel: irregular dark blobs or "
        "streaks with high local contrast against a clean grey background. "
        "Edges are jagged, not linear. Often elongated along the rolling "
        "direction but not pencil-thin like a scratch."
    ),
    "scratch": (
        "Long, thin, straight or gently curved LINEAR mark, much longer than "
        "it is wide. Aspect ratio > 10:1. Usually a single bright or dark "
        "line; can appear in groups of parallel lines from a tool drag."
    ),
    "patch": (
        "A LARGE region of the strip whose brightness or texture differs "
        "from the surrounding metal. Covers > 10 percent of the visible area. "
        "Edges are diffuse, not sharp. Often appears as a lighter or darker "
        "zone of rolled-in scale or oxidation."
    ),
}
DefectClass = Literal[
    # NEU (kept for back-compat)
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
    # Severstal + KSDD2
    "pitting",
    "scratch",
    "patch",
    "surface_anomaly",
    # Reject buckets
    "no_defect",
    "uncertain",
]


class DefectPrediction(BaseModel):
    """Structured JSON schema enforced on VLM responses."""

    defect_class: DefectClass = Field(
        description=(
            "One of the supported defect classes, 'no_defect' if the surface "
            "is clean, or 'uncertain' if image quality / framing prevents a "
            "confident judgement."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Subjective confidence 0–1.")
    reasoning: str = Field(description="One-sentence rationale.")


NEU_SYSTEM_PROMPT = (
    "You are an expert quality-control inspector for rolled steel manufacturing. "
    "You classify surface defects in greyscale steel images into one of six classes: "
    f"{', '.join(NEU_CLASSES)}. Use 'no_defect' if no defect is visible. "
    "Use 'uncertain' if the image quality or framing prevents a confident judgement. "
    "You will see a few labelled reference examples followed by an unknown image. "
    "Respond ONLY in the required JSON schema."
)

_METAL_DEFINITIONS_BLOCK = "\n".join(
    f"  - {cls}: {desc}" for cls, desc in METAL_CLASS_DESCRIPTIONS.items()
)

METAL_SYSTEM_PROMPT = (
    "You are an expert quality-control inspector for flat-rolled steel sheet. "
    "You classify each image into EXACTLY ONE of these five labels and nothing "
    f"else: {', '.join(METAL_CLASSES)}.\n\n"
    "Visual definitions:\n"
    f"{_METAL_DEFINITIONS_BLOCK}\n\n"
    "Decision procedure:\n"
    "  1. Scan the image for the four defect signatures: dark spots "
    "(pitting), embedded blobs/streaks (inclusion), long thin lines "
    "(scratch), large discoloured zones (patch).\n"
    "  2. If exactly one signature matches, output that class.\n"
    "  3. If no signature is present and the surface only shows uniform "
    "mill texture / faint roller lines, output 'no_defect'.\n"
    "  4. If multiple signatures could fit, pick the one with the "
    "strongest visual evidence — do NOT default to 'no_defect' just to "
    "play safe, and do NOT default to a defect class just to look "
    "thorough.\n"
    "  5. NEVER invent labels like 'surface_anomaly', 'defect', 'crack', "
    "'rust', 'mark' or 'unknown'. Only the five labels above are valid.\n"
    "  6. NEVER answer 'uncertain' unless the image is corrupted, blank, "
    "or completely out of focus. Treat 'uncertain' as a last resort.\n\n"
    "You will see 3 labelled reference images per class (15 total) followed "
    "by the unknown image. Respond ONLY in the required JSON schema."
)

# Default = NEU for back-compat with v1 callers.
SYSTEM_PROMPT = NEU_SYSTEM_PROMPT


def _system_prompt_for(domain: str | None) -> str:
    if not domain:
        return SYSTEM_PROMPT
    return (METAL_SYSTEM_PROMPT
            if domain.lower() in {"ksdd2", "severstal", "metal"}
            else NEU_SYSTEM_PROMPT)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_messages(image_path: Path, seed_dir: Path,
                   *, domain: str | None = None,
                   detail: str = "low",
                   shots_per_class: int = 3) -> list[dict]:
    """Build the multimodal message list with few-shot exemplars + query image.

    ``domain`` selects the system prompt (NEU vs metal/Severstal+KSDD2).
    """
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt_for(domain)}
    ]

    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            descriptor = METAL_CLASS_DESCRIPTIONS.get(
                cls, "Provided reference example.")
            for img in sorted(class_dir.glob("*.jpg"))[:shots_per_class]:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{_b64(img)}",
                                "detail": detail,
                            },
                        },
                        {"type": "text",
                         "text": (f"Reference example. Ground-truth label: "
                                  f"'{cls}'. Study the visual signature.")},
                    ],
                })
                # Use the canonical visual descriptor as the assistant's
                # reasoning so the model learns the *cue*, not just the label.
                reasoning = descriptor.replace('"', "'")
                messages.append({
                    "role": "assistant",
                    "content": (
                        '{"defect_class": "' + cls + '", '
                        '"confidence": 0.97, '
                        '"reasoning": "' + reasoning + '"}'
                    ),
                })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_b64(image_path)}",
                    "detail": detail,
                },
            },
            {"type": "text",
             "text": "Classify this image. Respond ONLY in the JSON schema."},
        ],
    })
    return messages
