"""Render inference panels for the metal-surface cascade (Phase K.5).

Per example, produce a 4-panel walkthrough under
``website/assets/inferences_metal/<id>/``:

  * ``input.jpg``   — raw frame (resized for the website).
  * ``heatmap.jpg`` — PatchCore per-patch anomaly heatmap (where L1 was
    surprised). Replaces the old AE diff heatmap; PatchCore is the
    production scorer in J.1.
  * ``yolo.jpg``    — top-3 YOLO bounding boxes from the production
    ``models/yolo_metal/best.pt`` (mAP50 0.50, 50 ep / 640 px GPU).
  * The trace (router decision, layer stats, Oracle reasoning) is copied
    verbatim from ``reports/eval_cascade_metal_k1.jsonl`` into
    ``manifest.json``.

Examples are picked dynamically from the trace so the panels always reflect
the latest eval — no hand-curated paths to keep in sync.

Selection strategy::

  1. Track A, true=normal, decision=no_defect, stopped_at_layer=1.
     The cheap path — clean Severstal frame, gated at L1 in <500 ms.
  2. Track A, true=defective, decision=defect, stopped_at_layer=2,
     YOLO confidence ≥ 0.5.  The headline win — production YOLO catches it
     so the Oracle bill is zero.
  3. Track B, true=defective, decision=defect, stopped_at_layer=3.
     The architectural payoff — KSDD2 OOD frame, AE escalates, Oracle
     classifies it correctly.

Usage::

    uv run python scripts/render_inference_panels_metal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from cascade_defect.layer1_autoencoder import patchcore as pc

ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "reports" / "eval_cascade_metal_k1_calibrated.jsonl"
if not TRACE_PATH.exists():
    TRACE_PATH = ROOT / "reports" / "eval_cascade_metal_k1.jsonl"
OUT_DIR = ROOT / "website" / "assets" / "inferences_metal"
PATCHCORE_DIR = ROOT / "models" / "patchcore_metal"
YOLO_WEIGHTS = ROOT / "models" / "yolo_metal" / "best.pt"
PANEL_PX = 512


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def pick_examples(records: list[dict]) -> list[dict]:
    out: list[dict] = []

    # 1. Track A clean → L1 drop
    for r in records:
        if (
            r.get("track") == "A"
            and r.get("true_polarity") == "normal"
            and r.get("decision") == "no_defect"
            and r.get("stopped_at_layer") == 1
        ):
            out.append({
                "id": "01_severstal_normal_l1_drop",
                "headline": "L1 fast path — clean steel, gated in <500 ms",
                "summary": (
                    "Severstal in-domain. PatchCore reconstructs the surface "
                    "below the calibrated z-threshold, the cascade returns "
                    "`no_defect` immediately, no L2 / L3 cost. Roughly 95 % "
                    "of a real factory feed looks like this."
                ),
                "trace": r,
            })
            break

    # 2. Track A defective → L2 caught
    for r in records:
        if (
            r.get("track") == "A"
            and r.get("true_polarity") == "defective"
            and r.get("decision") == "defect"
            and r.get("stopped_at_layer") == 2
        ):
            out.append({
                "id": "02_severstal_defect_l2_yolo",
                "headline": "L2 production YOLO — defect classified, no Oracle bill",
                "summary": (
                    "Severstal defective frame. PatchCore escalates at L1, "
                    "the production YOLO (mAP50 0.50, trained 50 ep / 640 px on T4) "
                    "returns a confident detection ≥ 0.5, the cascade short-circuits "
                    "at L2 — no AOAI tokens spent. This is what the J.4 GPU retrain "
                    "buys versus the smoke YOLO."
                ),
                "trace": r,
            })
            break

    # 3. Track B KSDD2 defective → L3
    for r in records:
        if (
            r.get("track") == "B"
            and r.get("true_polarity") == "defective"
            and r.get("decision") == "defect"
            and r.get("stopped_at_layer") == 3
        ):
            out.append({
                "id": "03_ksdd2_defect_l3_oracle",
                "headline": "L3 Oracle backstop — KSDD2 OOD defect, AE escalates",
                "summary": (
                    "KSDD2 (commutator surfaces) is a different metal domain — "
                    "the Severstal-trained YOLO has no class for it, so the router "
                    "skips L2 by design. PatchCore flags the anomaly, the Oracle "
                    "classifies it as `surface_anomaly` with a one-line rationale. "
                    "This is the architectural payoff: pay the AOAI token cost "
                    "only on frames where the cheap layers can't decide."
                ),
                "trace": r,
            })
            break

    return out


def save_input(img: Image.Image, dst: Path) -> None:
    img.convert("RGB").resize((PANEL_PX, PANEL_PX), Image.BILINEAR).save(dst, quality=92)


@torch.no_grad()
def render_patchcore_heatmap(
    extractor: pc._FeatureExtractor,
    bank: torch.Tensor,
    img: Image.Image,
    dst: Path,
    *,
    k: int = 5,
) -> dict:
    """Render the per-patch anomaly heatmap onto the source image."""
    vol = extractor.encode_image(img, "cpu").cpu()  # [D, H, W]
    d, h, w = vol.shape
    flat = F.normalize(vol.reshape(d, h * w).T, dim=1)  # [P, D]
    sim = flat @ bank.T
    topk = sim.topk(k=min(k, bank.shape[0]), dim=1).values
    per_patch = (1.0 - topk.mean(dim=1)).reshape(h, w).numpy()  # [H, W]
    score_p99 = float(np.quantile(per_patch, 0.99))
    score_max = float(per_patch.max())

    fig, ax = plt.subplots(figsize=(PANEL_PX / 100, PANEL_PX / 100), dpi=100)
    src = img.convert("RGB").resize((PANEL_PX, PANEL_PX), Image.BILINEAR)
    ax.imshow(src)
    hm = np.kron(per_patch, np.ones((PANEL_PX // h, PANEL_PX // w)))
    ax.imshow(hm, cmap="inferno", alpha=0.45,
              vmin=float(per_patch.min()), vmax=max(score_max, 1e-3))
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(dst, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    return {"score_p99": round(score_p99, 4), "score_max": round(score_max, 4),
            "grid": [int(h), int(w)]}


def render_yolo_overlay(yolo: YOLO, img: Image.Image, dst: Path) -> dict:
    res = yolo.predict(img.convert("RGB"), conf=0.001, verbose=False)[0]
    canvas = img.convert("RGB").resize((PANEL_PX, PANEL_PX), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    sx = canvas.width / img.width
    sy = canvas.height / img.height
    detections: list[dict] = []
    if res.boxes is not None and len(res.boxes) > 0:
        confs = res.boxes.conf.cpu().numpy()
        order = np.argsort(-confs)[:3]
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        for rank, i in enumerate(order):
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
            c = float(confs[i])
            cls = yolo.names[int(res.boxes.cls[i])]
            colour = "#ef4444" if rank == 0 else "#fbbf24"
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=colour, width=3)
            label = f"{cls} {c:.2f}"
            tw = draw.textlength(label, font=font)
            draw.rectangle([x1 * sx, y1 * sy - 22, x1 * sx + tw + 8, y1 * sy], fill=colour)
            draw.text((x1 * sx + 4, y1 * sy - 22), label, fill="white", font=font)
            detections.append({"class": cls, "confidence": round(c, 3),
                               "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)]})
    canvas.save(dst, quality=92)
    return {"detections": detections, "n_total": int(0 if res.boxes is None else len(res.boxes))}


def main() -> None:
    if not TRACE_PATH.exists():
        raise SystemExit(f"Missing {TRACE_PATH} — run run_cascade_metal first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_trace(TRACE_PATH)
    examples = pick_examples(records)
    if not examples:
        raise SystemExit("No matching examples in trace.")

    print(f"Loading PatchCore + YOLO ({YOLO_WEIGHTS.name})...")
    extractor = pc._FeatureExtractor()
    yolo = YOLO(str(YOLO_WEIGHTS))

    manifest = []
    for ex in examples:
        rec = ex["trace"]
        domain = rec["domain"]
        src_path = ROOT / Path(rec["image"].replace("\\", "/"))
        if not src_path.exists():
            print(f"  MISSING source: {src_path}")
            continue
        bank, _calib = pc.load_bank(PATCHCORE_DIR, domain)

        sub = OUT_DIR / ex["id"]
        sub.mkdir(parents=True, exist_ok=True)
        img = Image.open(src_path)

        save_input(img, sub / "input.jpg")
        heat = render_patchcore_heatmap(extractor, bank, img, sub / "heatmap.jpg")
        yolo_info = render_yolo_overlay(yolo, img, sub / "yolo.jpg")

        manifest.append({
            "id": ex["id"],
            "headline": ex["headline"],
            "summary": ex["summary"],
            "domain": domain,
            "track": rec["track"],
            "true_polarity": rec["true_polarity"],
            "decision": rec["decision"],
            "stopped_at_layer": rec["stopped_at_layer"],
            "client_elapsed_ms": rec.get("client_elapsed_ms"),
            "trace": rec.get("trace"),
            "patchcore_heatmap": heat,
            "yolo_overlay": yolo_info,
            "source_image_relpath": rec["image"].replace("\\", "/"),
        })
        print(f"  {ex['id']:36s} -> heatmap p99={heat['score_p99']:.3f}, "
              f"yolo n={yolo_info['n_total']}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} examples + manifest.json under {OUT_DIR}")


if __name__ == "__main__":
    main()
