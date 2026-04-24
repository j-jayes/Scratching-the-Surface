"""Render inference visualisations for the website.

For a hand-picked set of cascade traces from `reports/eval_cascade.jsonl`,
generate four panels per example:

  1. Original input image
  2. Layer-1 autoencoder reconstruction
  3. Layer-1 absolute-difference heatmap (where the AE "got surprised")
  4. Layer-2 YOLO detection overlay (bbox + class label)

Outputs:  website/assets/inferences/<example_id>/{input,recon,diff,yolo}.jpg
          website/assets/inferences/manifest.json
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO

from cascade_defect.layer1_autoencoder.model import ConvAutoencoder

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "website" / "assets" / "inferences"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hand-picked examples spanning the cascade's three behaviours.
# Keys must match `image` field in reports/eval_cascade.jsonl (Windows-style
# paths preserved as-recorded).
EXAMPLES = [
    {
        "id": "01_l1_dropped_inclusion",
        "trace_image": "data\\splits\\test\\inclusion\\img_001.jpg",
        "headline": "L1 short-circuit (an honest miss)",
        "summary": (
            "The autoencoder reconstructs this inclusion almost perfectly "
            "(MSE 0.0009 vs threshold 0.0067), so the cascade returns "
            "`no_defect` in 65 ms without invoking L2 or L3. This is the v1 "
            "dataset's fingerprint: the AE was trained on `rolled-in_scale` "
            "as a normal proxy, and inclusions happen to look 'normal-ish' "
            "to it. The Phase-J swap to VisA is exactly the fix."
        ),
    },
    {
        "id": "02_l3_oracle_crazing",
        "trace_image": "data\\splits\\test\\crazing\\img_005.jpg",
        "headline": "Full cascade L1 → L2 → L3 (Oracle nails it)",
        "summary": (
            "The AE flags this frame (MSE 0.0088 > τ=0.0067), L2's "
            "COCO-pretrained YOLO returns garbage (`cat`, conf 0.012 — the "
            "v1 YOLO is not yet defect-trained), so the router escalates to "
            "the Oracle. GPT-4.1-mini correctly identifies the crazing "
            "pattern with a one-line rationale grounded in the few-shot "
            "exemplars."
        ),
    },
    {
        "id": "03_l3_oracle_patches",
        "trace_image": "data\\splits\\test\\patches\\img_000.jpg",
        "headline": "Same path, different defect class",
        "summary": (
            "Demonstrates that the Oracle's few-shot prompt generalises "
            "across the six defect classes — same routing, same latency "
            "envelope, same per-token cost. The router is class-agnostic; "
            "all of the domain knowledge sits in the L3 system prompt."
        ),
    },
]

IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

print(f"Loading models on {DEVICE}...")
ae = ConvAutoencoder().to(DEVICE)
ae.load_state_dict(torch.load(ROOT / "models" / "autoencoder" / "best.pt", map_location=DEVICE))
ae.eval()

yolo = YOLO(str(ROOT / "models" / "yolo" / "best.pt"))


def load_trace(image_field: str) -> dict | None:
    with (ROOT / "reports" / "eval_cascade.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("image") == image_field:
                return row
    return None


def save_input(img: Image.Image, dst: Path) -> None:
    img.convert("RGB").resize((512, 512), Image.NEAREST).save(dst, quality=92)


def render_recon_and_diff(img: Image.Image, dst_recon: Path, dst_diff: Path) -> float:
    x = TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon = ae(x).clamp(0, 1)
    mse = float(((recon - x) ** 2).mean().item())
    recon_img = (recon.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(recon_img).resize((512, 512), Image.NEAREST).save(dst_recon, quality=92)
    diff = np.abs(recon - x).squeeze().mean(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.imshow(diff, cmap="inferno", vmin=0, vmax=max(diff.max(), 1e-3))
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(dst_diff, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close(fig)
    return mse


def render_yolo_overlay(img: Image.Image, dst: Path) -> dict:
    res = yolo.predict(img.convert("RGB"), conf=0.001, verbose=False)[0]
    canvas = img.convert("RGB").resize((512, 512), Image.NEAREST)
    draw = ImageDraw.Draw(canvas)
    sx = canvas.width / img.width
    sy = canvas.height / img.height
    detections = []
    if res.boxes is not None and len(res.boxes) > 0:
        # Plot the top-3 boxes
        confs = res.boxes.conf.cpu().numpy()
        order = np.argsort(-confs)[:3]
        for rank, i in enumerate(order):
            x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
            c = float(confs[i])
            cls = yolo.names[int(res.boxes.cls[i])]
            colour = "#ef4444" if rank == 0 else "#fbbf24"
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=colour, width=3)
            label = f"{cls} {c:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except OSError:
                font = ImageFont.load_default()
            tw = draw.textlength(label, font=font)
            draw.rectangle([x1 * sx, y1 * sy - 22, x1 * sx + tw + 8, y1 * sy], fill=colour)
            draw.text((x1 * sx + 4, y1 * sy - 22), label, fill="white", font=font)
            detections.append({"class": cls, "confidence": round(c, 3),
                               "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)]})
    canvas.save(dst, quality=92)
    return {"detections": detections, "n_total": int(0 if res.boxes is None else len(res.boxes))}


manifest = []
for ex in EXAMPLES:
    print(f"\n=== {ex['id']} ===")
    src_path = ROOT / ex["trace_image"].replace("\\", "/")
    if not src_path.exists():
        print(f"  MISSING: {src_path}")
        continue
    img = Image.open(src_path)
    sub = OUT_DIR / ex["id"]
    sub.mkdir(parents=True, exist_ok=True)

    save_input(img, sub / "input.jpg")
    local_mse = render_recon_and_diff(img, sub / "recon.jpg", sub / "diff.jpg")
    yolo_info = render_yolo_overlay(img, sub / "yolo.jpg")
    trace = load_trace(ex["trace_image"]) or {}

    manifest.append({
        "id": ex["id"],
        "headline": ex["headline"],
        "summary": ex["summary"],
        "true_class": trace.get("true_class"),
        "decision": trace.get("decision"),
        "stopped_at_layer": trace.get("stopped_at_layer"),
        "elapsed_ms": trace.get("elapsed_ms"),
        "trace": trace.get("trace"),
        "local_recon_mse": round(local_mse, 6),
        "local_yolo": yolo_info,
        "source_image_relpath": ex["trace_image"].replace("\\", "/"),
    })
    print(f"  recon MSE={local_mse:.6f}  yolo n={yolo_info['n_total']}")

(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nWrote {len(manifest)} examples + manifest.json under {OUT_DIR}")
