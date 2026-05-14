"""Render a 5×4 grid of representative samples for the slides.

Picks 4 random images per class from the train split and tiles them with class
labels overlaid. Used by `website/slides.qmd` (slide B, "Steel surface defects").

Run::

    uv run python scripts/render_samples_grid.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["no_defect", "pitting", "inclusion", "scratch", "patch"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/splits_metal_v2/train_labels.csv")
    p.add_argument("--out", default="website/assets/samples/severstal_grid.png")
    p.add_argument("--per-class", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    by_class: dict[str, list[str]] = {c: [] for c in CLASSES}
    with (ROOT / args.csv).open() as f:
        for r in csv.DictReader(f):
            if r["label"] in by_class:
                by_class[r["label"]].append(r["image_path"])

    n_rows = len(CLASSES)
    n_cols = args.per_class
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 1.2))
    for ri, cls in enumerate(CLASSES):
        sample = random.sample(by_class[cls], min(n_cols, len(by_class[cls])))
        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ci < len(sample):
                img = Image.open(ROOT / sample[ci]).convert("L")
                ax.imshow(img, cmap="gray", aspect="auto")
            if ci == 0:
                ax.set_ylabel(cls.replace("_", " "), fontsize=11,
                              rotation=0, ha="right", va="center", labelpad=10)

    fig.suptitle("Severstal — 4 samples per class", fontsize=12)
    fig.tight_layout()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
