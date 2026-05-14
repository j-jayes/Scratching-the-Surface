"""Render the exact 15-image few-shot grid the VLM sees, with class labels.

Tiles ``data/splits_metal_v2/seed/<class>/`` into a 5-row × 3-col grid annotated
with the class label and the visual descriptor used in the system prompt.
Used by the slide "What we tell the VLM" in `website/slides.qmd`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from PIL import Image

from cascade_defect.vlm.prompt import (
    METAL_CLASSES,
    METAL_CLASS_DESCRIPTIONS,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-dir", default="data/splits_metal_v2/seed")
    p.add_argument("--out", default="website/assets/samples/vlm_few_shot_grid.png")
    p.add_argument("--shots", type=int, default=3)
    args = p.parse_args()

    seed_dir = ROOT / args.seed_dir
    classes = [c for c in METAL_CLASSES if (seed_dir / c).is_dir()]
    n_rows = len(classes)
    n_cols = args.shots + 1  # +1 column for the descriptor text

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.6, n_rows * 1.5),
                             gridspec_kw={"width_ratios": [1] * args.shots + [2.4]})

    for ri, cls in enumerate(classes):
        imgs = sorted((seed_dir / cls).glob("*.jpg"))[:args.shots]
        for ci in range(args.shots):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ci < len(imgs):
                img = Image.open(imgs[ci]).convert("L")
                ax.imshow(img, cmap="gray", aspect="auto")
            if ci == 0:
                ax.set_ylabel(f"{cls}", fontsize=11, rotation=0,
                              ha="right", va="center", labelpad=8,
                              fontweight="bold")
        # Right-most column: the descriptor text
        ax = axes[ri, -1]
        ax.axis("off")
        descriptor = METAL_CLASS_DESCRIPTIONS.get(cls, "")
        ax.text(0.0, 0.5, "\n".join(wrap(descriptor, width=42)),
                fontsize=8, va="center", ha="left", family="monospace")

    fig.suptitle("Few-shot prompt: 3 reference images × 5 classes + visual cue",
                 fontsize=12)
    fig.tight_layout()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
