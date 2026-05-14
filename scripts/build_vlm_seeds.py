"""Pick N reference images per defect class for VLM few-shot prompting.

Reads ``data/splits_metal_v2/train_labels.csv`` (excluding the locked test set
and the VLM benchmark subset by construction — they live in different files)
and copies ``--n-per-class`` images per class into
``data/splits_metal_v2/seed/<class>/``. The same seed dir is consumed by both
the Azure GPT-4.1-mini client and the OpenRouter Qwen client so the comparison
is apples-to-apples.

Run::

    uv run python scripts/build_vlm_seeds.py --n-per-class 3
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv",
                   default="data/splits_metal_v2/train_labels.csv")
    p.add_argument("--out-dir", default="data/splits_metal_v2/seed")
    p.add_argument("--n-per-class", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    by_cls: dict[str, list[str]] = defaultdict(list)
    with (ROOT / args.train_csv).open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != "train":
                continue
            by_cls[r["label"]].append(r["image_path"])

    out_root = ROOT / args.out_dir
    if out_root.exists():
        shutil.rmtree(out_root)

    summary: dict[str, list[str]] = {}
    for cls, paths in sorted(by_cls.items()):
        rng.shuffle(paths)
        take = paths[:args.n_per_class]
        cls_dir = out_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for src in take:
            src_path = ROOT / src
            dst = cls_dir / src_path.name.replace(".png", ".jpg")
            # prompt builder globs *.jpg, so make sure all files have that ext.
            if src_path.suffix.lower() == ".jpg":
                shutil.copy(src_path, dst)
            else:
                from PIL import Image  # local import
                Image.open(src_path).convert("RGB").save(dst, "JPEG")
            copied.append(str(dst.relative_to(ROOT)))
        summary[cls] = copied
        print(f"  {cls:12s} → {len(copied)} files in {cls_dir.relative_to(ROOT)}")
    print(f"\nWrote seed dir {out_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
