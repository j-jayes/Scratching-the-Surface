"""Build a YOLOv8 dataset on disk from the Severstal split (Phase J.2).

Reads `data/splits_metal/yolo_{train,val}/` (Severstal defectives only) and
writes Ultralytics-format labels + `data.yaml` into a target directory.

Output layout::

    models/yolo_metal/dataset/
        data.yaml
        images/
            train/   (symlinks or copies of severstal_*.jpg)
            val/
        labels/
            train/   (one .txt per image, YOLO format)
            val/

Each label line is ``<class_id> <x_centre> <y_centre> <w> <h>`` with all
coordinates normalised to [0, 1]. Class indices are 0..3 mapping to
``["pitting", "inclusion", "scratch", "patch"]`` (matches Severstal class IDs
1..4 minus one). When ``--include-ksdd2`` is passed, KSDD2 defects from
``cascade_test/ksdd2/defective/`` are added as a 5th class
``ksdd2_generic`` with bboxes derived from each image's ``_GT.png`` mask
(this collapses Track C into Track B but makes the deployed YOLO usable on
KSDD2 \u2014 the Phase J.2 trade-off).

Class imbalance is reported in ``stats.json`` and consumed by
``layer2_yolo.train_metal`` to derive the YOLO ``class_weights`` arg.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from cascade_defect.data import severstal

logger = logging.getLogger(__name__)

DEFAULT_SPLITS = Path("data/splits_metal")
DEFAULT_OUTPUT = Path("models/yolo_metal/dataset")
SEVERSTAL_CLASS_NAMES = [
    severstal.CLASS_NAMES[1],  # pitting
    severstal.CLASS_NAMES[2],  # inclusion
    severstal.CLASS_NAMES[3],  # scratch
    severstal.CLASS_NAMES[4],  # patch
]
KSDD2_EXTRA_CLASS = "ksdd2_generic"


def _resolve_severstal_image_id(filename: str) -> str:
    """``severstal_0002cc93b.jpg`` \u2192 ``0002cc93b.jpg``."""
    return filename.removeprefix("severstal_")


def _write_severstal_labels(
    images_dir: Path, labels_dir: Path, severstal_root: Path, *, min_area: int
) -> Counter:
    rle_index = severstal._read_train_csv(severstal_root / severstal.CSV_NAME)
    counts: Counter = Counter()
    skipped_no_box = 0
    for img_path in sorted(images_dir.glob("severstal_*.jpg")):
        image_id = _resolve_severstal_image_id(img_path.name)
        defects = rle_index.get(image_id, {})
        lines: list[str] = []
        for class_id, rle in defects.items():
            mask = severstal.rle_to_mask(rle)
            for x, y, w, h in severstal.mask_to_yolo_bboxes(mask, min_area=min_area):
                yolo_cls = class_id - 1  # 1..4 \u2192 0..3
                lines.append(f"{yolo_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                counts[yolo_cls] += 1
        if not lines:
            skipped_no_box += 1
        (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
    logger.info("Severstal labels written: %s (skipped %d with no bbox)", dict(counts), skipped_no_box)
    return counts


def _ksdd2_bboxes_from_mask(mask_path: Path, *, min_area: int) -> list[tuple[float, float, float, float]]:
    import cv2

    mask = np.asarray(Image.open(mask_path).convert("L"))
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out: list[tuple[float, float, float, float]] = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        out.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


def _add_ksdd2(
    output_dir: Path,
    *,
    splits_root: Path,
    ksdd2_class_index: int,
    min_area: int,
) -> int:
    """Copy KSDD2 defectives into the YOLO train set with class_idx labels."""
    src = splits_root / "cascade_test" / "ksdd2" / "defective"
    if not src.exists():
        logger.info("No KSDD2 defective set found at %s \u2014 skipping 5th class", src)
        return 0
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    n = 0
    for img_path in sorted(src.glob("*.png")):
        mask_path = img_path.with_name(f"{img_path.stem}_GT.png")
        if not mask_path.exists():
            continue
        bboxes = _ksdd2_bboxes_from_mask(mask_path, min_area=min_area)
        if not bboxes:
            continue
        dest_img = images_dir / f"ksdd2_{img_path.name}"
        shutil.copy2(img_path, dest_img)
        lines = [
            f"{ksdd2_class_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for x, y, w, h in bboxes
        ]
        (labels_dir / f"{dest_img.stem}.txt").write_text("\n".join(lines))
        n += 1
    logger.info("KSDD2 5th-class images added: %d", n)
    return n


def build(
    splits_root: Path = DEFAULT_SPLITS,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    severstal_root: Path = severstal.DEFAULT_ROOT,
    min_area: int = 16,
    include_ksdd2: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy (or hardlink) the Severstal-defective images into images/{split}/.
    counts = {"train": Counter(), "val": Counter()}
    for split in ("train", "val"):
        src = splits_root / f"yolo_{split}"
        dst = output_dir / "images" / split
        for f in sorted(src.glob("severstal_*.jpg")):
            target = dst / f.name
            if not target.exists():
                shutil.copy2(f, target)
        counts[split] = _write_severstal_labels(
            dst, output_dir / "labels" / split, severstal_root, min_area=min_area,
        )

    class_names = list(SEVERSTAL_CLASS_NAMES)
    if include_ksdd2:
        ksdd2_idx = len(class_names)
        n_added = _add_ksdd2(
            output_dir,
            splits_root=splits_root,
            ksdd2_class_index=ksdd2_idx,
            min_area=min_area,
        )
        if n_added:
            class_names.append(KSDD2_EXTRA_CLASS)
            counts["train"][ksdd2_idx] = n_added

    data_yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml)

    stats = {
        "class_names": class_names,
        "train_counts": dict(counts["train"]),
        "val_counts": dict(counts["val"]),
        "min_area_px": min_area,
        "include_ksdd2": include_ksdd2,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Built YOLO dataset: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--severstal-root", type=Path, default=severstal.DEFAULT_ROOT)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument(
        "--include-ksdd2", action="store_true",
        help="Add KSDD2 defectives as a 5th class (collapses Track C into B).",
    )
    args = parser.parse_args()
    print(json.dumps(
        build(
            splits_root=args.splits_root,
            output_dir=args.output_dir,
            severstal_root=args.severstal_root,
            min_area=args.min_area,
            include_ksdd2=args.include_ksdd2,
        ),
        indent=2,
    ))
