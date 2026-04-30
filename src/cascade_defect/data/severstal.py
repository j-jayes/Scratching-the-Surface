"""Severstal Steel Defect Detection dataset loader.

Source: https://www.kaggle.com/competitions/severstal-steel-defect-detection
The Kaggle competition T&Cs apply (research/portfolio use only — no commercial
redistribution of imagery).

Expected layout on disk (you must download manually after accepting the
competition rules)::

    data/raw/severstal/
        train.csv               # ImageId, ClassId, EncodedPixels (RLE)
        train_images/
            0002cc93b.jpg
            ...
        test_images/            # (optional — competition test set has no labels)

`train.csv` is *long-format*: one row per (image, class) defect run-length, so an
image with no defect rows in the CSV is implicitly defect-free. A single image
may also have multiple defect classes — we aggregate per-image.

There are 4 defect classes (1, 2, 3, 4 in the original competition). The
competition never released human-readable names; we use these working labels:

    1 → "pitting"
    2 → "inclusion"
    3 → "scratch"
    4 → "patch"
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("data/raw/severstal-steel-defect-detection")
IMAGE_DIR_NAME = "train_images"
CSV_NAME = "train.csv"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600

CLASS_NAMES: dict[int, str] = {1: "pitting", 2: "inclusion", 3: "scratch", 4: "patch"}


@dataclass(frozen=True, slots=True)
class SeverstalSample:
    image_path: Path
    image_id: str
    defective: bool
    class_ids: tuple[int, ...]
    rle_by_class: dict[int, str] = field(default_factory=dict)


def _read_train_csv(csv_path: Path) -> dict[str, dict[int, str]]:
    """Return ``{image_id: {class_id: rle_string}}``.

    Tolerates both the original 3-column schema (``ImageId, ClassId,
    EncodedPixels``) and the alternative wide schema sometimes seen on Kaggle
    mirrors (``ImageId_ClassId, EncodedPixels``).
    """
    import csv as _csv

    out: dict[str, dict[int, str]] = {}
    with csv_path.open(newline="") as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        if header == ["ImageId_ClassId", "EncodedPixels"]:
            for row in reader:
                key, rle = row[0], row[1]
                if not rle.strip():
                    continue
                image_id, class_str = key.rsplit("_", 1)
                out.setdefault(image_id, {})[int(class_str)] = rle
        else:
            # Assume new 3-column schema. Be lenient about column order.
            cols = {name: i for i, name in enumerate(header)}
            i_img, i_cls, i_rle = cols["ImageId"], cols["ClassId"], cols["EncodedPixels"]
            for row in reader:
                rle = row[i_rle]
                if not rle.strip():
                    continue
                out.setdefault(row[i_img], {})[int(row[i_cls])] = rle
    return out


def iter_samples(root: Path = DEFAULT_ROOT) -> Iterator[SeverstalSample]:
    """Yield every image in ``train_images/`` with its (possibly empty) RLE map."""
    image_dir = root / IMAGE_DIR_NAME
    csv_path = root / CSV_NAME
    if not image_dir.exists() or not csv_path.exists():
        return

    rle_index = _read_train_csv(csv_path)
    for img in sorted(image_dir.glob("*.jpg")):
        defects = rle_index.get(img.name, {})
        yield SeverstalSample(
            image_path=img,
            image_id=img.name,
            defective=bool(defects),
            class_ids=tuple(sorted(defects.keys())),
            rle_by_class=defects,
        )


def iter_normal(root: Path = DEFAULT_ROOT) -> Iterator[SeverstalSample]:
    return (s for s in iter_samples(root) if not s.defective)


def iter_defective(root: Path = DEFAULT_ROOT) -> Iterator[SeverstalSample]:
    return (s for s in iter_samples(root) if s.defective)


def rle_to_mask(
    rle: str, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH
) -> np.ndarray:
    """Decode the Severstal *column-major* RLE string into a binary mask.

    Severstal uses 1-indexed pixel positions, column-major order — different from
    COCO. Returns ``uint8`` ``(height, width)``.
    """
    s = rle.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    flat = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths, strict=True):
        flat[start : start + length] = 1
    # Column-major reshape, then transpose to (H, W).
    return flat.reshape((width, height)).T


def mask_to_yolo_bboxes(
    mask: np.ndarray, *, min_area: int = 16
) -> list[tuple[float, float, float, float]]:
    """Convert a binary mask to YOLO-format bboxes (x_centre, y_centre, w, h),
    all normalised to [0, 1]. Connected components below ``min_area`` are dropped.
    """
    import cv2  # local import — keeps loader importable without cv2 installed

    h, w = mask.shape
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    bboxes: list[tuple[float, float, float, float]] = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        bboxes.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return bboxes


def summarise(root: Path = DEFAULT_ROOT) -> dict[str, int]:
    normals = defects = 0
    per_class = {c: 0 for c in CLASS_NAMES}
    for s in iter_samples(root):
        if s.defective:
            defects += 1
            for c in s.class_ids:
                per_class[c] = per_class.get(c, 0) + 1
        else:
            normals += 1
    return {
        "normal": normals,
        "defective": defects,
        "total": normals + defects,
        **{f"class_{c}_{CLASS_NAMES[c]}": per_class[c] for c in sorted(per_class)},
    }


if __name__ == "__main__":
    import json

    print(json.dumps(summarise(), indent=2))
