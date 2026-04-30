"""Kolektor Surface-Defect Dataset 2 (KSDD2) loader.

Layout on disk (`data/raw/KolektorSDD2/`)::

    train/
        10000.png       # input
        10000_GT.png    # binary mask; empty → defect-free
        ...
    test/
        20000.png
        20000_GT.png
        ...

A frame is *defective* iff its companion `_GT.png` mask has any non-zero pixel.
KSDD2 ships ~3,335 images: ~2,085 train + 1,250 test, with the *vast majority*
being defect-free (~89%). This is exactly the regime the cascade architecture
needs.

Note on licence: KSDD2 is **CC BY-NC-SA 4.0** — research / portfolio use only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ROOT = Path("data/raw/KolektorSDD2")
SPLITS = ("train", "test")


@dataclass(frozen=True, slots=True)
class KsddSample:
    image_path: Path
    mask_path: Path
    split: str  # "train" | "test"
    defective: bool


def _mask_is_defective(mask_path: Path) -> bool:
    """A mask counts as defective iff at least one pixel is non-zero."""
    with Image.open(mask_path) as im:
        return bool(np.any(np.asarray(im)))


def iter_samples(root: Path = DEFAULT_ROOT, split: str | None = None) -> Iterator[KsddSample]:
    """Yield every (image, mask, defective) triple in the requested split(s).

    Parameters
    ----------
    root: dataset root (containing ``train/`` and ``test/``).
    split: ``"train"``, ``"test"``, or ``None`` for both.
    """
    splits = SPLITS if split is None else (split,)
    for s in splits:
        split_dir = root / s
        if not split_dir.exists():
            continue
        for img in sorted(split_dir.glob("[0-9]*.png")):
            mask = img.with_name(f"{img.stem}_GT.png")
            if not mask.exists():
                continue
            yield KsddSample(
                image_path=img,
                mask_path=mask,
                split=s,
                defective=_mask_is_defective(mask),
            )


def iter_normal(root: Path = DEFAULT_ROOT, split: str | None = None) -> Iterator[KsddSample]:
    return (s for s in iter_samples(root, split) if not s.defective)


def iter_defective(root: Path = DEFAULT_ROOT, split: str | None = None) -> Iterator[KsddSample]:
    return (s for s in iter_samples(root, split) if s.defective)


def summarise(root: Path = DEFAULT_ROOT) -> dict[str, dict[str, int]]:
    """Return ``{split: {"normal": N, "defective": M, "total": N+M}}``."""
    out: dict[str, dict[str, int]] = {}
    for s in SPLITS:
        normals = defects = 0
        for sample in iter_samples(root, s):
            if sample.defective:
                defects += 1
            else:
                normals += 1
        out[s] = {"normal": normals, "defective": defects, "total": normals + defects}
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(summarise(), indent=2))
