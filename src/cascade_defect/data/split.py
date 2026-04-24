"""Stratified train/test split utility for the NEU Metal Surface Defects dataset.

Split strategy
--------------
- Few-Shot Seed (1%)  : 3 images × 6 classes = 18 images (labels kept for GPT-4o prompt)
- Unlabelled Pool (79%): ~1,420 images       (labels stripped → pseudo-labelled by GPT-4o)
- Golden Test Set (20%): 360 images           (ground-truth labels, never seen during training)
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SEED_PER_CLASS = 3
TEST_FRACTION = 0.20
RANDOM_SEED = 42

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def split_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    seed_per_class: int = SEED_PER_CLASS,
    test_fraction: float = TEST_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> dict[str, int]:
    """Split the NEU dataset into seed / unlabelled / test subsets.

    Parameters
    ----------
    raw_dir:
        Directory containing one sub-folder per class (e.g. ``data/raw/NEU-DET/images/``).
    output_dir:
        Root output directory. Creates ``seed/``, ``unlabelled/``, and ``test/`` sub-folders.
    seed_per_class:
        Number of labelled examples to keep per class for the few-shot seed.
    test_fraction:
        Fraction of the dataset to hold out as the golden test set.
    random_seed:
        RNG seed for reproducibility.

    Returns
    -------
    dict mapping split name → image count.
    """
    rng = random.Random(random_seed)

    splits: dict[str, list[tuple[Path, str]]] = {"seed": [], "unlabelled": [], "test": []}

    for class_name in NEU_CLASSES:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            logger.warning("Class directory not found: %s", class_dir)
            continue

        images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        if not images:
            logger.warning("No images found in %s", class_dir)
            continue

        rng.shuffle(images)

        n_test = max(1, round(len(images) * test_fraction))
        test_imgs = images[:n_test]
        remaining = images[n_test:]

        seed_imgs = remaining[:seed_per_class]
        unlabelled_imgs = remaining[seed_per_class:]

        splits["test"].extend((p, class_name) for p in test_imgs)
        splits["seed"].extend((p, class_name) for p in seed_imgs)
        splits["unlabelled"].extend((p, class_name) for p in unlabelled_imgs)

    # Copy files to output directories
    for split_name, items in splits.items():
        for img_path, class_name in items:
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)

    counts = {k: len(v) for k, v in splits.items()}
    logger.info(
        "Split complete — seed: %d, unlabelled: %d, test: %d",
        counts["seed"],
        counts["unlabelled"],
        counts["test"],
    )

    # Persist a manifest describing every split → useful for reproducibility and
    # for the GPT-4o annotator (it needs class labels for the seed images only).
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "seed_per_class": seed_per_class,
        "test_fraction": test_fraction,
        "counts": counts,
        "splits": {
            split_name: [
                {"path": (output_dir / split_name / cls / p.name).as_posix(), "class": cls}
                for p, cls in items
            ]
            for split_name, items in splits.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to %s", manifest_path)

    return counts


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Split NEU dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/neu"), help="Path to raw NEU dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"), help="Output directory")
    args = parser.parse_args()

    counts = split_dataset(args.raw_dir, args.output_dir)
    print(counts)
