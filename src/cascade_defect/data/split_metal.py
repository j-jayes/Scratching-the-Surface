"""Build the unified split layout for the metal-surface refit (Phase J).

Output layout under ``data/splits_metal/`` (paths copied — keeps Quarto happy
across OS without symlink permissions)::

    ae_train/                # union of KSDD2 + Severstal normals (AE training)
    ae_val/
        ksdd2/               # held-out normals → derive τ_ksdd2
        severstal/           # held-out normals → derive τ_severstal
    yolo_train/              # Severstal defectives only (80%)
    yolo_val/                # Severstal defectives only (20%)
    cascade_test/
        severstal/
            normal/          # held-out Severstal negatives (Track A)
            defective/       # held-out Severstal positives (Track A)
        ksdd2/
            normal/          # KSDD2 test normals (Track B/C)
            defective/       # KSDD2 test defectives (Track B/C — true OOD for YOLO)
    manifest.json

Counts are deterministic given ``--seed`` and the on-disk dataset state.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from cascade_defect.data import ksdd2, severstal

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/splits_metal")
RANDOM_SEED = 42
AE_VAL_FRACTION = 0.10  # held-out normals per domain for τ derivation
SEVERSTAL_TEST_FRACTION = 0.20  # held-out for cascade Track A (both polarities)
YOLO_VAL_FRACTION = 0.20

# Group-key extractors (for J.2 "held-out by image source" splits).
# For KSDD2 the filename is e.g. ``10000.png`` — sequential frames from the
# same production run share their leading digits. Grouping by the first 3
# digits keeps adjacent frames out of opposite splits. For Severstal the
# competition publishes randomised hashes with no coil info; we fall back to
# the first 2 hex chars (256 buckets), which gives a coarse but
# *deterministic* grouping that still beats random sampling at the margin.


def _ksdd2_group_key(p: Path) -> str:
    return p.stem[:3]


def _severstal_group_key(p: Path) -> str:
    return p.stem[:2]


def _grouped_split(
    items: list, key_fn, *, hold_out_fraction: float, rng: random.Random
) -> tuple[list, list]:
    """Assign whole groups to (held_out, rest) splits. Determinism by ``rng``.

    Items must expose ``image_path``. Groups are sized roughly evenly so the
    realised hold-out fraction is within ± 1 / n_groups of the requested one.
    """
    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(key_fn(it.image_path), []).append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    target = int(len(items) * hold_out_fraction)
    held: list = []
    rest: list = []
    for k in keys:
        bucket = held if len(held) < target else rest
        bucket.extend(groups[k])
    return held, rest


def _copy(src: Path, dest_dir: Path, prefix: str = "") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{prefix}{src.name}"
    # Resume-friendly: skip if already copied with matching size.
    try:
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            return dest
    except OSError:
        pass
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            shutil.copy2(src, dest)
            return dest
        except PermissionError as exc:  # transient AV/indexer locks on Windows
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_err  # type: ignore[misc]


def build_splits(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    ksdd2_root: Path = ksdd2.DEFAULT_ROOT,
    severstal_root: Path = severstal.DEFAULT_ROOT,
    seed: int = RANDOM_SEED,
) -> dict:
    rng = random.Random(seed)
    counts: dict[str, int] = {}

    # ---- KSDD2 ------------------------------------------------------------
    ksdd2_train_normals = list(ksdd2.iter_normal(ksdd2_root, "train"))
    ksdd2_ae_val, ksdd2_ae_train = _grouped_split(
        ksdd2_train_normals,
        _ksdd2_group_key,
        hold_out_fraction=AE_VAL_FRACTION,
        rng=rng,
    )

    ksdd2_test_normals = list(ksdd2.iter_normal(ksdd2_root, "test"))
    ksdd2_test_defects = list(ksdd2.iter_defective(ksdd2_root, "test"))

    counts["ksdd2_ae_train"] = len(ksdd2_ae_train)
    counts["ksdd2_ae_val"] = len(ksdd2_ae_val)
    counts["ksdd2_cascade_test_normal"] = len(ksdd2_test_normals)
    counts["ksdd2_cascade_test_defective"] = len(ksdd2_test_defects)

    # ---- Severstal --------------------------------------------------------
    severstal_normals = list(severstal.iter_normal(severstal_root))
    severstal_defects = list(severstal.iter_defective(severstal_root))

    if not severstal_normals and not severstal_defects:
        logger.warning(
            "No Severstal data under %s — skipping Severstal portion of the split. "
            "Drop the kaggle archive in place and re-run.",
            severstal_root,
        )
        sev_ae_train = sev_ae_val = sev_test_normal = sev_test_defect = []
        sev_yolo_train = sev_yolo_val = []
    else:
        # Group-aware: ImageId prefix — cosmetic for Severstal (random hashes)
        # but still deterministic and beats pure random shuffle.
        sev_test_normal, rest_normals = _grouped_split(
            severstal_normals, _severstal_group_key,
            hold_out_fraction=SEVERSTAL_TEST_FRACTION, rng=rng,
        )
        sev_ae_val, sev_ae_train = _grouped_split(
            rest_normals, _severstal_group_key,
            hold_out_fraction=AE_VAL_FRACTION, rng=rng,
        )

        sev_test_defect, rest_defects = _grouped_split(
            severstal_defects, _severstal_group_key,
            hold_out_fraction=SEVERSTAL_TEST_FRACTION, rng=rng,
        )
        sev_yolo_val, sev_yolo_train = _grouped_split(
            rest_defects, _severstal_group_key,
            hold_out_fraction=YOLO_VAL_FRACTION, rng=rng,
        )

    counts["severstal_ae_train"] = len(sev_ae_train)
    counts["severstal_ae_val"] = len(sev_ae_val)
    counts["severstal_cascade_test_normal"] = len(sev_test_normal)
    counts["severstal_cascade_test_defective"] = len(sev_test_defect)
    counts["severstal_yolo_train"] = len(sev_yolo_train)
    counts["severstal_yolo_val"] = len(sev_yolo_val)

    counts["ae_train_total"] = counts["ksdd2_ae_train"] + counts["severstal_ae_train"]

    # ---- Materialise on disk ---------------------------------------------
    # Resume-friendly: leave existing files in place; _copy skips matching ones.
    output_dir.mkdir(parents=True, exist_ok=True)
    ae_train_dir = output_dir / "ae_train"
    for sample in ksdd2_ae_train:
        _copy(sample.image_path, ae_train_dir, prefix="ksdd2_")
    for sample in sev_ae_train:
        _copy(sample.image_path, ae_train_dir, prefix="severstal_")

    for sample in ksdd2_ae_val:
        _copy(sample.image_path, output_dir / "ae_val" / "ksdd2", prefix="ksdd2_")
    for sample in sev_ae_val:
        _copy(sample.image_path, output_dir / "ae_val" / "severstal", prefix="severstal_")

    for sample in sev_yolo_train:
        _copy(sample.image_path, output_dir / "yolo_train", prefix="severstal_")
    for sample in sev_yolo_val:
        _copy(sample.image_path, output_dir / "yolo_val", prefix="severstal_")

    for sample in sev_test_normal:
        _copy(sample.image_path, output_dir / "cascade_test" / "severstal" / "normal")
    for sample in sev_test_defect:
        _copy(sample.image_path, output_dir / "cascade_test" / "severstal" / "defective")
    for sample in ksdd2_test_normals:
        _copy(sample.image_path, output_dir / "cascade_test" / "ksdd2" / "normal")
    for sample in ksdd2_test_defects:
        _copy(sample.image_path, output_dir / "cascade_test" / "ksdd2" / "defective")

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "ae_val_fraction": AE_VAL_FRACTION,
        "severstal_test_fraction": SEVERSTAL_TEST_FRACTION,
        "yolo_val_fraction": YOLO_VAL_FRACTION,
        "split_strategy": "group_aware_by_image_id_prefix",
        "counts": counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Split complete: %s", counts)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build metal-surface unified split")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ksdd2-root", type=Path, default=ksdd2.DEFAULT_ROOT)
    parser.add_argument("--severstal-root", type=Path, default=severstal.DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print(json.dumps(build_splits(
        output_dir=args.output_dir,
        ksdd2_root=args.ksdd2_root,
        severstal_root=args.severstal_root,
        seed=args.seed,
    ), indent=2))
