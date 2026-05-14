"""Phase 0b — Build the v2 Severstal split for the new "Classical CV vs VLM" story.

Reuses the existing ``data/splits_metal/`` partitioning (so we don't waste prior
training compute) and adds:

1. **Per-image class labels** for the *defective* images by joining against
   ``data/raw/severstal-steel-defect-detection/train.csv``. Severstal's 4
   classes (1..4) map to ``{1: pitting, 2: inclusion, 3: scratch, 4: patch}``.
   When an image carries multiple class IDs we keep the one with the largest
   total RLE area (most-prominent defect), matching how a quick human glance
   would label it.
2. A locked **VLM benchmark subset** of N images (default 240 = 120 normal + 120
   defective, stratified across the 4 classes) — small enough to keep
   benchmarking under a few dollars while large enough for stable 95 % CIs.

Outputs (all under ``data/splits_metal_v2/``):

* ``manifest.json``                    — counts + provenance
* ``train_labels.csv``                 — image_path,label,split  (resnet train pool)
* ``test_labels.csv``                  — image_path,label        (locked holdout)
* ``vlm_benchmark.csv``                — image_path,label        (subsampled holdout)

Run::

    uv run python scripts/build_split_v2.py
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_SPLITS = ROOT / "data/splits_metal"
RAW_CSV = ROOT / "data/raw/severstal-steel-defect-detection/train.csv"
OUT_DIR = ROOT / "data/splits_metal_v2"

CLASS_BY_ID = {1: "pitting", 2: "inclusion", 3: "scratch", 4: "patch"}
SEED = 42
BENCH_NORMALS = 120
BENCH_DEFECTIVES = 120  # 30 per class (4 × 30 = 120)


def _rle_area(encoded_pixels: str) -> int:
    """Sum of run lengths in a Severstal RLE string (every 2nd token)."""
    parts = encoded_pixels.strip().split()
    if not parts:
        return 0
    return sum(int(parts[i]) for i in range(1, len(parts), 2))


def load_dominant_class() -> dict[str, str]:
    """Return ``{image_id.jpg: class_name}`` keeping the largest-area defect."""
    by_image: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with RAW_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = int(row["ClassId"])
            area = _rle_area(row["EncodedPixels"])
            by_image[row["ImageId"]].append((cid, area))
    out: dict[str, str] = {}
    for img, items in by_image.items():
        items.sort(key=lambda t: t[1], reverse=True)
        out[img] = CLASS_BY_ID[items[0][0]]
    return out


def collect(folder: Path, label: str, glob: str = "*.jpg") -> list[tuple[Path, str]]:
    return [(p, label) for p in sorted(folder.glob(glob))]


def collect_prefix(folder: Path, prefix: str, label: str) -> list[tuple[Path, str]]:
    """Files in flat ae_train/ae_val are named ``<prefix>_<id>.<ext>``."""
    return [
        (p, label)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.name.startswith(prefix + "_")
    ]


def _strip_prefix(name: str) -> str:
    """yolo_train/yolo_val files are renamed ``severstal_<id>.jpg``; strip prefix."""
    return name.removeprefix("severstal_")


def relabel_defectives(items: list[tuple[Path, str]],
                       label_map: dict[str, str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    missing = 0
    for path, _ in items:
        key = _strip_prefix(path.name)
        cls = label_map.get(key)
        if cls is None:
            missing += 1
            continue
        out.append((path, cls))
    if missing:
        print(f"  ⚠ {missing} defective image(s) had no row in train.csv (dropped)")
    return out


def write_csv(path: Path, rows: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "label"])
        for p, lbl in rows:
            w.writerow([str(p.relative_to(ROOT)), lbl])


def stratified_sample(rows: list[tuple[Path, str]], n_per_class: dict[str, int],
                      rng: random.Random) -> list[tuple[Path, str]]:
    by_class: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for p, lbl in rows:
        by_class[lbl].append((p, lbl))
    picked: list[tuple[Path, str]] = []
    for cls, n in n_per_class.items():
        bucket = by_class.get(cls, [])
        if len(bucket) < n:
            print(f"  ⚠ class {cls!r}: only {len(bucket)} available, requested {n}")
            picked.extend(bucket)
        else:
            picked.extend(rng.sample(bucket, n))
    rng.shuffle(picked)
    return picked


def main() -> None:
    rng = random.Random(SEED)
    print(f"Loading per-image class labels from {RAW_CSV.relative_to(ROOT)} ...")
    label_map = load_dominant_class()
    print(f"  → {len(label_map):,} defective images labelled "
          f"({Counter(label_map.values())})")

    sev = SRC_SPLITS / "cascade_test/severstal"
    test_normal = collect(sev / "normal", "no_defect")
    test_defective_raw = collect(sev / "defective", "defect")
    test_defective = relabel_defectives(test_defective_raw, label_map)

    train_normal = collect_prefix(SRC_SPLITS / "ae_train", "severstal", "no_defect")
    # ae_val/ has subdirs (ksdd2/, severstal/); ae_train/ has flat prefixed files.
    val_normal = collect(SRC_SPLITS / "ae_val/severstal", "no_defect")
    train_defective_raw = (
        collect(SRC_SPLITS / "yolo_train", "defect")
        + collect(SRC_SPLITS / "yolo_val", "defect")
    )
    train_defective = relabel_defectives(train_defective_raw, label_map)

    # ── Train / val for ResNet50: keep AE val as the val split (already disjoint).
    train_rows = [
        *[(p, lbl) for p, lbl in train_normal],
        *[(p, lbl) for p, lbl in train_defective],
    ]
    val_rows = list(val_normal)
    # Carve off a small stratified val from the defective training pool too.
    by_cls: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for p, lbl in train_defective:
        by_cls[lbl].append((p, lbl))
    train_rows_filtered: list[tuple[Path, str]] = list(train_normal)
    for cls, items in by_cls.items():
        rng.shuffle(items)
        n_val = max(20, len(items) // 10)  # 10 % to val (min 20)
        val_rows.extend(items[:n_val])
        train_rows_filtered.extend(items[n_val:])
    rng.shuffle(train_rows_filtered)
    rng.shuffle(val_rows)

    train_labels: list[tuple[Path, str, str]] = (
        [(p, lbl, "train") for p, lbl in train_rows_filtered]
        + [(p, lbl, "val") for p, lbl in val_rows]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "train_labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path", "label", "split"])
        for p, lbl, sp in train_labels:
            w.writerow([str(p.relative_to(ROOT)), lbl, sp])

    test_rows = test_normal + test_defective
    rng.shuffle(test_rows)
    write_csv(OUT_DIR / "test_labels.csv", test_rows)

    # ── VLM benchmark subset ── stratified, deterministic, locked.
    n_per_defect = max(1, BENCH_DEFECTIVES // 4)
    bench_def = stratified_sample(
        test_defective,
        {cls: n_per_defect for cls in CLASS_BY_ID.values()},
        rng,
    )
    bench_norm = stratified_sample(
        test_normal, {"no_defect": BENCH_NORMALS}, rng,
    )
    bench_rows = bench_def + bench_norm
    rng.shuffle(bench_rows)
    write_csv(OUT_DIR / "vlm_benchmark.csv", bench_rows)

    counts = {
        "train": Counter(lbl for _, lbl, sp in train_labels if sp == "train"),
        "val": Counter(lbl for _, lbl, sp in train_labels if sp == "val"),
        "test": Counter(lbl for _, lbl in test_rows),
        "vlm_benchmark": Counter(lbl for _, lbl in bench_rows),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_manifest": "data/splits_metal/manifest.json",
        "source_csv": "data/raw/severstal-steel-defect-detection/train.csv",
        "label_strategy": "Severstal class IDs 1-4 → "
                          "{pitting, inclusion, scratch, patch}; for "
                          "multi-class images keep the class with the largest "
                          "total RLE area.",
        "classes": ["no_defect", "pitting", "inclusion", "scratch", "patch"],
        "counts": {k: dict(v) for k, v in counts.items()},
        "vlm_benchmark_target": {
            "no_defect": BENCH_NORMALS,
            **{c: n_per_defect for c in CLASS_BY_ID.values()},
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== v2 split written ===")
    for split, cnt in counts.items():
        total = sum(cnt.values())
        print(f"  {split:14s} n={total:5d} {dict(cnt)}")
    print(f"\nManifest: {OUT_DIR.relative_to(ROOT) / 'manifest.json'}")


if __name__ == "__main__":
    main()
