"""Train YOLOv8 on the Severstal (\u00b1 KSDD2) dataset for Phase J / J.2.

Defaults are CPU-friendly so the trainer runs end-to-end in CI / on a laptop;
the script also auto-detects CUDA and bumps batch size when available.

Class imbalance handling
========================
Severstal class ``scratch`` outnumbers ``inclusion`` ~20\u00d7 and the YOLOv8
default loss is a plain BCE. We push two levers:

1. **Per-class oversampling** of the train manifest: minority classes are
   duplicated in ``images/train/`` listings via a generated ``manifest.txt``,
   then the YAML override ``train: manifest.txt`` lets Ultralytics consume it.
   Default oversampling ratio brings every class to within 2× of the largest.
2. **Focal-loss gamma** (``--fl-gamma``) is recorded in ``summary.json`` for
   provenance but is no-op against modern Ultralytics releases that removed
   the ``fl_gamma`` train arg — the lever now lives inside the loss config.
   Oversampling alone is the carrying mitigation.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("models/yolo_metal/dataset")
DEFAULT_OUTPUT = Path("models/yolo_metal")
DEFAULT_BASE_WEIGHTS = "yolov8n.pt"


def _read_label_classes(label_path: Path) -> set[int]:
    if not label_path.exists():
        return set()
    out: set[int] = set()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            out.add(int(parts[0]))
    return out


def _build_oversampled_manifest(
    dataset_dir: Path, *, ratio_cap: float = 2.0
) -> tuple[Path, dict[int, int]]:
    """Produce a manifest where minority classes are duplicated so the largest
    class is at most ``ratio_cap``\u00d7 the smallest.

    Returns (manifest_path, counts_after).
    """
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"

    image_classes: dict[Path, set[int]] = {}
    counts_before: Counter = Counter()
    for img in sorted(images_train.glob("*.jpg")) + sorted(images_train.glob("*.png")):
        classes = _read_label_classes(labels_train / f"{img.stem}.txt")
        image_classes[img] = classes
        for c in classes:
            counts_before[c] += 1
    if not counts_before:
        raise RuntimeError(f"No labels found under {labels_train}")

    max_count = max(counts_before.values())
    min_count = min(counts_before.values())
    target = max(int(max_count / ratio_cap), min_count)
    multipliers = {c: max(1, target // max(1, n)) for c, n in counts_before.items()}
    logger.info("Class counts before: %s, multipliers: %s", dict(counts_before), multipliers)

    manifest = dataset_dir / "manifest_train.txt"
    counts_after: Counter = Counter(counts_before)
    with manifest.open("w") as fh:
        for img, classes in image_classes.items():
            # Multiplier = max over classes the image contributes to.
            mult = max((multipliers[c] for c in classes), default=1)
            for _ in range(mult):
                fh.write(str(img.resolve().as_posix()) + "\n")
            for c in classes:
                counts_after[c] += (mult - 1)
    logger.info("Counts after oversampling: %s", dict(counts_after))
    return manifest, dict(counts_after)


def _patch_data_yaml(dataset_dir: Path, manifest: Path) -> Path:
    """Write a sibling YAML that points train at the oversampled manifest."""
    src = (dataset_dir / "data.yaml").read_text()
    new = []
    for line in src.splitlines():
        if line.startswith("train:"):
            new.append(f"train: {manifest.resolve().as_posix()}")
        else:
            new.append(line)
    out = dataset_dir / "data_oversampled.yaml"
    out.write_text("\n".join(new) + "\n")
    return out


def train(
    *,
    dataset_dir: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT,
    base_weights: str = DEFAULT_BASE_WEIGHTS,
    epochs: int = 50,
    image_size: int = 640,
    batch_size: int | None = None,
    fl_gamma: float = 1.5,
    use_oversampling: bool = True,
    oversample_ratio_cap: float = 2.0,
    device: str = "",
    workers: int = 0,
    cache: str = "ram",
) -> dict:
    import ultralytics.data.dataset as _ud
    import ultralytics.data.utils as _udu

    # Containers ship with a 64 MiB ``/dev/shm`` by default, which is *not*
    # enough room for the ``Pool(NUM_THREADS=7)`` SemLock allocations Ultralytics
    # uses inside ``cache_labels`` and image caching on an 8-vCPU GPU profile.
    # ACA does not expose ``--shm-size``, so we cap the pool size from Python
    # before any data path code touches the constant.
    import ultralytics.utils as _uu
    from ultralytics import YOLO  # local import — heavy
    from ultralytics.utils import SETTINGS
    _uu.NUM_THREADS = max(1, min(max(workers, 2), _uu.NUM_THREADS))
    _ud.NUM_THREADS = _uu.NUM_THREADS
    if hasattr(_udu, "NUM_THREADS"):
        _udu.NUM_THREADS = _uu.NUM_THREADS
    logger.info("Capped Ultralytics NUM_THREADS=%d (avoid /dev/shm OSError 28)", _uu.NUM_THREADS)

    # Even ``ThreadPool(2)`` allocates SemLock objects in ``/dev/shm`` because
    # ``multiprocessing.pool.ThreadPool`` inherits from ``Pool`` and constructs
    # a ``SimpleQueue`` regardless of whether workers are threads. Replace it
    # with a ``concurrent.futures.ThreadPoolExecutor``-backed shim that uses
    # plain ``threading.Lock`` (heap-allocated, no shm).
    import concurrent.futures as _cf
    import multiprocessing.pool as _mpp

    class _ThreadPoolShim:
        def __init__(self, processes=None, *args, **kwargs):
            self._exec = _cf.ThreadPoolExecutor(max_workers=processes or _uu.NUM_THREADS)
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            self._exec.shutdown(wait=True)
        def imap(self, fn, iterable, chunksize=1):
            return self._exec.map(fn, iterable)
        def imap_unordered(self, fn, iterable, chunksize=1):
            return self._exec.map(fn, iterable)
        def map(self, fn, iterable, chunksize=1):
            return list(self._exec.map(fn, iterable))
        def close(self):
            pass
        def join(self):
            self._exec.shutdown(wait=True)
        def terminate(self):
            self._exec.shutdown(wait=False)

    _mpp.ThreadPool = _ThreadPoolShim
    logger.info("Patched multiprocessing.pool.ThreadPool -> ThreadPoolExecutor shim")

    # Ultralytics auto-loads any optional integration whose package is
    # importable (mlflow, comet, wandb…) and will crash the run if a stale
    # tracking URI is leaking in from another project on the same machine.
    # Force the cascade trainer to be self-contained.
    for integration in ("mlflow", "comet", "wandb", "dvc", "neptune", "tensorboard", "clearml"):
        if integration in SETTINGS:
            SETTINGS[integration] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(
            f"{dataset_dir / 'data.yaml'} missing \u2014 run "
            f"`python -m cascade_defect.data.severstal_yolo` first."
        )

    if use_oversampling:
        manifest, counts_after = _build_oversampled_manifest(
            dataset_dir, ratio_cap=oversample_ratio_cap
        )
        data_yaml = _patch_data_yaml(dataset_dir, manifest)
    else:
        data_yaml = dataset_dir / "data.yaml"
        counts_after = json.loads((dataset_dir / "stats.json").read_text())["train_counts"]

    if batch_size is None:
        # Conservative: 4 on CPU, 16 on GPU. Ultralytics tunes if -1.
        try:
            import torch
            batch_size = 16 if torch.cuda.is_available() else 4
        except ImportError:
            batch_size = 4

    model = YOLO(base_weights)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device or None,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        cache=cache,
        workers=workers,
        verbose=True,
    )
    # Newer Ultralytics releases removed ``fl_gamma`` from the train CLI; pass
    # it only if the installed version still accepts it.
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT  # type: ignore
        if "fl_gamma" in DEFAULT_CFG_DICT:
            train_kwargs["fl_gamma"] = fl_gamma
        else:
            logger.info("fl_gamma=%.2f recorded for provenance only (not a train arg in this Ultralytics version)", fl_gamma)
    except ImportError:
        pass
    results = model.train(**train_kwargs)

    # Persist the *best* weights at the canonical inference path.
    src_best = Path(results.save_dir) / "weights" / "best.pt"
    if src_best.exists():
        target = output_dir / "best.pt"
        target.write_bytes(src_best.read_bytes())

    # Evaluate on the val split for the headline numbers.
    val_metrics = model.val(data=str(data_yaml), imgsz=image_size, device=device or None)
    summary = {
        "base_weights": base_weights,
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "fl_gamma": fl_gamma,
        "use_oversampling": use_oversampling,
        "oversample_ratio_cap": oversample_ratio_cap,
        "train_counts_after": counts_after,
        "val_metrics": {
            "map50": float(getattr(val_metrics.box, "map50", 0.0)),
            "map50_95": float(getattr(val_metrics.box, "map", 0.0)),
            "precision": float(getattr(val_metrics.box, "mp", 0.0)),
            "recall": float(getattr(val_metrics.box, "mr", 0.0)),
        },
        "checkpoint": str(output_dir / "best.pt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Training complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-weights", default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fl-gamma", type=float, default=1.5)
    parser.add_argument("--no-oversampling", action="store_true")
    parser.add_argument("--oversample-ratio-cap", type=float, default=2.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers. 0 (default) avoids /dev/shm in containers; cache='ram' compensates.")
    parser.add_argument("--cache", default="ram",
                        help="Ultralytics cache: 'ram' (best for small datasets on slow shares), 'disk', or '' to disable.")
    args = parser.parse_args()
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        base_weights=args.base_weights,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        fl_gamma=args.fl_gamma,
        use_oversampling=not args.no_oversampling,
        oversample_ratio_cap=args.oversample_ratio_cap,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
    )
