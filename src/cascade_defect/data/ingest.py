"""Download the NEU Metal Surface Defects dataset from Kaggle.

Uses the official `kaggle` Python package (wrapping the Kaggle CLI).
Reads credentials from `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars (loaded from .env).

Idempotent: skips download if `data/raw/neu/` already contains class folders.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "kaushal2896/neu-metal-surface-defects-data"
DEFAULT_RAW_DIR = Path("data/raw/neu")

# Public HuggingFace mirror of the NEU-CLS dataset (no auth needed, 1440 train + 360 test).
# Labels are 0..5 in the standard alphabetical NEU class order.
HF_MIRROR_REPO = "newguyme/neu_cls"
HF_MIRROR_FILES = (
    "data/train-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
)
NEU_LABELS = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

# Names the unpacked archive uses — we normalise to lowercase + underscores.
_KAGGLE_CLASS_RENAMES = {
    "Crazing": "crazing",
    "Inclusion": "inclusion",
    "Patches": "patches",
    "Pitted": "pitted_surface",
    "Rolled": "rolled-in_scale",
    "Scratches": "scratches",
}


def _ensure_kaggle_credentials() -> None:
    """Validate Kaggle creds exist in env. Supports both auth styles.

    New (2025+): single ``KAGGLE_API_TOKEN`` (token starts with ``KGAT_``).
    Old: ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` pair.
    """
    if os.getenv("KAGGLE_API_TOKEN"):
        return
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return
    raise RuntimeError(
        "Missing Kaggle credentials. Set KAGGLE_API_TOKEN (new style) OR "
        "KAGGLE_USERNAME + KAGGLE_KEY (legacy) in .env. "
        "Get a token at https://www.kaggle.com/settings."
    )


def _normalise_class_dirs(root: Path) -> None:
    """Flatten Kaggle's nested folder structure into <root>/<class_name>/*.jpg."""
    # The archive typically has structure: NEU Metal Surface Defects Data/{train,valid,test}/<Class>/*.jpg
    # We collapse train+valid+test into a single per-class folder (split.py re-splits).
    target_dirs = {v: root / v for v in set(_KAGGLE_CLASS_RENAMES.values())}
    for d in target_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for img in root.rglob("*.jpg"):
        # Find the class token in any path component.
        class_token = next(
            (
                _KAGGLE_CLASS_RENAMES[part]
                for part in img.parts
                if part in _KAGGLE_CLASS_RENAMES
            ),
            None,
        )
        if class_token is None:
            continue
        target = target_dirs[class_token] / img.name
        if img.resolve() != target.resolve():
            shutil.move(str(img), str(target))

    # Remove now-empty Kaggle folders (anything that isn't one of our target dirs).
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and path.name not in target_dirs:
            shutil.rmtree(path, ignore_errors=True)


def download_neu(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> Path:
    """Download + unpack the NEU dataset into ``raw_dir``."""
    raw_dir = Path(raw_dir)
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        logger.info("Dataset already present at %s — skipping download.", raw_dir)
        return raw_dir

    load_dotenv()
    _ensure_kaggle_credentials()

    # Import lazily so missing kaggle package only fails when this is actually called.
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-not-found]

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading %s → %s", KAGGLE_DATASET, raw_dir)
    api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), quiet=False, unzip=False)

    # Unzip whatever .zip the API dropped.
    for zip_path in raw_dir.glob("*.zip"):
        logger.info("Unzipping %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()

    _normalise_class_dirs(raw_dir)
    logger.info("Download complete. Class counts: %s", _class_counts(raw_dir))
    return raw_dir


def _class_counts(raw_dir: Path) -> dict[str, int]:
    return {p.name: len(list(p.glob("*.jpg"))) for p in raw_dir.iterdir() if p.is_dir()}


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic-data fallback (so the pipeline can be exercised without a Kaggle token)
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic(
    raw_dir: Path = DEFAULT_RAW_DIR,
    images_per_class: int = 50,
    image_size: int = 200,
    random_seed: int = 42,
) -> Path:
    """Generate a synthetic NEU-shaped dataset for end-to-end pipeline testing.

    Produces the same folder layout as the real Kaggle download
    (``raw_dir/<class>/img_NNN.jpg``) using cheap procedural noise so downstream
    code (split, upload, training smoke tests) can run without credentials.
    """
    import numpy as np
    from PIL import Image

    raw_dir = Path(raw_dir)
    rng = np.random.default_rng(random_seed)
    classes = list(set(_KAGGLE_CLASS_RENAMES.values()))

    for idx, class_name in enumerate(sorted(classes)):
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        # Class-specific texture seed so AE can later distinguish them.
        base_intensity = 80 + idx * 20
        for i in range(images_per_class):
            noise = rng.integers(0, 60, size=(image_size, image_size), dtype=np.uint8)
            arr = np.clip(noise.astype(int) + base_intensity, 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(class_dir / f"img_{i:03d}.jpg", quality=85)

    logger.info("Generated synthetic dataset → %s", _class_counts(raw_dir))
    return raw_dir


def download_neu_from_hf(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download NEU-CLS from the public HuggingFace mirror (no auth).

    Decodes the parquet rows into ``raw_dir/<class>/img_NNN.jpg`` so the
    output is shape-compatible with :func:`download_neu`.
    """
    import io
    import urllib.request

    import pyarrow.parquet as pq

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    counters = {cls: 0 for cls in NEU_LABELS}

    for fname in HF_MIRROR_FILES:
        url = f"https://huggingface.co/datasets/{HF_MIRROR_REPO}/resolve/main/{fname}"
        logger.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310
            blob = r.read()
        table = pq.ParquetFile(io.BytesIO(blob)).read()
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for img, lbl in zip(images, labels, strict=True):
            cls = NEU_LABELS[int(lbl)]
            class_dir = raw_dir / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            (class_dir / f"img_{counters[cls]:03d}.jpg").write_bytes(img["bytes"])
            counters[cls] += 1

    logger.info("HF mirror download complete. Class counts: %s", _class_counts(raw_dir))
    return raw_dir


def fetch_or_synthesize(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> Path:
    """Try real NEU sources, then fall back to synthetic.

    Order of preference:
    1. Public HuggingFace mirror (`newguyme/neu_cls`) — no auth, ~70 MB.
    2. Kaggle (`kaushal2896/neu-metal-surface-defects-data`) — needs creds + accepted terms.
    3. Synthetic procedural noise.
    """
    raw_dir = Path(raw_dir)
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        logger.info("Dataset already present at %s — skipping.", raw_dir)
        return raw_dir
    try:
        return download_neu_from_hf(raw_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("HF mirror unavailable (%s) — trying Kaggle.", e)
    try:
        return download_neu(raw_dir, force=force)
    except (RuntimeError, ImportError) as e:
        logger.warning("Kaggle download unavailable (%s) — generating synthetic dataset.", e)
        return generate_synthetic(raw_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_or_synthesize()
