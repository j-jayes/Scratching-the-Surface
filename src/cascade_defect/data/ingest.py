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
    """Validate Kaggle creds exist in env. Raise with a clear message if not."""
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise RuntimeError(
            "Missing Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY in .env "
            "(token from https://www.kaggle.com/settings → 'Create New API Token')."
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_neu()
