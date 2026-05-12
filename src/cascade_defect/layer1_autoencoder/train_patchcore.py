"""Build per-domain PatchCore-lite memory banks (Phase J.1 modern path).

Reads `data/splits_metal/ae_train/` (union normals) and writes:

    models/patchcore_metal/
        bank_ksdd2.pt
        bank_severstal.pt
        summary.json     # per-domain (n_patches, dim, score_mean, score_std)

Calibration sets are taken from `data/splits_metal/ae_val/{ksdd2,severstal}/`
so the score distribution is honest (held-out normals, never seen during bank
construction).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from cascade_defect.layer1_autoencoder.patchcore import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_K,
    DEFAULT_QUANTILE,
    _FeatureExtractor,
    build_memory_bank,
    calibrate,
    save_bank,
)
from cascade_defect.layer1_autoencoder.scoring import infer_domain_from_filename

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/splits_metal")
DEFAULT_OUTPUT = Path("models/patchcore_metal")
DOMAINS = ("ksdd2", "severstal")


def _list_images(root: Path) -> list[Path]:
    return sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg"))


def main(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    backbone: str = "resnet18",
    image_size: int = DEFAULT_IMAGE_SIZE,
    bank_fraction: float = 0.10,
    k: int = DEFAULT_K,
    quantile: float = DEFAULT_QUANTILE,
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = _FeatureExtractor(backbone=backbone)

    train_paths = _list_images(data_dir / "ae_train")
    by_domain: dict[str, list[Path]] = {d: [] for d in DOMAINS}
    for p in train_paths:
        by_domain.setdefault(infer_domain_from_filename(p.name), []).append(p)

    banks: dict[str, torch.Tensor] = {}
    calibs = {}
    for d in DOMAINS:
        if not by_domain.get(d):
            logger.warning("No train images for domain %s — skipping bank build", d)
            continue
        logger.info("Building bank for %s (%d images)", d, len(by_domain[d]))
        banks[d] = build_memory_bank(
            extractor, by_domain[d], device=device, bank_fraction=bank_fraction
        )
        val_paths = _list_images(data_dir / "ae_val" / d)
        if not val_paths:
            logger.warning("No val normals for %s — calibration will be degenerate", d)
            from cascade_defect.layer1_autoencoder.patchcore import PatchCoreCalibration

            calibs[d] = PatchCoreCalibration(0.0, 1.0, 0)
        else:
            logger.info("Calibrating %s on %d held-out normals", d, len(val_paths))
            calibs[d] = calibrate(
                extractor, banks[d], val_paths, device=device, k=k, quantile=quantile
            )
        logger.info("Calibration %s: %s", d, calibs[d])

    save_bank(
        output_dir,
        bank_by_domain=banks,
        calibration_by_domain=calibs,
        backbone=extractor.backbone_name,
        image_size=image_size,
        k=k,
        quantile=quantile,
    )
    logger.info("PatchCore banks saved to %s", output_dir)
    return {d: {"n_patches": int(b.shape[0]), "dim": int(b.shape[1])} for d, b in banks.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "wrn50"],
        default="resnet18",
        help="PatchCore feature extractor backbone.",
    )
    parser.add_argument("--bank-fraction", type=float, default=0.10)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        image_size=args.image_size,
        bank_fraction=args.bank_fraction,
        k=args.k,
        quantile=args.quantile,
    )
