"""Compare Layer-1 anomaly scorers — baseline AE (z-score) vs PatchCore-lite.

Outputs:
    reports/ae_metal_sanity.json     — both methods × both domains × both polarities
    reports/ae_metal_sanity.png      — 2×2 histogram grid

The website's evaluation page reads this file to render the "baseline → modern"
contrast for Phase J.1.

Usage::

    uv run python scripts/ae_metal_sanity.py
    uv run python scripts/ae_metal_sanity.py --skip-patchcore   # AE only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image

from cascade_defect.data import ksdd2, severstal
from cascade_defect.layer1_autoencoder.model import ConvAutoencoder
from cascade_defect.layer1_autoencoder.scoring import (
    load_calibration,
    make_transform,
    score_tensor,
)

logger = logging.getLogger(__name__)

AE_DIR = Path("models/autoencoder_metal")
PATCHCORE_DIR = Path("models/patchcore_metal")
DOMAINS = ("ksdd2", "severstal")


def _ae_scores(
    model: ConvAutoencoder,
    paths: list[Path],
    *,
    domain: str,
    image_size: int,
    device: str,
    calibration: dict,
) -> list[float]:
    stats = calibration.get(domain)
    tf = make_transform(image_size, stats)
    out: list[float] = []
    for p in paths:
        try:
            t = tf(Image.open(p).convert("RGB")).to(device)
        except OSError:
            continue
        result = score_tensor(model, t, stats=stats, domain=domain)
        out.append(result.z_score if stats is not None else result.raw_score)
    return out


def _patchcore_scores(extractor, bank, paths: list[Path], *, calib, device: str) -> list[float]:
    from cascade_defect.layer1_autoencoder.patchcore import score_image

    out: list[float] = []
    for p in paths:
        try:
            img = Image.open(p)
        except OSError:
            continue
        raw = score_image(extractor, bank, img, device=device)
        # Z-score against held-out normal distribution → directly comparable
        # to the AE z-scores above.
        z = (raw - calib.score_mean) / max(calib.score_std, 1e-9)
        out.append(z)
    return out


def _summary_block(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    t = torch.tensor(values)
    return {
        "n": len(values),
        "mean": round(float(t.mean()), 6),
        "std": round(float(t.std()), 6),
        "min": round(float(t.min()), 6),
        "max": round(float(t.max()), 6),
    }


def main(
    *,
    image_size: int = 256,
    sample: int = 200,
    skip_patchcore: bool = False,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples: dict[str, dict[str, list[Path]]] = {
        "ksdd2": {
            "normal": [s.image_path for s in ksdd2.iter_normal(split="test")][:sample],
            "defective": [s.image_path for s in ksdd2.iter_defective(split="test")][:sample],
        },
        "severstal": {
            "normal": [s.image_path for s in severstal.iter_normal()][:sample],
            "defective": [s.image_path for s in severstal.iter_defective()][:sample],
        },
    }

    summary: dict = {"ae": {}, "patchcore": {}}
    fig_data: dict = {"ae": {}, "patchcore": {}}

    # ── AE baseline ──────────────────────────────────────────────────────
    ae_ckpt = AE_DIR / "best.pt"
    if ae_ckpt.exists():
        model = ConvAutoencoder().to(device)
        model.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=True))
        calib_path = AE_DIR / "calibration.json"
        calibration = load_calibration(calib_path) if calib_path.exists() else {}
        for d in DOMAINS:
            summary["ae"][d] = {}
            fig_data["ae"][d] = {}
            for polarity, paths in samples[d].items():
                if not paths:
                    continue
                vals = _ae_scores(
                    model, paths, domain=d, image_size=image_size, device=device,
                    calibration=calibration,
                )
                fig_data["ae"][d][polarity] = vals
                summary["ae"][d][polarity] = _summary_block(vals)
    else:
        logger.warning("No AE checkpoint at %s — skipping AE block", ae_ckpt)

    # ── PatchCore-lite ───────────────────────────────────────────────────
    pc_summary = PATCHCORE_DIR / "summary.json"
    if not skip_patchcore and pc_summary.exists():
        from cascade_defect.layer1_autoencoder.patchcore import _FeatureExtractor, load_bank

        extractor = _FeatureExtractor().to(device)
        for d in DOMAINS:
            try:
                bank, calib = load_bank(PATCHCORE_DIR, d)
            except FileNotFoundError:
                logger.warning("No PatchCore bank for %s", d)
                continue
            summary["patchcore"][d] = {}
            fig_data["patchcore"][d] = {}
            for polarity, paths in samples[d].items():
                if not paths:
                    continue
                vals = _patchcore_scores(extractor, bank, paths, calib=calib, device=device)
                fig_data["patchcore"][d][polarity] = vals
                summary["patchcore"][d][polarity] = _summary_block(vals)
    elif not skip_patchcore:
        logger.warning("No PatchCore summary at %s — skipping PatchCore block", pc_summary)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    (reports / "ae_metal_sanity.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib.pyplot as plt

        methods = [m for m in ("ae", "patchcore") if fig_data[m]]
        if methods:
            fig, axes = plt.subplots(
                len(methods), len(DOMAINS),
                figsize=(6 * len(DOMAINS), 4 * len(methods)),
                squeeze=False,
            )
            for r, m in enumerate(methods):
                for c, d in enumerate(DOMAINS):
                    ax = axes[r][c]
                    block = fig_data[m].get(d, {})
                    for polarity, vals in block.items():
                        if vals:
                            ax.hist(vals, bins=30, alpha=0.6, label=polarity)
                    ax.set_title(f"{m} — {d}")
                    ax.set_xlabel("z-score")
                    ax.legend()
            plt.tight_layout()
            plt.savefig(reports / "ae_metal_sanity.png", dpi=120)
            logger.info("Wrote reports/ae_metal_sanity.png")
    except ImportError:
        logger.info("matplotlib not available — JSON only")

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--skip-patchcore", action="store_true")
    args = parser.parse_args()

    print(json.dumps(
        main(image_size=args.image_size, sample=args.sample, skip_patchcore=args.skip_patchcore),
        indent=2,
    ))
