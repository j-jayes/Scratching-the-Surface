"""Layer-1 anomaly scoring — Tier-1 improvements (Phase J.1).

This module is the *single source of truth* for converting an image into an
anomaly score. Both training (threshold derivation) and serving (FastAPI)
import from here, so any tweak to the score function applies end-to-end.

What it does differently from the v1 baseline (`ConvAutoencoder.reconstruction_mse`)
=================================================================================

1. **Per-domain channel normalisation.** Severstal mean intensity is markedly
   lower than KSDD2; feeding both through the same standardisation forces the
   AE to spend capacity on a brightness offset that is not a defect signal.
   We standardise per-domain before encoding.

2. **Patch-level max scoring.** A Severstal frame is 256×1600. A localised
   defect occupies <1% of pixels, so an image-mean MSE drowns the signal in a
   sea of normal background. We unfold the residual map into 32×32 patches and
   take the *p99* patch MSE — robust to outlier pixels but still spatially
   local.

3. **Z-score against per-domain normal distribution.** The raw patch-max MSE
   has a different scale on each domain (KSDD2 patches are visually busier).
   We persist the (mean, std) of the normal-only training distribution and
   return a *standardised score*: `(raw - μ_normal) / σ_normal`. Threshold
   becomes a single scalar (e.g. 3.0) shared across domains — no more
   `τ_ksdd2` vs `τ_severstal` bookkeeping.

The module never imports cv2 or anomalib — it is pure torch + numpy and is
safe to call from the inference container.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from cascade_defect.layer1_autoencoder.model import ConvAutoencoder

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN = "severstal"
PATCH_SIZE = 32
PATCH_STRIDE = 16  # 50% overlap — cheap and improves spatial coverage
QUANTILE = 0.99


@dataclass(frozen=True, slots=True)
class DomainStats:
    """Per-domain calibration computed once on the held-out normal set."""

    mean: tuple[float, float, float]  # channel mean for input normalisation
    std: tuple[float, float, float]   # channel std for input normalisation
    score_mean: float                  # mean of raw patch-max MSE on normals
    score_std: float                   # std of raw patch-max MSE on normals
    n_calibration: int                 # how many normal images contributed


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    raw_score: float       # raw patch-quantile MSE (un-normalised)
    z_score: float         # standardised against normal distribution
    domain: str
    is_anomaly: bool       # z_score >= z_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Image → tensor pipeline
# ─────────────────────────────────────────────────────────────────────────────
CONTRAST_SIGMA = 3.0  # ±3σ spans ~99.7% of a Gaussian — clamp tails into [0,1]


def _contrast_transform(stats: DomainStats):
    """Per-domain contrast normalisation that *stays in [0,1]*.

    The decoder ends with ``Sigmoid()`` so its output is constrained to
    [0,1]; reconstruction loss is only meaningful if the *input* lives in
    the same range. Plain z-score normalisation pushes inputs outside [0,1]
    and breaks the loss surface (we observed this on the J.1 first run).

    The transform centres each channel on the per-domain mean and rescales
    so ±3σ maps to ±1, then offsets to [0,1]. Pixels beyond ±3σ are clamped
    — they are statistical outliers in the normal distribution and likely
    correspond to defect-edge pixels we *want* the AE to fail to model.
    """
    mean = torch.tensor(stats.mean).view(3, 1, 1)
    std = torch.tensor(stats.std).view(3, 1, 1).clamp_min(1e-6)

    def _apply(t: torch.Tensor) -> torch.Tensor:
        z = (t - mean) / (CONTRAST_SIGMA * std)
        return (z.clamp(-1.0, 1.0) + 1.0) * 0.5

    return transforms.Lambda(_apply)


def make_transform(image_size: int, stats: DomainStats | None) -> transforms.Compose:
    """Domain-aware transform. Without stats, falls back to plain [0,1]."""
    ops: list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if stats is not None:
        ops.append(_contrast_transform(stats))
    return transforms.Compose(ops)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────
def _patch_max_mse(
    residual: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
    quantile: float = QUANTILE,
) -> torch.Tensor:
    """``residual`` is ``[B, C, H, W]``. Returns a ``[B]`` tensor of per-image
    patch-quantile reconstruction error.

    We use a high quantile rather than the strict max so a single dead pixel
    cannot drive the score.
    """
    sq = residual.pow(2).mean(dim=1, keepdim=True)  # [B, 1, H, W]
    # Aggregate per patch via average pooling — equivalent to mean pixel error
    # over the patch window.
    patch_mse = F.avg_pool2d(sq, kernel_size=patch_size, stride=stride)
    flat = patch_mse.flatten(1)  # [B, P]
    if quantile >= 1.0:
        return flat.max(dim=1).values
    # torch.quantile is exact; fine for P in the low thousands.
    return torch.quantile(flat, q=quantile, dim=1)


@torch.no_grad()
def score_tensor(
    model: ConvAutoencoder,
    image: torch.Tensor,  # [3, H, W] in the normalisation space of `stats`
    *,
    stats: DomainStats | None,
    z_threshold: float = 3.0,
    domain: str = DEFAULT_DOMAIN,
) -> AnomalyResult:
    batch = image.unsqueeze(0)
    recon = model(batch)
    residual = batch - recon
    raw = float(_patch_max_mse(residual).item())
    if stats is None or stats.score_std <= 0:
        z = float("nan")
        is_anom = False
    else:
        z = (raw - stats.score_mean) / stats.score_std
        is_anom = z >= z_threshold
    return AnomalyResult(
        raw_score=raw, z_score=z, domain=domain, is_anomaly=is_anom
    )


@torch.no_grad()
def calibrate_domain(
    model: ConvAutoencoder,
    image_paths: list[Path],
    *,
    image_size: int,
    device: str,
    sample_cap: int | None = 500,
) -> DomainStats | None:
    """Fit per-domain channel mean/std AND score mean/std on a normal set.

    Two-pass: pass 1 computes channel statistics on raw [0,1] tensors; pass 2
    computes the patch-quantile score under those statistics. ``sample_cap``
    keeps calibration fast even when the val set has thousands of frames.
    """
    if not image_paths:
        return None
    paths = image_paths[:sample_cap] if sample_cap else image_paths

    # Pass 1: channel mean/std on [0,1] tensors.
    raw_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    s = torch.zeros(3)
    sq = torch.zeros(3)
    n = 0
    for p in paths:
        t = raw_tf(load_image(p))  # [3, H, W]
        s += t.mean(dim=(1, 2))
        sq += t.pow(2).mean(dim=(1, 2))
        n += 1
    mean = (s / n)
    var = (sq / n) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt()

    provisional = DomainStats(
        mean=tuple(mean.tolist()),
        std=tuple(std.tolist()),
        score_mean=0.0,
        score_std=1.0,
        n_calibration=n,
    )

    # Pass 2: score distribution under the provisional normalisation.
    norm_tf = make_transform(image_size, provisional)
    raw_scores: list[float] = []
    model.eval()
    for p in paths:
        t = norm_tf(load_image(p)).to(device)
        result = score_tensor(model, t, stats=None, domain="calibration")
        raw_scores.append(result.raw_score)
    arr = torch.tensor(raw_scores)
    return DomainStats(
        mean=provisional.mean,
        std=provisional.std,
        score_mean=float(arr.mean()),
        score_std=float(arr.std().clamp_min(1e-9)),
        n_calibration=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_calibration(path: Path, calib: dict[str, DomainStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: asdict(v) for k, v in calib.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> dict[str, DomainStats]:
    data = json.loads(path.read_text())
    return {
        k: DomainStats(
            mean=tuple(v["mean"]),
            std=tuple(v["std"]),
            score_mean=float(v["score_mean"]),
            score_std=float(v["score_std"]),
            n_calibration=int(v["n_calibration"]),
        )
        for k, v in data.items()
    }


def infer_domain_from_filename(name: str) -> str:
    n = name.lower()
    if n.startswith("ksdd2_") or "kolektor" in n:
        return "ksdd2"
    return "severstal"
