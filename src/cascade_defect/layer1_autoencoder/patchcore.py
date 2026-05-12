"""PatchCore-lite — pretrained-backbone anomaly detection (Phase J.1, Tier-3).

Why "lite": full PatchCore (Roth et al., CVPR 2022) uses a greedy-coreset
sub-sample of WideResNet50 mid-level features and Mahalanobis / kNN scoring.
We implement the same algorithm with two pragmatic shortcuts:

* **ResNet18** (~11 M params, ~50 ms / image on CPU at 224 px) instead of
  WideResNet50 (~70 M, ~250 ms). Less expressive but keeps the cascade L1
  budget under 300 ms.
* **Random subsampling** of the memory bank instead of greedy coreset
  selection. The coreset trick saves memory at constant accuracy; for a
  ~6 k-image normal pool a random 10% subsample is statistically equivalent
  and 50× faster to build.

Everything else matches the paper: hooks on `layer2` + `layer3`, bilinearly
upsampled and concatenated into a (D, H, W) feature volume per image, then
flattened to per-patch feature vectors. Anomaly score = mean of the top-k
nearest-neighbour distances in feature space, aggregated to one scalar per
image via a high quantile (matches the AE scoring contract).

Dependencies: torch + torchvision only — no anomalib, no lightning, no extra
container layers. The Layer-1 image already has both.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 224
DEFAULT_K = 5
DEFAULT_QUANTILE = 0.99
DEFAULT_BANK_FRACTION = 0.10  # random coreset substitute
MAX_BANK_VECTORS = 200_000  # hard cap on memory footprint


@dataclass(frozen=True, slots=True)
class PatchCoreCalibration:
    score_mean: float
    score_std: float
    n_calibration: int


# ─────────────────────────────────────────────────────────────────────────────
# Backbone (ResNet18, ImageNet pretrained, frozen)
# ─────────────────────────────────────────────────────────────────────────────
class _FeatureExtractor(torch.nn.Module):
    """Hooks `layer2` (256-d) + `layer3` (512-d) of ResNet18 → 768-d patches."""

    def __init__(self, *, backbone: str = "resnet18") -> None:
        super().__init__()
        backbone_l = backbone.lower()
        if backbone_l == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
            self.backbone_name = "resnet18-imagenet1k_v1"
        elif backbone_l in {"wrn50", "wide_resnet50_2"}:
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
            self.backbone = models.wide_resnet50_2(weights=weights)
            self.backbone_name = "wide_resnet50_2-imagenet1k_v2"
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._features: dict[str, torch.Tensor] = {}
        self.backbone.layer2.register_forward_hook(self._hook("layer2"))
        self.backbone.layer3.register_forward_hook(self._hook("layer3"))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def _hook(self, name: str):
        def fn(_module, _inp, out):
            self._features[name] = out

        return fn

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Return [B, D, H, W] concatenated multi-scale feature volume.

        ``layer3`` is upsampled bilinearly to match ``layer2`` spatial size,
        then concatenated along the channel axis. Final spatial grid is 28×28
        for a 224 px input.
        """
        self._features.clear()
        _ = self.backbone(batch)
        f2 = self._features["layer2"]
        f3 = self._features["layer3"]
        f3u = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3u], dim=1)

    @torch.no_grad()
    def encode_image(self, img: Image.Image, device: str) -> torch.Tensor:
        t = self.transform(img.convert("RGB")).unsqueeze(0).to(device)
        return self.forward(t)[0]  # [D, H, W]


def _patches_from_volume(vol: torch.Tensor) -> torch.Tensor:
    """[D, H, W] → [H*W, D] L2-normalised patches."""
    d, h, w = vol.shape
    flat = vol.reshape(d, h * w).T  # [H*W, D]
    return F.normalize(flat, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Memory bank training
# ─────────────────────────────────────────────────────────────────────────────
def build_memory_bank(
    extractor: _FeatureExtractor,
    image_paths: Iterable[Path],
    *,
    device: str = "cpu",
    bank_fraction: float = DEFAULT_BANK_FRACTION,
    max_vectors: int = MAX_BANK_VECTORS,
    seed: int = 42,
) -> torch.Tensor:
    """Encode every normal image into patches, randomly subsample, return bank
    as a single [N, D] float32 tensor on CPU.
    """
    rng = random.Random(seed)
    chunks: list[torch.Tensor] = []
    n_images = 0
    extractor.to(device).eval()
    for p in image_paths:
        try:
            img = Image.open(p)
        except OSError:
            continue
        vol = extractor.encode_image(img, device).cpu()
        patches = _patches_from_volume(vol)  # [P, D]
        n = patches.shape[0]
        keep = max(1, int(n * bank_fraction))
        if keep < n:
            idx = torch.tensor(rng.sample(range(n), keep))
            patches = patches.index_select(0, idx)
        chunks.append(patches)
        n_images += 1
    if not chunks:
        raise RuntimeError("No images contributed to memory bank")
    bank = torch.cat(chunks, dim=0)
    if bank.shape[0] > max_vectors:
        idx = torch.randperm(bank.shape[0], generator=torch.Generator().manual_seed(seed))[
            :max_vectors
        ]
        bank = bank.index_select(0, idx)
    logger.info(
        "Built bank from %d images → %d patches × %d dims", n_images, bank.shape[0], bank.shape[1]
    )
    return bank.contiguous().float()


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_image(
    extractor: _FeatureExtractor,
    bank: torch.Tensor,
    img: Image.Image,
    *,
    device: str = "cpu",
    k: int = DEFAULT_K,
    quantile: float = DEFAULT_QUANTILE,
) -> float:
    """Return per-image anomaly score (high = anomalous).

    For each query patch, take mean cosine distance to its k nearest bank
    patches; aggregate to one scalar per image by taking the ``quantile`` of
    per-patch scores (p99 by default — robust max).
    """
    vol = extractor.encode_image(img, device).cpu()
    q = _patches_from_volume(vol)  # [P, D]
    # Cosine distance via dot product on L2-normalised vectors.
    sim = q @ bank.T  # [P, N]
    topk = sim.topk(k=min(k, bank.shape[0]), dim=1).values
    per_patch = 1.0 - topk.mean(dim=1)  # cosine distance
    return float(torch.quantile(per_patch, quantile).item())


def calibrate(
    extractor: _FeatureExtractor,
    bank: torch.Tensor,
    image_paths: list[Path],
    *,
    device: str = "cpu",
    sample_cap: int = 300,
    k: int = DEFAULT_K,
    quantile: float = DEFAULT_QUANTILE,
) -> PatchCoreCalibration:
    paths = image_paths[:sample_cap]
    scores: list[float] = []
    for p in paths:
        try:
            img = Image.open(p)
        except OSError:
            continue
        scores.append(score_image(extractor, bank, img, device=device, k=k, quantile=quantile))
    arr = np.asarray(scores, dtype=np.float64)
    return PatchCoreCalibration(
        score_mean=float(arr.mean()) if arr.size else 0.0,
        score_std=float(arr.std()) if arr.size else 1.0,
        n_calibration=int(arr.size),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_bank(
    output_dir: Path,
    *,
    bank_by_domain: dict[str, torch.Tensor],
    calibration_by_domain: dict[str, PatchCoreCalibration],
    backbone: str = "resnet18-imagenet1k_v1",
    image_size: int = DEFAULT_IMAGE_SIZE,
    k: int = DEFAULT_K,
    quantile: float = DEFAULT_QUANTILE,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for d, bank in bank_by_domain.items():
        torch.save(bank, output_dir / f"bank_{d}.pt")
    summary = {
        "image_size": image_size,
        "k": k,
        "quantile": quantile,
        "backbone": backbone,
        "feature_layers": ["layer2", "layer3"],
        "per_domain": {
            d: {
                "n_patches": int(bank.shape[0]),
                "dim": int(bank.shape[1]),
                "score_mean": calibration_by_domain[d].score_mean,
                "score_std": calibration_by_domain[d].score_std,
                "n_calibration": calibration_by_domain[d].n_calibration,
            }
            for d, bank in bank_by_domain.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def load_bank(output_dir: Path, domain: str) -> tuple[torch.Tensor, PatchCoreCalibration]:
    bank = torch.load(output_dir / f"bank_{domain}.pt", map_location="cpu", weights_only=True)
    summary = json.loads((output_dir / "summary.json").read_text())
    block = summary["per_domain"][domain]
    calib = PatchCoreCalibration(
        score_mean=float(block["score_mean"]),
        score_std=float(block["score_std"]),
        n_calibration=int(block["n_calibration"]),
    )
    return bank, calib
