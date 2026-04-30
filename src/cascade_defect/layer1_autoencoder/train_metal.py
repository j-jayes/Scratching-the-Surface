"""Train Layer-1 ConvAutoencoder on KSDD2+Severstal normals (Phase J / J.1).

Improvements over the v1 baseline:

* Training tensors are standardised with **per-domain channel mean/std** so the
  AE no longer has to model a brightness offset between the two datasets.
* After training, both domains get a **calibration block** written to
  ``models/autoencoder_metal/calibration.json`` — channel ``(mean, std)`` plus
  the score ``(mean, std)`` of the patch-quantile MSE on held-out normals.
  Inference reduces to a single z-score lookup; per-domain raw τ values are
  still reported in ``summary.json`` for backward compatibility.
* The score function is the patch-quantile MSE from
  ``layer1_autoencoder.scoring`` — robust to localised defects on Severstal's
  256×1600 strips, which the v1 image-mean MSE drowned out.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cascade_defect.layer1_autoencoder.model import ConvAutoencoder
from cascade_defect.layer1_autoencoder.scoring import (
    DomainStats,
    calibrate_domain,
    infer_domain_from_filename,
    make_transform,
    save_calibration,
    score_tensor,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/splits_metal")
DEFAULT_OUTPUT = Path("models/autoencoder_metal")
DOMAINS = ("ksdd2", "severstal")


def _list_images(root: Path) -> list[Path]:
    return sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg"))


def _quick_channel_stats(
    paths: Iterable[Path], image_size: int
) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    s = torch.zeros(3)
    sq = torch.zeros(3)
    n = 0
    for p in paths:
        t = tf(Image.open(p).convert("RGB"))
        s += t.mean(dim=(1, 2))
        sq += t.pow(2).mean(dim=(1, 2))
        n += 1
    if n == 0:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0
    mean = s / n
    var = (sq / n) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt()
    return tuple(mean.tolist()), tuple(std.tolist()), n


class _PerDomainNormalisedDataset(Dataset):
    """Apply each image's domain-specific transform on demand."""

    def __init__(
        self,
        paths: list[Path],
        image_size: int,
        domain_stats: dict[str, DomainStats],
    ) -> None:
        self.paths = paths
        self.transforms = {d: make_transform(image_size, s) for d, s in domain_stats.items()}
        self.fallback = make_transform(image_size, None)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        domain = infer_domain_from_filename(p.name)
        tf = self.transforms.get(domain, self.fallback)
        return tf(Image.open(p).convert("RGB"))


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    image_size: int = 256,
    z_threshold: float = 3.0,
    device: str | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device=%s, image_size=%d", device, image_size)

    ae_train_dir = data_dir / "ae_train"
    all_paths = _list_images(ae_train_dir)
    if not all_paths:
        raise RuntimeError(f"No images found under {ae_train_dir}")

    by_domain: dict[str, list[Path]] = {d: [] for d in DOMAINS}
    for p in all_paths:
        by_domain.setdefault(infer_domain_from_filename(p.name), []).append(p)
    for d, lst in by_domain.items():
        logger.info("Train images %s: %d", d, len(lst))

    sample_cap = 1000
    channel_stats: dict[str, DomainStats] = {}
    for d in DOMAINS:
        if not by_domain.get(d):
            continue
        mean, std, n = _quick_channel_stats(by_domain[d][:sample_cap], image_size)
        channel_stats[d] = DomainStats(
            mean=mean, std=std, score_mean=0.0, score_std=1.0, n_calibration=n
        )
        logger.info("Channel stats %s (n=%d): mean=%s std=%s", d, n, mean, std)

    train_ds = _PerDomainNormalisedDataset(all_paths, image_size, channel_stats)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ConvAutoencoder().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimiser.step()
            running += loss.item() * batch.size(0)
        epoch_loss = running / len(train_ds)
        logger.info("Epoch %d/%d  loss=%.6f", epoch + 1, epochs, epoch_loss)
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), output_dir / "best.pt")

    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))
    model.eval()

    calibration: dict[str, DomainStats] = {}
    per_domain_summary: dict[str, dict[str, float] | None] = {}
    for d in DOMAINS:
        val_dir = data_dir / "ae_val" / d
        val_paths = _list_images(val_dir)
        if not val_paths:
            per_domain_summary[d] = None
            continue
        calib = calibrate_domain(model, val_paths, image_size=image_size, device=device)
        if calib is None:
            per_domain_summary[d] = None
            continue
        calibration[d] = calib
        per_domain_summary[d] = {
            "n": calib.n_calibration,
            "channel_mean": [round(x, 4) for x in calib.mean],
            "channel_std": [round(x, 4) for x in calib.std],
            "score_mean": round(calib.score_mean, 6),
            "score_std": round(calib.score_std, 6),
            "z_threshold": z_threshold,
            "raw_threshold_at_z": round(
                calib.score_mean + z_threshold * calib.score_std, 6
            ),
        }
        logger.info("Calibration %s: %s", d, per_domain_summary[d])

    calibration_path = output_dir / "calibration.json"
    if calibration:
        save_calibration(calibration_path, calibration)

    summary = {
        "n_train_images": len(train_ds),
        "epochs": epochs,
        "image_size": image_size,
        "best_train_mse": round(best, 6),
        "checkpoint": str(output_dir / "best.pt"),
        "calibration_path": str(calibration_path) if calibration else None,
        "score_kind": "patch_quantile_z",
        "z_threshold": z_threshold,
        "per_domain": per_domain_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Training complete: %s", summary)
    return summary


__all__ = ["train", "score_tensor"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        z_threshold=args.z_threshold,
    )
