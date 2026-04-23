"""Convolutional Autoencoder — model definition and training."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    """Downsampling encoder: 3×256×256 → 512-dim latent vector."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 128×128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64×64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32×32
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 16×16
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ConvDecoder(nn.Module):
    """Upsampling decoder: 256×16×16 → 3×256×256."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ConvAutoencoder(nn.Module):
    """Full convolutional autoencoder used as Layer 1 Gatekeeper."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

    def reconstruction_mse(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-image MSE reconstruction error (shape: [B])."""
        x_hat = self(x)
        return ((x - x_hat) ** 2).mean(dim=(1, 2, 3))


def train(
    model: ConvAutoencoder,
    train_loader: DataLoader,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
    checkpoint_dir: Path = Path("models/autoencoder"),
) -> None:
    """Train the autoencoder on defect-free images and save the best checkpoint."""
    import mlflow

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    with mlflow.start_run(run_name="layer1-autoencoder"):
        mlflow.log_params({"epochs": epochs, "lr": lr, "device": device})
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                optimizer.zero_grad()
                recon = model(imgs)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_mse", avg_loss, step=epoch)
            logger.info("Epoch %d/%d — MSE: %.6f", epoch + 1, epochs, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = checkpoint_dir / "best.pt"
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(str(ckpt_path))

        mlflow.pytorch.log_model(model, "autoencoder_model")
    logger.info("Training complete. Best MSE: %.6f", best_loss)
