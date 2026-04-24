"""Train the Layer-1 Convolutional Autoencoder on defect-free imagery.

For the v1 vertical slice we just train on every image in the unlabelled split —
the AE learns the *background texture* of the steel surface. In production you'd
filter to known-good frames captured during defect-free production runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cascade_defect.layer1_autoencoder.model import ConvAutoencoder

logger = logging.getLogger(__name__)


class FlatImageDataset(Dataset):
    """Walk a directory tree, yielding RGB tensors normalised to [0,1]."""

    def __init__(self, root: Path, image_size: int = 128) -> None:
        self.paths = sorted(root.rglob("*.jpg"))
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


def train(
    data_dir: Path = Path("data/splits/unlabelled"),
    output_dir: Path = Path("models/autoencoder"),
    *,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    image_size: int = 128,
    device: str | None = None,
) -> dict:
    """Lightweight CPU-friendly training loop. Returns a summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device=%s, image_size=%d", device, image_size)

    ds = FlatImageDataset(data_dir, image_size=image_size)
    if len(ds) == 0:
        raise RuntimeError(f"No images found under {data_dir}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Use the bundled architecture, but operate at 128×128 (the model handles any
    # square input that's a multiple of 16).
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
        epoch_loss = running / len(ds)
        logger.info("Epoch %d/%d  loss=%.6f", epoch + 1, epochs, epoch_loss)
        if epoch_loss < best:
            best = epoch_loss
            torch.save(model.state_dict(), output_dir / "best.pt")

    # Compute MSE on a held-out "normal" sample to derive a sane threshold (mean + 3σ).
    # Prefer same-class test images when training on a single class.
    threshold_src: Path | None = None
    if data_dir.parent.name == "unlabelled":
        cand = Path("data/splits/test") / data_dir.name
        if cand.exists():
            threshold_src = cand
    if threshold_src is None:
        threshold_src = Path("data/splits/seed")
    seed_ds = FlatImageDataset(threshold_src, image_size=image_size)
    if len(seed_ds) > 0:
        model.eval()
        with torch.no_grad():
            mses = torch.stack(
                [model.reconstruction_mse(seed_ds[i].unsqueeze(0).to(device)).cpu() for i in range(len(seed_ds))]
            ).flatten()
        threshold = float(mses.mean() + 3 * mses.std())
    else:
        threshold = 0.02

    summary = {
        "n_train_images": len(ds),
        "epochs": epochs,
        "best_train_mse": round(best, 6),
        "threshold_source": str(threshold_src),
        "suggested_mse_threshold": round(threshold, 6),
        "checkpoint": str(output_dir / "best.pt"),
    }
    (output_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2))
    logger.info("Training complete: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normal-class",
        default=None,
        help="If set, train only on this class folder under unlabelled/ (treats it as 'normal').",
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_dir = (
        Path("data/splits/unlabelled") / args.normal_class
        if args.normal_class else Path("data/splits/unlabelled")
    )
    train(data_dir=data_dir, epochs=args.epochs)
