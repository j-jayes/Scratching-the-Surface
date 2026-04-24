"""Quick sanity report: per-class AE reconstruction MSE on the test split."""
from __future__ import annotations

import statistics
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cascade_defect.layer1_autoencoder.model import ConvAutoencoder

m = ConvAutoencoder()
m.load_state_dict(torch.load("models/autoencoder/best.pt", weights_only=True, map_location="cpu"))
m.eval()
tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

print(f"{'class':18s}  mean    p50     p95")
for cls_dir in sorted(Path("data/splits/test").iterdir()):
    mses: list[float] = []
    for p in list(cls_dir.glob("*.jpg"))[:30]:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            mse = m.reconstruction_mse(x).item()
        mses.append(mse)
    s = sorted(mses)
    print(f"{cls_dir.name:18s}  {statistics.mean(mses):.5f}  {s[len(s)//2]:.5f}  {s[int(len(s)*0.95)]:.5f}")
