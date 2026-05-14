"""Severstal v2 split → torch ``Dataset`` + ``DataLoader`` factories."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .resnet50 import CLASSES

ROOT = Path(__file__).resolve().parents[3]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}

# Severstal images are 256×1600 grayscale; ImageNet expects 3-channel 224×224.
# We resize the *short* side to 256 then take a 224×224 crop. This preserves
# the elongated geometry better than a square resize that would squish defects.
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

TRAIN_TFM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    _NORMALIZE,
])

EVAL_TFM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])


class SeverstalCSV(Dataset):
    """Reads a labels CSV (``image_path,label[,split]``) and returns tensors."""

    def __init__(self, csv_path: str | Path,
                 split_filter: str | None = None,
                 transform=EVAL_TFM):
        self.transform = transform
        self.rows: list[tuple[str, int]] = []
        with Path(csv_path).open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if split_filter is not None and r.get("split") != split_filter:
                    continue
                self.rows.append((r["image_path"], LABEL2IDX[r["label"]]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        rel, y = self.rows[idx]
        img = Image.open(ROOT / rel).convert("L")
        return self.transform(img), y, rel


def make_loaders(train_csv: str | Path, batch_size: int = 32,
                 num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
    train = SeverstalCSV(train_csv, split_filter="train", transform=TRAIN_TFM)
    val = SeverstalCSV(train_csv, split_filter="val", transform=EVAL_TFM)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=False, drop_last=True),
        DataLoader(val, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=False),
    )


def make_eval_loader(csv_path: str | Path, batch_size: int = 64,
                     num_workers: int = 2) -> DataLoader:
    ds = SeverstalCSV(csv_path, split_filter=None, transform=EVAL_TFM)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


def class_weights_from_csv(csv_path: str | Path,
                           split_filter: str = "train") -> torch.Tensor:
    """Inverse-frequency class weights, normalised so mean weight = 1."""
    counts = [0] * len(CLASSES)
    with Path(csv_path).open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != split_filter:
                continue
            counts[LABEL2IDX[r["label"]]] += 1
    counts_t = torch.tensor(counts, dtype=torch.float32).clamp_min(1.0)
    inv = counts_t.sum() / (len(counts_t) * counts_t)
    return inv
