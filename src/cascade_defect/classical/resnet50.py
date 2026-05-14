"""ResNet50 backbone + 5-class head for the classical CV track.

Classes (index order, matches ``data/splits_metal_v2/manifest.json``):

    0: no_defect
    1: pitting
    2: inclusion
    3: scratch
    4: patch
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

CLASSES: list[str] = ["no_defect", "pitting", "inclusion", "scratch", "patch"]
NUM_CLASSES = len(CLASSES)


def build_model(num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """Return a ResNet50 with a fresh ``num_classes`` linear head."""
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net


def load_checkpoint(path: str | Path,
                    device: str | torch.device = "cpu") -> nn.Module:
    """Rebuild the architecture and load a saved ``state_dict``."""
    net = build_model(pretrained=False)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net.to(device)
