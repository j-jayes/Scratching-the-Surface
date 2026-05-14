"""Classical CV track — ResNet50 multi-class defect classifier."""

from .resnet50 import CLASSES, build_model, load_checkpoint

__all__ = ["CLASSES", "build_model", "load_checkpoint"]
