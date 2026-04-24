"""Layer-2 YOLO model setup.

For the v1 vertical slice we use the pretrained ``yolov8n.pt`` weights from
Ultralytics as a placeholder. They detect 80 COCO classes (not steel defects)
so the cascade is structurally correct but quality is symbolic.

A future iteration will fine-tune on the GPT-Oracle pseudo-labels:

    yolo detect train data=data/processed/yolo/data.yaml \\
        model=yolov8n.pt epochs=30 imgsz=640
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_v1_placeholder(output_dir: Path = Path("models/yolo")) -> Path:
    """Download / copy the pretrained YOLOv8n weights as the Layer-2 v1 model."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "best.pt"
    if target.exists():
        logger.info("YOLO v1 placeholder already present at %s", target)
        return target

    # Triggers a one-time download to the Ultralytics cache.
    model = YOLO("yolov8n.pt")
    src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") and model.ckpt_path else None
    if src is None or not src.exists():
        # Fallback: ultralytics drops yolov8n.pt in CWD on first download.
        src = Path("yolov8n.pt")
    shutil.copy2(src, target)
    logger.info("Copied %s → %s", src, target)
    return target


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_v1_placeholder()
