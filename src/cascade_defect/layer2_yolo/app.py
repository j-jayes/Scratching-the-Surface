"""Layer 2 — YOLOv8 inference endpoint.

Receives an image, runs YOLOv8 inference, and returns the top detection.
Escalation to Layer 3 is handled by the cascade router, not here.

Model weights
-------------
Two ways to specify the weights file (in priority order):

1. ``YOLO_MODEL_BLOB`` (recommended)
       Path inside the models Blob container, e.g.
       ``visa/pcb1/yolo/best.pt``. Downloaded once at startup and cached at
       ``/tmp/yolo_best.pt``. Lets us swap models by changing one env var,
       no image rebuild.
2. ``YOLO_MODEL_PATH``
       Local filesystem path. Used as fallback (and for the legacy NEU build
       which baked weights into the image).
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

app = FastAPI(title="Layer 2 — YOLOv8 Specialist", version="0.2.0")

DEFAULT_LOCAL_PATH = Path("/tmp/yolo_best.pt")
MODEL_PATH_ENV = "YOLO_MODEL_PATH"
MODEL_BLOB_ENV = "YOLO_MODEL_BLOB"
BLOB_ACCOUNT_ENV = "BLOB_ACCOUNT"
BLOB_CONTAINER_ENV = "BLOB_CONTAINER_MODELS"

_model: YOLO | None = None


def _resolve_model_path() -> Path:
    """Resolve weight path, downloading from Blob on first call if needed."""
    blob_name = os.getenv(MODEL_BLOB_ENV)
    if blob_name:
        account = os.environ[BLOB_ACCOUNT_ENV]
        container = os.getenv(BLOB_CONTAINER_ENV, "models")
        local = DEFAULT_LOCAL_PATH
        if local.exists() and local.stat().st_size > 0:
            logger.info("YOLO weights already cached at %s — reusing", local)
            return local
        logger.info(
            "Downloading YOLO weights from az://%s/%s/%s → %s",
            account, container, blob_name, local,
        )
        # Lazy import — keeps non-Blob deployments cheap to start.
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        bsc = BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as f:
            f.write(
                bsc.get_container_client(container)
                .get_blob_client(blob_name)
                .download_blob()
                .readall()
            )
        return local

    fallback = Path(os.getenv(MODEL_PATH_ENV, "models/yolo/best.pt"))
    if not fallback.exists():
        raise RuntimeError(
            f"YOLO weights not found at {fallback} and no {MODEL_BLOB_ENV} set."
        )
    return fallback


@app.on_event("startup")
async def load_model() -> None:
    global _model
    weights = _resolve_model_path()
    logger.info("Loading YOLO weights from %s", weights)
    _model = YOLO(str(weights))


@app.get("/health")
async def health() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": device, "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:  # noqa: B008
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    results = _model.predict(img, conf=0.01, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return JSONResponse({"result": "no_detection", "confidence": 0.0})

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    best_conf = float(boxes.conf[best_idx])
    best_cls = int(boxes.cls[best_idx])
    class_name = _model.names[best_cls]

    return JSONResponse({
        "result": "defect_detected",
        "class": class_name,
        "confidence": best_conf,
    })
