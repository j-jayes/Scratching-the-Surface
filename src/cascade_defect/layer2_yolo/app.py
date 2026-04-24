"""Layer 2 — YOLOv8 inference endpoint.

Receives an image, runs YOLOv8 inference, and returns the top detection.
Escalation to Layer 3 is handled by the cascade router, not here.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

app = FastAPI(title="Layer 2 — YOLOv8 Specialist", version="0.1.0")

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo/best.pt")

_model: YOLO | None = None


@app.on_event("startup")
async def load_model() -> None:
    global _model
    _model = YOLO(MODEL_PATH)


@app.get("/health")
async def health() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
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
