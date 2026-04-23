"""Layer 2 — YOLOv8 inference endpoint.

Receives an image URI (from Service Bus via KEDA), runs YOLOv8 inference, and:
- If confidence ≥ CONF_THRESHOLD → logs the defect result directly.
- If confidence < CONF_THRESHOLD → escalates to Layer 3 (GPT-4o).
"""

from __future__ import annotations

import logging
import os
from io import BytesIO

import httpx
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

app = FastAPI(title="Layer 2 — YOLOv8 Specialist", version="0.1.0")

# ── Configuration ─────────────────────────────────────────────────────────────
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.85"))
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolo/best.pt")
LAYER3_URL = os.getenv("LAYER3_URL", "http://layer3-gpt4o:8002/predict")

_model: YOLO | None = None


@app.on_event("startup")
async def load_model() -> None:
    global _model
    _model = YOLO(MODEL_PATH)


@app.get("/health")
async def health() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": device, "conf_threshold": CONF_THRESHOLD}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    results = _model.predict(img, conf=0.01, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return JSONResponse({"result": "no_detection", "layer": 2})

    # Pick highest-confidence detection
    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    best_conf = float(boxes.conf[best_idx])
    best_cls = int(boxes.cls[best_idx])
    class_name = _model.names[best_cls]

    if best_conf >= CONF_THRESHOLD:
        return JSONResponse({
            "result": "defect_detected",
            "class": class_name,
            "confidence": best_conf,
            "layer": 2,
        })

    # Escalate to Layer 3
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                LAYER3_URL,
                files={"file": (file.filename, contents, file.content_type)},
            )
            resp.raise_for_status()
            layer3_result = resp.json()
            layer3_result["layer"] = 3
            return JSONResponse(layer3_result)
        except httpx.HTTPError:
            logger.warning("Layer 3 escalation failed for file %s", file.filename)
            return JSONResponse({
                "result": "defect_detected",
                "class": class_name,
                "confidence": best_conf,
                "layer": 2,
                "escalation_error": "Layer 3 unavailable; falling back to Layer 2 result.",
            })
