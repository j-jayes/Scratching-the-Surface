"""Layer 1 FastAPI inference endpoint.

Receives an image, computes reconstruction MSE, and either:
- Returns {"result": "no_defect"} if MSE < threshold, or
- Enqueues the image URI to Azure Service Bus and returns {"result": "defect_candidate"}.
"""

from __future__ import annotations

import os
from io import BytesIO

import torch
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from cascade_defect.layer1_autoencoder.model import ConvAutoencoder

app = FastAPI(title="Layer 1 — Autoencoder Gatekeeper", version="0.1.0")

# ── Configuration (read from environment variables) ──────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MSE_THRESHOLD = float(os.getenv("MSE_THRESHOLD", "0.02"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/autoencoder/best.pt")
SB_CONN_STR = os.getenv("SERVICEBUS_CONNECTION_STRING", "")
SB_QUEUE_NAME = os.getenv("SERVICEBUS_QUEUE_NAME", "defect-queue")

# ── Load model at startup ─────────────────────────────────────────────────────
_model: ConvAutoencoder | None = None
_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


@app.on_event("startup")
async def load_model() -> None:
    global _model
    _model = ConvAutoencoder()
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    _model.load_state_dict(state)
    _model.eval()
    _model.to(DEVICE)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "device": DEVICE, "mse_threshold": MSE_THRESHOLD}


@app.post("/predict")
async def predict(file: UploadFile = File(...), image_uri: str = "") -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mse = _model.reconstruction_mse(tensor).item()

    if mse < MSE_THRESHOLD:
        return JSONResponse({"result": "no_defect", "mse": mse})

    # Defect candidate — enqueue to Service Bus
    if SB_CONN_STR:
        with ServiceBusClient.from_connection_string(SB_CONN_STR) as client:
            with client.get_queue_sender(SB_QUEUE_NAME) as sender:
                msg = ServiceBusMessage(image_uri or file.filename or "unknown")
                sender.send_messages(msg)

    return JSONResponse({"result": "defect_candidate", "mse": mse, "enqueued": bool(SB_CONN_STR)})
