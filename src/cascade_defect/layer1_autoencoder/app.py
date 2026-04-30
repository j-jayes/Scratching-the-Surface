"""Layer 1 FastAPI inference endpoint (Phase J.1).

Scores an image with the patch-quantile z-score from
``cascade_defect.layer1_autoencoder.scoring``, falling back to plain
image-mean MSE if no calibration file is present (back-compat with the v1
NEU autoencoder).

Request::

    POST /predict   multipart: file=<image>, domain=<ksdd2|severstal>

Response::

    { "result": "no_defect"|"defect_candidate",
      "score": <z or raw mse>,
      "score_kind": "patch_quantile_z"|"image_mean_mse",
      "domain": <domain>,
      "enqueued": bool }
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path

import torch
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from cascade_defect.layer1_autoencoder.model import ConvAutoencoder
from cascade_defect.layer1_autoencoder.scoring import (
    DomainStats,
    load_calibration,
    make_transform,
    score_tensor,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Layer 1 — Autoencoder Gatekeeper", version="0.2.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORER = os.getenv("SCORER", "ae").lower()  # "ae" | "patchcore"
MSE_THRESHOLD = float(os.getenv("MSE_THRESHOLD", "0.02"))
Z_THRESHOLD = float(os.getenv("Z_THRESHOLD", "3.0"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/autoencoder/best.pt")
CALIBRATION_PATH = os.getenv(
    "CALIBRATION_PATH", "models/autoencoder_metal/calibration.json"
)
PATCHCORE_DIR = os.getenv("PATCHCORE_DIR", "models/patchcore_metal")
DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN", "severstal")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
SB_CONN_STR = os.getenv("SERVICEBUS_CONNECTION_STRING", "")
SB_QUEUE_NAME = os.getenv("SERVICEBUS_QUEUE_NAME", "defect-queue")

_model: ConvAutoencoder | None = None
_calibration: dict[str, DomainStats] = {}
_patchcore_extractor = None  # _FeatureExtractor when SCORER == "patchcore"
_patchcore_banks: dict = {}  # domain -> (bank_tensor, PatchCoreCalibration)
_legacy_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


@app.on_event("startup")
async def load_model() -> None:
    global _model, _calibration, _patchcore_extractor, _patchcore_banks
    if SCORER == "patchcore":
        from cascade_defect.layer1_autoencoder.patchcore import (
            _FeatureExtractor,
            load_bank,
        )

        _patchcore_extractor = _FeatureExtractor().to(DEVICE)
        pc_dir = Path(PATCHCORE_DIR)
        for d in ("ksdd2", "severstal"):
            try:
                _patchcore_banks[d] = load_bank(pc_dir, d)
            except FileNotFoundError:
                logger.warning("No PatchCore bank for %s under %s", d, pc_dir)
        if not _patchcore_banks:
            raise RuntimeError(
                f"SCORER=patchcore but no banks loaded from {pc_dir}"
            )
        logger.info("PatchCore loaded for domains: %s", sorted(_patchcore_banks))
        return

    _model = ConvAutoencoder()
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    _model.load_state_dict(state)
    _model.eval()
    _model.to(DEVICE)
    calib_path = Path(CALIBRATION_PATH)
    if calib_path.exists():
        _calibration = load_calibration(calib_path)
        logger.info("Loaded calibration for domains: %s", sorted(_calibration))
    else:
        logger.warning(
            "No calibration at %s — falling back to image-mean MSE", calib_path
        )


@app.get("/health")
async def health() -> dict:
    if SCORER == "patchcore":
        return {
            "status": "ok",
            "device": DEVICE,
            "score_kind": "patchcore_z",
            "z_threshold": Z_THRESHOLD,
            "domains": sorted(_patchcore_banks),
            "default_domain": DEFAULT_DOMAIN,
        }
    return {
        "status": "ok",
        "device": DEVICE,
        "score_kind": "patch_quantile_z" if _calibration else "image_mean_mse",
        "z_threshold": Z_THRESHOLD,
        "mse_threshold": MSE_THRESHOLD,
        "domains": sorted(_calibration),
        "default_domain": DEFAULT_DOMAIN,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),  # noqa: B008  (FastAPI dependency injection)
    image_uri: str = Form(""),  # noqa: B008
    domain: str = Form(""),  # noqa: B008
) -> JSONResponse:
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    chosen_domain = (domain or DEFAULT_DOMAIN).lower()

    if SCORER == "patchcore":
        from cascade_defect.layer1_autoencoder.patchcore import score_image

        bundle = _patchcore_banks.get(chosen_domain)
        if bundle is None:
            raise HTTPException(
                status_code=404, detail=f"No PatchCore bank for domain={chosen_domain}"
            )
        bank, pc_calib = bundle
        raw = score_image(_patchcore_extractor, bank, img, device=DEVICE)
        z = (raw - pc_calib.score_mean) / max(pc_calib.score_std, 1e-9)
        is_defect = z >= Z_THRESHOLD
        score_value = z
        score_kind = "patchcore_z"
    else:
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        stats = _calibration.get(chosen_domain) if _calibration else None
        if stats is not None:
            tensor = make_transform(IMAGE_SIZE, stats)(img).to(DEVICE)
            result = score_tensor(
                _model, tensor, stats=stats, z_threshold=Z_THRESHOLD, domain=chosen_domain
            )
            is_defect = result.is_anomaly
            score_value = result.z_score
            score_kind = "patch_quantile_z"
        else:
            tensor = _legacy_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                score_value = float(_model.reconstruction_mse(tensor).item())
            is_defect = score_value >= MSE_THRESHOLD
            score_kind = "image_mean_mse"

    payload: dict = {
        "result": "defect_candidate" if is_defect else "no_defect",
        "score": score_value,
        "score_kind": score_kind,
        "domain": chosen_domain,
    }

    if is_defect and SB_CONN_STR:
        with (
            ServiceBusClient.from_connection_string(SB_CONN_STR) as client,
            client.get_queue_sender(SB_QUEUE_NAME) as sender,
        ):
            msg = ServiceBusMessage(image_uri or file.filename or "unknown")
            sender.send_messages(msg)
        payload["enqueued"] = True
    elif is_defect:
        payload["enqueued"] = False

    return JSONResponse(payload)
