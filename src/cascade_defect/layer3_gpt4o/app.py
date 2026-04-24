"""Layer 3 — Oracle FastAPI endpoint (online cascade).

Reuses the shared prompt + Pydantic schema from `oracle.py`.
Handles low-confidence edge cases escalated by Layer 2.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cascade_defect.layer3_gpt4o.oracle import predict

logger = logging.getLogger(__name__)

app = FastAPI(title="Layer 3 — Oracle", version="0.1.0")

SEED_DIR = Path(os.getenv("FEW_SHOT_SEED_DIR", "data/splits/seed"))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "deployment": os.getenv("AOAI_DEPLOYMENT", "oracle")}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        prediction, usage = predict(tmp_path, SEED_DIR)
    except Exception as e:
        logger.exception("Oracle call failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({
        "layer": 3,
        "result": "defect" if prediction.defect_class != "no_defect" else "no_defect",
        "class": prediction.defect_class,
        "confidence": prediction.confidence,
        "reasoning": prediction.reasoning,
        "usage": usage,
    })
