"""Layer 3 — Oracle FastAPI endpoint (online cascade).

Reuses the shared prompt + Pydantic schema from `oracle.py`.
Handles low-confidence edge cases escalated by Layer 2.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cascade_defect.layer3_gpt4o.oracle import predict

logger = logging.getLogger(__name__)

app = FastAPI(title="Layer 3 — Oracle", version="0.2.0")

SEED_DIR = Path(os.getenv("FEW_SHOT_SEED_DIR", "data/splits/seed"))
DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN", "")  # empty = NEU back-compat


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "deployment": os.getenv("AOAI_DEPLOYMENT", "oracle"),
        "default_domain": DEFAULT_DOMAIN or "neu",
        "seed_dir": str(SEED_DIR),
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),  # noqa: B008  (FastAPI dependency injection)
    domain: str = Form(""),  # noqa: B008
) -> JSONResponse:
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    chosen_domain = domain or DEFAULT_DOMAIN or None
    try:
        prediction, usage = predict(tmp_path, SEED_DIR, domain=chosen_domain)
    except Exception as e:
        logger.exception("Oracle call failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({
        "layer": 3,
        "result": (
            "no_defect" if prediction.defect_class == "no_defect"
            else "uncertain" if prediction.defect_class == "uncertain"
            else "defect"
        ),
        "class": prediction.defect_class,
        "confidence": prediction.confidence,
        "reasoning": prediction.reasoning,
        "domain": chosen_domain or "neu",
        "usage": usage,
    })
