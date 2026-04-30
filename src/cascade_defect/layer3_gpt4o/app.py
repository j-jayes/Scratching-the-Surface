"""Layer 3 — Oracle FastAPI endpoint (online cascade).

Reuses the shared prompt + Pydantic schema from `oracle.py`.
Handles low-confidence edge cases escalated by Layer 2.

Phase L additions:
  * dHash perceptual cache (`cascade_defect.layer3_gpt4o.cache`) is enabled by
    default (USE_ORACLE_CACHE=1) so repeated demo requests are free.
  * Per-day USD cap (AOAI_DAILY_USD_CAP, default 5.0) tracked in
    /tmp/aoai_usage.json. When exceeded, the oracle is bypassed and a
    `rate_limited` result is returned. The counter resets at UTC midnight or
    whenever the container restarts (scale-to-zero).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cascade_defect.layer3_gpt4o.cache import cache_stats, cached_predict
from cascade_defect.layer3_gpt4o.oracle import predict

logger = logging.getLogger(__name__)

app = FastAPI(title="Layer 3 — Oracle", version="0.3.0")

SEED_DIR = Path(os.getenv("FEW_SHOT_SEED_DIR", "data/splits/seed"))
DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN", "")  # empty = NEU back-compat
USE_CACHE = os.getenv("USE_ORACLE_CACHE", "1") == "1"
DAILY_USD_CAP = float(os.getenv("AOAI_DAILY_USD_CAP", "5.0"))
USAGE_FILE = Path(os.getenv("AOAI_USAGE_FILE", "/tmp/aoai_usage.json"))

# gpt-4.1-mini pricing (USD per token) — keep in sync with eval/metrics_metal.py.
PRICE_IN = 0.40 / 1e6
PRICE_OUT = 1.60 / 1e6

_usage_lock = threading.Lock()


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _load_usage() -> dict:
    if not USAGE_FILE.exists():
        return {"date": _today(), "usd": 0.0}
    try:
        data = json.loads(USAGE_FILE.read_text())
        if data.get("date") != _today():
            return {"date": _today(), "usd": 0.0}
        return data
    except Exception:
        return {"date": _today(), "usd": 0.0}


def _save_usage(data: dict) -> None:
    try:
        USAGE_FILE.write_text(json.dumps(data))
    except OSError as e:
        logger.warning("Could not persist usage file %s: %s", USAGE_FILE, e)


def _record_cost(usage: dict) -> float:
    cost = (
        usage.get("prompt_tokens", 0) * PRICE_IN
        + usage.get("completion_tokens", 0) * PRICE_OUT
    )
    with _usage_lock:
        data = _load_usage()
        data["usd"] = float(data.get("usd", 0.0)) + cost
        _save_usage(data)
        return data["usd"]


def _today_usd() -> float:
    with _usage_lock:
        return float(_load_usage().get("usd", 0.0))


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "deployment": os.getenv("AOAI_DEPLOYMENT", "oracle"),
        "default_domain": DEFAULT_DOMAIN or "neu",
        "seed_dir": str(SEED_DIR),
        "cache_enabled": USE_CACHE,
        "daily_usd_cap": DAILY_USD_CAP,
        "today_usd": round(_today_usd(), 6),
        "cache": cache_stats() if USE_CACHE else None,
    }


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),  # noqa: B008  (FastAPI dependency injection)
    domain: str = Form(""),  # noqa: B008
) -> JSONResponse:
    chosen_domain = domain or DEFAULT_DOMAIN or None

    # Daily $ cap — best-effort safeguard. Resets at UTC midnight or restart.
    if _today_usd() >= DAILY_USD_CAP:
        return JSONResponse({
            "layer": 3,
            "result": "rate_limited",
            "class": "uncertain",
            "confidence": 0.0,
            "reasoning": "Daily Azure OpenAI demo cap reached; please retry tomorrow.",
            "domain": chosen_domain or "neu",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "today_usd": round(_today_usd(), 4),
            "daily_usd_cap": DAILY_USD_CAP,
        }, status_code=200)

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    cache_info: dict | None = None
    try:
        if USE_CACHE:
            prediction, usage, cache_info = cached_predict(
                tmp_path, SEED_DIR, domain=chosen_domain
            )
        else:
            prediction, usage = predict(tmp_path, SEED_DIR, domain=chosen_domain)
    except Exception as e:
        logger.exception("Oracle call failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    # Only charge real (non-cache-hit) calls toward the daily cap.
    if not (cache_info and cache_info.get("hit")):
        _record_cost(usage)

    payload = {
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
    }
    if cache_info is not None:
        payload["cache"] = cache_info
    return JSONResponse(payload)
