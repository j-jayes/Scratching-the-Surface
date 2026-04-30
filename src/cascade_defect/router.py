"""Cascade router — the single ingress endpoint.

Orchestrates the three layers:
1. Forwards the image to Layer 1 (autoencoder gatekeeper).
2. If Layer 1 says "no_defect" → return immediately. Cheap path.
3. Otherwise forwards to Layer 2 (YOLO specialist).
4. If Layer 2's confidence is below threshold → forwards to Layer 3 (Oracle).

Returns a per-layer trace so the caller can see the cascade decision path.

Also serves a small public demo page from /static and the index at "/".
Per-IP rate limit (slowapi) and a hard image-size guard protect the demo.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

LAYER1_URL = os.getenv("LAYER1_URL", "http://layer1:8000")
LAYER2_URL = os.getenv("LAYER2_URL", "http://layer2:8000")
LAYER3_URL = os.getenv("LAYER3_URL", "http://layer3:8000")
L2_CONF_ESCALATE_BELOW = float(os.getenv("L2_CONF_ESCALATE_BELOW", "0.7"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "src/cascade_defect/static"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))  # 2 MB
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "2048"))
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "10/minute;200/day")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Cascade Router", version="0.2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Same-origin demo page; CORS is permissive so other origins can also try the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _post_image(
    client: httpx.AsyncClient,
    url: str,
    name: str,
    blob: bytes,
    content_type: str | None,
    domain: str | None,
) -> dict[str, Any]:
    files = {"file": (name, blob, content_type or "image/jpeg")}
    data = {"domain": domain} if domain else None
    r = await client.post(f"{url}/predict", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "layers": {"l1": LAYER1_URL, "l2": LAYER2_URL, "l3": LAYER3_URL}}


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Demo page not bundled in image.")
    return FileResponse(str(index_path))


def _validate_image(blob: bytes) -> None:
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes ({len(blob)}).",
        )
    try:
        with Image.open(io.BytesIO(blob)) as img:
            w, h = img.size
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Not a valid image: {e}") from e
    if max(w, h) > MAX_IMAGE_DIM:
        raise HTTPException(
            status_code=413,
            detail=f"Image dimensions {w}x{h} exceed max {MAX_IMAGE_DIM}px.",
        )


@app.post("/predict")
@limiter.limit(RATE_LIMIT_PREDICT)
async def predict(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    domain: str = Form(""),  # noqa: B008
) -> JSONResponse:
    blob = await file.read()
    _validate_image(blob)
    name = file.filename or "image.jpg"
    chosen_domain = (domain or "").strip().lower() or None
    trace: list[dict[str, Any]] = []
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        # ─ Layer 1 ─
        try:
            l1 = await _post_image(client, LAYER1_URL, name, blob, file.content_type, chosen_domain)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Layer 1 unavailable: {e}") from e
        trace.append({"layer": 1, **l1})
        if l1.get("result") == "no_defect":
            return JSONResponse({
                "decision": "no_defect",
                "stopped_at_layer": 1,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
                "trace": trace,
            })

        # ─ Layer 2 ─
        try:
            l2 = await _post_image(client, LAYER2_URL, name, blob, file.content_type, chosen_domain)
        except httpx.HTTPError as e:
            logger.warning("Layer 2 failed, escalating to Layer 3: %s", e)
            l2 = {"result": "error", "confidence": 0.0}
        trace.append({"layer": 2, **l2})
        if l2.get("confidence", 0.0) >= L2_CONF_ESCALATE_BELOW and l2.get("result") != "error":
            return JSONResponse({
                "decision": "defect",
                "class": l2.get("class"),
                "stopped_at_layer": 2,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
                "trace": trace,
            })

        # ─ Layer 3 ─
        try:
            l3 = await _post_image(client, LAYER3_URL, name, blob, file.content_type, chosen_domain)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Layer 3 unavailable: {e}") from e
        trace.append({"layer": 3, **l3})
        return JSONResponse({
            "decision": l3.get("result", "unknown"),
            "class": l3.get("class"),
            "stopped_at_layer": 3,
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "trace": trace,
        })
