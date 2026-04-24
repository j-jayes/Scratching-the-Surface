"""Cascade router — the single ingress endpoint.

Orchestrates the three layers:
1. Forwards the image to Layer 1 (autoencoder gatekeeper).
2. If Layer 1 says "no_defect" → return immediately. Cheap path.
3. Otherwise forwards to Layer 2 (YOLO specialist).
4. If Layer 2's confidence is below threshold → forwards to Layer 3 (Oracle).

Returns a per-layer trace so the caller can see the cascade decision path.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Cascade Router", version="0.1.0")

LAYER1_URL = os.getenv("LAYER1_URL", "http://layer1:8000")
LAYER2_URL = os.getenv("LAYER2_URL", "http://layer2:8000")
LAYER3_URL = os.getenv("LAYER3_URL", "http://layer3:8000")
L2_CONF_ESCALATE_BELOW = float(os.getenv("L2_CONF_ESCALATE_BELOW", "0.7"))


async def _post_image(client: httpx.AsyncClient, url: str, name: str, blob: bytes, content_type: str | None) -> dict[str, Any]:
    files = {"file": (name, blob, content_type or "image/jpeg")}
    r = await client.post(f"{url}/predict", files=files, timeout=60)
    r.raise_for_status()
    return r.json()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "layers": {"l1": LAYER1_URL, "l2": LAYER2_URL, "l3": LAYER3_URL}}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    blob = await file.read()
    name = file.filename or "image.jpg"
    trace: list[dict[str, Any]] = []
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        # ─ Layer 1 ─
        try:
            l1 = await _post_image(client, LAYER1_URL, name, blob, file.content_type)
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
            l2 = await _post_image(client, LAYER2_URL, name, blob, file.content_type)
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
            l3 = await _post_image(client, LAYER3_URL, name, blob, file.content_type)
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
