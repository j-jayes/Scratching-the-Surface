"""Layer 3 — GPT-4o Oracle endpoint.

Handles low-confidence edge cases escalated by Layer 2.
Uses Azure OpenAI structured outputs (Pydantic schema) to enforce JSON responses.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from openai import AzureOpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Layer 3 — GPT-4o Oracle", version="0.1.0")

# ── Azure OpenAI configuration ────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

FEW_SHOT_SEED_DIR = Path(os.getenv("FEW_SHOT_SEED_DIR", "data/splits/seed"))

# ── Pydantic schema for structured output ────────────────────────────────────
NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


class DefectPrediction(BaseModel):
    """Structured JSON output enforced on GPT-4o responses."""

    defect_class: str = Field(
        description=f"One of: {', '.join(NEU_CLASSES)}. Use 'no_defect' if no defect is visible."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Subjective confidence score 0–1")
    reasoning: str = Field(description="One-sentence explanation for the classification decision")
    bounding_box_present: bool = Field(description="Whether a visible defect region was identified")


def _build_few_shot_messages(image_b64: str) -> list[dict]:
    """Build the GPT-4o message list with few-shot seed examples."""
    system_content = (
        "You are an expert quality-control inspector for rolled steel manufacturing. "
        "Your task is to classify surface defects in greyscale steel surface images. "
        "You will be given a few reference examples followed by an unknown image. "
        "Respond strictly in the requested JSON format — no extra text.\n\n"
        "Defect classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.\n"
        "If no defect is visible, use 'no_defect'."
    )
    messages: list[dict] = [{"role": "system", "content": system_content}]

    # Load few-shot seed images from disk (3 per class = 18 total)
    if FEW_SHOT_SEED_DIR.exists():
        for class_dir in sorted(FEW_SHOT_SEED_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*.jpg"))[:3]:
                with img_path.open("rb") as f:
                    seed_b64 = base64.b64encode(f.read()).decode()
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{seed_b64}"}},
                        {"type": "text", "text": f"This is an example of defect class: {class_dir.name}"},
                    ],
                })
                messages.append({"role": "assistant", "content": f'{{"defect_class": "{class_dir.name}", "confidence": 0.95, "reasoning": "Reference example.", "bounding_box_present": true}}'})

    # Add the query image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "Classify this image. Respond only in the required JSON format."},
        ],
    })
    return messages


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "deployment": AZURE_OPENAI_DEPLOYMENT}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode()

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    messages = _build_few_shot_messages(image_b64)

    response = client.beta.chat.completions.parse(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        response_format=DefectPrediction,
        max_tokens=300,
        temperature=0,
    )

    prediction = response.choices[0].message.parsed
    return JSONResponse({
        "result": "defect_detected" if prediction.defect_class != "no_defect" else "no_defect",
        "class": prediction.defect_class,
        "confidence": prediction.confidence,
        "reasoning": prediction.reasoning,
        "bounding_box_present": prediction.bounding_box_present,
        "layer": 3,
    })
