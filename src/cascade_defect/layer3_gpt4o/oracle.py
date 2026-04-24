"""Shared GPT-4.1-mini Oracle client + few-shot prompt builder.

Used by both the FastAPI app (online cascade) and the offline batch annotator.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Literal

from openai import AzureOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
DefectClass = Literal[
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
    "no_defect",
]


class DefectPrediction(BaseModel):
    """Structured JSON schema enforced on Oracle responses."""

    defect_class: DefectClass = Field(description="One of the six NEU classes or 'no_defect'.")
    confidence: float = Field(ge=0.0, le=1.0, description="Subjective confidence 0–1.")
    reasoning: str = Field(description="One-sentence rationale.")


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────
def get_client() -> AzureOpenAI:
    endpoint = os.environ.get("AOAI_ENDPOINT")
    api_key = os.environ.get("AOAI_API_KEY")
    api_version = os.environ.get("AOAI_API_VERSION", "2024-10-21")
    if not endpoint or not api_key:
        raise RuntimeError("AOAI_ENDPOINT and AOAI_API_KEY must be set in .env.")
    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)


def get_deployment() -> str:
    return os.environ.get("AOAI_DEPLOYMENT", "oracle")


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot prompt
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert quality-control inspector for rolled steel manufacturing. "
    "You classify surface defects in greyscale steel images into one of six classes: "
    f"{', '.join(NEU_CLASSES)}. Use 'no_defect' if no defect is visible. "
    "You will see a few labelled reference examples followed by an unknown image. "
    "Respond ONLY in the required JSON schema."
)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_messages(image_path: Path, seed_dir: Path) -> list[dict]:
    """Build the multimodal message list with few-shot exemplars + query image."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if seed_dir.exists():
        for class_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            for img in sorted(class_dir.glob("*.jpg"))[:3]:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{_b64(img)}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": f"Reference: defect class = {class_dir.name}"},
                    ],
                })
                messages.append({
                    "role": "assistant",
                    "content": (
                        '{"defect_class": "' + class_dir.name + '", '
                        '"confidence": 0.97, '
                        '"reasoning": "Provided reference example."}'
                    ),
                })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_b64(image_path)}",
                    "detail": "low",
                },
            },
            {"type": "text", "text": "Classify this image. Respond ONLY in the JSON schema."},
        ],
    })
    return messages


def predict(image_path: Path, seed_dir: Path) -> tuple[DefectPrediction, dict]:
    """Single-image prediction. Returns (parsed prediction, raw usage dict)."""
    client = get_client()
    response = client.beta.chat.completions.parse(
        model=get_deployment(),
        messages=build_messages(image_path, seed_dir),
        response_format=DefectPrediction,
        max_tokens=200,
        temperature=0,
    )
    msg = response.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"Model refused: {msg.refusal}")
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return msg.parsed, usage
