"""Provider-agnostic VLM clients (Azure OpenAI + OpenRouter)."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AzureOpenAI, OpenAI

from .prompt import DefectPrediction, build_messages

logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    provider: str
    model: str
    prediction: DefectPrediction | None
    latency_s: float
    in_tokens: int | None = None
    out_tokens: int | None = None
    cost_usd: float | None = None
    raw_text: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.prediction is not None and self.error is None


def _parse_json_object(text: str) -> dict | None:
    """Tolerate fenced markdown / prose around the JSON object."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return None


def _coerce_prediction(obj: dict | None) -> DefectPrediction | None:
    if obj is None:
        return None
    try:
        return DefectPrediction.model_validate(obj)
    except Exception:  # noqa: BLE001
        # Lenient fallback: clip confidence into [0,1], coerce missing fields.
        try:
            return DefectPrediction(
                defect_class=obj.get("defect_class", "uncertain"),
                confidence=max(0.0, min(1.0, float(obj.get("confidence", 0.0)))),
                reasoning=str(obj.get("reasoning", ""))[:500] or "n/a",
            )
        except Exception:  # noqa: BLE001
            return None


class VLMClient(ABC):
    """Abstract single-image classifier."""

    provider: str = "abstract"
    model: str = ""

    @abstractmethod
    def predict(self, image_path: Path, seed_dir: Path,
                *, domain: str | None = None) -> VLMResponse:
        ...


class AzureOpenAIClient(VLMClient):
    """Wraps the Azure OpenAI ``beta.chat.completions.parse`` structured-output API."""

    provider = "azure_openai"

    # GPT-4.1-mini list price (USD per 1M tokens, May 2026 — update if changes).
    DEFAULT_PRICE_IN = 0.15
    DEFAULT_PRICE_OUT = 0.60

    def __init__(self, deployment: str | None = None,
                 endpoint: str | None = None,
                 api_key: str | None = None,
                 api_version: str | None = None,
                 price_in: float | None = None,
                 price_out: float | None = None):
        endpoint = endpoint or os.environ.get("AOAI_ENDPOINT")
        api_key = api_key or os.environ.get("AOAI_API_KEY")
        api_version = (api_version or os.environ.get("AOAI_API_VERSION",
                                                     "2024-10-21"))
        if not endpoint or not api_key:
            raise RuntimeError(
                "AOAI_ENDPOINT and AOAI_API_KEY must be set in the environment."
            )
        self._client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key,
                                   api_version=api_version)
        self.model = deployment or os.environ.get("AOAI_DEPLOYMENT", "oracle")
        self.price_in = price_in if price_in is not None else self.DEFAULT_PRICE_IN
        self.price_out = price_out if price_out is not None else self.DEFAULT_PRICE_OUT

    def predict(self, image_path: Path, seed_dir: Path,
                *, domain: str | None = None) -> VLMResponse:
        messages = build_messages(image_path, seed_dir, domain=domain)
        t0 = time.perf_counter()
        try:
            resp = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=DefectPrediction,
                max_tokens=200,
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001
            return VLMResponse(provider=self.provider, model=self.model,
                               prediction=None,
                               latency_s=round(time.perf_counter() - t0, 3),
                               error=f"{type(exc).__name__}: {exc}")
        dt = round(time.perf_counter() - t0, 3)
        msg = resp.choices[0].message
        if msg.refusal:
            return VLMResponse(provider=self.provider, model=self.model,
                               prediction=None, latency_s=dt,
                               error=f"refusal: {msg.refusal}")
        in_tok = resp.usage.prompt_tokens if resp.usage else None
        out_tok = resp.usage.completion_tokens if resp.usage else None
        cost = None
        if in_tok is not None and out_tok is not None:
            cost = round(in_tok * self.price_in / 1e6
                         + out_tok * self.price_out / 1e6, 6)
        return VLMResponse(
            provider=self.provider, model=self.model,
            prediction=msg.parsed, latency_s=dt,
            in_tokens=in_tok, out_tokens=out_tok, cost_usd=cost,
            raw_text=msg.content,
        )


class OpenRouterClient(VLMClient):
    """OpenAI-compatible client pointed at ``openrouter.ai``.

    OpenRouter doesn't (yet) honour ``response_format`` for arbitrary models, so
    we ask for plain JSON in the prompt and parse it ourselves.
    """

    provider = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str,
                 price_in: float, price_out: float,
                 api_key: str | None = None):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY must be set in the environment.")
        self._client = OpenAI(base_url=self.BASE_URL, api_key=api_key)
        self.model = model
        self.price_in = price_in
        self.price_out = price_out

    def predict(self, image_path: Path, seed_dir: Path,
                *, domain: str | None = None) -> VLMResponse:
        messages = build_messages(image_path, seed_dir, domain=domain)
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001
            return VLMResponse(provider=self.provider, model=self.model,
                               prediction=None,
                               latency_s=round(time.perf_counter() - t0, 3),
                               error=f"{type(exc).__name__}: {exc}")
        dt = round(time.perf_counter() - t0, 3)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        in_tok = resp.usage.prompt_tokens if resp.usage else None
        out_tok = resp.usage.completion_tokens if resp.usage else None
        cost = None
        if in_tok is not None and out_tok is not None:
            cost = round(in_tok * self.price_in / 1e6
                         + out_tok * self.price_out / 1e6, 6)
        parsed = _coerce_prediction(_parse_json_object(text))
        err = None if parsed else "json_parse_failed"
        return VLMResponse(
            provider=self.provider, model=self.model,
            prediction=parsed, latency_s=dt,
            in_tokens=in_tok, out_tokens=out_tok, cost_usd=cost,
            raw_text=text[:1000], error=err,
        )
