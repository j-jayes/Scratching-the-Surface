"""Provider-agnostic Vision-Language Model client + few-shot prompt builder.

This package was extracted from ``layer3_gpt4o.oracle`` so the same prompt and
schema can target multiple back-ends (Azure OpenAI, OpenRouter, future
self-hosted vLLM endpoints) with one call site.
"""

from .prompt import (
    DefectPrediction,
    METAL_CLASSES,
    NEU_CLASSES,
    SYSTEM_PROMPT,
    build_messages,
)
from .client import (
    AzureOpenAIClient,
    OpenRouterClient,
    VLMClient,
    VLMResponse,
)

__all__ = [
    "DefectPrediction",
    "METAL_CLASSES",
    "NEU_CLASSES",
    "SYSTEM_PROMPT",
    "build_messages",
    "AzureOpenAIClient",
    "OpenRouterClient",
    "VLMClient",
    "VLMResponse",
]
