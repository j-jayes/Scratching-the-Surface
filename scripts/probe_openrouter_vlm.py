"""Phase 0a — Probe OpenRouter VLMs for image-input capability + JSON adherence.

Sends one defective and one normal Severstal image to a small set of candidate
models and records: success/failure, latency, in/out tokens, $ cost, parsed
defect_class. Output: ``reports/openrouter_vlm_probe.json``.

Run::

    uv run python scripts/probe_openrouter_vlm.py

Requires ``OPENROUTER_API_KEY`` in the environment / ``.env``.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "openrouter_vlm_probe.json"
SAMPLES = [
    ("defective", ROOT / "data/splits_metal/cascade_test/severstal/defective/0007a71bf.jpg"),
    ("normal", ROOT / "data/splits_metal/cascade_test/severstal/normal/000789191.jpg"),
]

# (model_id, advertised_input_modality, $/M_in, $/M_out)
CANDIDATES: list[tuple[str, str, float, float]] = [
    ("deepseek/deepseek-v4-pro", "text(?)", 0.435, 0.87),
    ("deepseek/deepseek-v4-flash", "text(?)", 0.126, 0.252),
    ("qwen/qwen3.6-35b-a3b", "text+image+video", 0.15, 1.00),
    ("qwen/qwen3.5-plus-20260420", "text+image+video", 0.40, 2.40),
    ("qwen/qwen3-vl-30b-a3b-instruct", "text+image+video", 0.13, 0.52),
]

SYSTEM = (
    "You are an expert quality-control inspector for flat-rolled steel sheet. "
    "Classify the image into one of: pitting, inclusion, scratch, patch, "
    "surface_anomaly, no_defect, uncertain. "
    "Respond ONLY with a single JSON object on one line with keys "
    '"defect_class" (string), "confidence" (0-1 float), "reasoning" (string).'
)


@dataclass
class ProbeResult:
    model: str
    sample: str
    ok: bool
    latency_s: float | None = None
    in_tokens: int | None = None
    out_tokens: int | None = None
    cost_usd: float | None = None
    raw_text: str | None = None
    parsed_class: str | None = None
    error: str | None = None


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def probe(client: OpenAI, model: str, label: str, img: Path,
          price_in: float, price_out: float) -> ProbeResult:
    res = ProbeResult(model=model, sample=label, ok=False)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}"},
                },
                {"type": "text", "text": "Classify this image. JSON only."},
            ],
        },
    ]
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=200, temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        res.latency_s = round(time.perf_counter() - t0, 2)
        return res

    res.latency_s = round(time.perf_counter() - t0, 2)
    txt = resp.choices[0].message.content or ""
    res.raw_text = txt[:500]
    if resp.usage:
        res.in_tokens = resp.usage.prompt_tokens
        res.out_tokens = resp.usage.completion_tokens
        res.cost_usd = round(
            res.in_tokens * price_in / 1e6 + res.out_tokens * price_out / 1e6, 6
        )
    # try to parse the JSON
    try:
        # tolerate fenced or prefix text — find first '{' .. last '}'
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(txt[start: end + 1])
            res.parsed_class = obj.get("defect_class")
            res.ok = True
    except Exception as exc:  # noqa: BLE001
        res.error = f"parse: {type(exc).__name__}: {exc}"
    return res


def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        timeout=60.0,
        max_retries=1,
    )

    results: list[dict] = []
    for model, modality, p_in, p_out in CANDIDATES:
        print(f"\n=== {model} (advertised: {modality}) ===")
        for label, path in SAMPLES:
            r = probe(client, model, label, path, p_in, p_out)
            tag = "OK " if r.ok else "FAIL"
            print(
                f"  [{tag}] {label:9s} {r.latency_s}s "
                f"in={r.in_tokens} out={r.out_tokens} ${r.cost_usd} "
                f"class={r.parsed_class!r} err={r.error}"
            )
            results.append({"advertised": modality, **r.__dict__})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
