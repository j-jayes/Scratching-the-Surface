"""Compute summary metrics from cascade + oracle-only eval JSONL files.

Writes `reports/metrics.json` and prints a summary.
gpt-4.1-mini pricing (as of 2025-04): $0.40/1M input, $1.60/1M output.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

REPORTS = Path("reports")
PRICE_IN = 0.40 / 1e6
PRICE_OUT = 1.60 / 1e6


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def summarise_cascade(recs: list[dict]) -> dict:
    total = len(recs)
    # All test images are defects, so any "no_defect" decision is a missed defect.
    # We also report top-1 only over the subset that the cascade actually classified
    # (i.e. escalated past Layer 1) — this is the "cascade-conditional" accuracy.
    classified = [r for r in recs if r.get("decision") not in (None, "no_defect", "error")]
    n_classified = len(classified)
    correct_overall = sum(1 for r in recs if r.get("class") == r["true_class"])
    correct_classified = sum(1 for r in classified if r.get("class") == r["true_class"])
    n_dropped_by_l1 = sum(1 for r in recs if r.get("decision") == "no_defect")
    by_layer = collections.Counter(r.get("stopped_at_layer") for r in recs)
    in_tok = out_tok = 0
    for r in recs:
        for t in r.get("trace", []):
            if t.get("layer") == 3 and "usage" in t:
                in_tok += t["usage"]["prompt_tokens"]
                out_tok += t["usage"]["completion_tokens"]
    cost = in_tok * PRICE_IN + out_tok * PRICE_OUT
    lats = [r.get("client_elapsed_ms", 0) for r in recs]
    confusion: dict[str, dict[str, int]] = {}
    for r in recs:
        confusion.setdefault(r["true_class"], {}).setdefault(r.get("class") or "no_defect", 0)
        confusion[r["true_class"]][r.get("class") or "no_defect"] += 1
    return {
        "n": total,
        "top1_accuracy_overall": correct_overall / total if total else 0.0,
        "top1_accuracy_classified_only": correct_classified / n_classified if n_classified else 0.0,
        "n_classified": n_classified,
        "n_dropped_by_l1": n_dropped_by_l1,
        "l1_drop_rate": n_dropped_by_l1 / total if total else 0.0,
        "stopped_at_layer": dict(by_layer),
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "oracle_cost_usd": round(cost, 6),
        "cost_per_100k_frames_usd": round(cost / total * 100_000, 2) if total else None,
        "latency_ms": {
            "mean": round(sum(lats) / total, 1) if total else 0,
            "p50": sorted(lats)[len(lats) // 2] if lats else 0,
            "p95": sorted(lats)[int(len(lats) * 0.95)] if lats else 0,
        },
        "confusion": confusion,
    }


def summarise_oracle(recs: list[dict]) -> dict:
    total = len(recs)
    correct = sum(1 for r in recs if r.get("class") == r["true_class"])
    in_tok = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in recs)
    out_tok = sum(r.get("usage", {}).get("completion_tokens", 0) for r in recs)
    cost = in_tok * PRICE_IN + out_tok * PRICE_OUT
    lats = [r.get("client_elapsed_ms", 0) for r in recs]
    return {
        "n": total,
        "top1_accuracy": correct / total if total else 0.0,
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "oracle_cost_usd": round(cost, 6),
        "cost_per_100k_frames_usd": round(cost / total * 100_000, 2) if total else None,
        "latency_ms": {
            "mean": round(sum(lats) / total, 1) if total else 0,
            "p50": sorted(lats)[len(lats) // 2] if lats else 0,
            "p95": sorted(lats)[int(len(lats) * 0.95)] if lats else 0,
        },
    }


def main() -> None:
    cascade = load(REPORTS / "eval_cascade.jsonl")
    oracle = load(REPORTS / "eval_oracle_only.jsonl")
    out = {
        "cascade": summarise_cascade(cascade),
        "oracle_only": summarise_oracle(oracle),
        "pricing": {"model": "gpt-4.1-mini", "usd_per_1m_in": 0.40, "usd_per_1m_out": 1.60},
    }
    if out["oracle_only"]["cost_per_100k_frames_usd"] and out["cascade"]["cost_per_100k_frames_usd"]:
        savings = (
            1 - out["cascade"]["cost_per_100k_frames_usd"] / out["oracle_only"]["cost_per_100k_frames_usd"]
        )
        out["cost_savings_pct"] = round(savings * 100, 1)
    metrics_path = REPORTS / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
