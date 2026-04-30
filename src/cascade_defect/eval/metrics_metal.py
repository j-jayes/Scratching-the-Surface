"""Roll the per-image JSONL trace from ``run_cascade_metal`` into per-track
summary metrics → ``reports/metrics_metal.json``.

This is the apples-to-apples companion to ``metrics.py`` (NEU v1). The cascade
test set has both polarities so we can compute proper precision / recall / F1
on the binary defect/no-defect decision, plus per-track L1 drop rate, latency
percentiles, and Oracle cost.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

REPORTS = Path("reports")
TRACE_PATH = REPORTS / "eval_cascade_metal.jsonl"
OUT_PATH = REPORTS / "metrics_metal.json"

# gpt-4.1-mini pricing (as of 2025-04). Match metrics.py.
PRICE_IN = 0.40 / 1e6
PRICE_OUT = 1.60 / 1e6


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(len(s) * p))
    return float(s[idx])


def _binary_decision(rec: dict) -> str:
    """Map cascade decision → {defect, no_defect, uncertain, error}."""
    d = rec.get("decision", "error")
    if d in ("error", "uncertain"):
        return d
    return "defect" if d == "defect" else "no_defect"


def summarise_track(recs: list[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {"n": 0}

    # Binary (defect / no_defect). Treat "uncertain" as a separate column.
    tp = fp = tn = fn = uncertain = error = 0
    # Per-class confusion (Phase K.5) — only on actual defectives, keyed on
    # the cascade's final class label.
    class_confusion: collections.Counter = collections.Counter()
    for r in recs:
        true_pos = r.get("true_polarity") == "defective"
        decision = _binary_decision(r)
        if decision == "uncertain":
            uncertain += 1
        elif decision == "error":
            error += 1
        elif true_pos and decision == "defect":
            tp += 1
        elif true_pos and decision == "no_defect":
            fn += 1
        elif (not true_pos) and decision == "defect":
            fp += 1
        else:
            tn += 1
        if true_pos:
            class_confusion[r.get("class", "-")] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # L1 drop = stopped at L1 with no_defect (i.e. AE/PatchCore short-circuit).
    n_dropped_l1 = sum(
        1 for r in recs
        if r.get("stopped_at_layer") == 1 and r.get("decision") == "no_defect"
    )
    n_negatives = sum(1 for r in recs if r.get("true_polarity") == "normal")
    l1_drop_rate_on_negatives = (
        sum(1 for r in recs
            if r.get("true_polarity") == "normal"
            and r.get("stopped_at_layer") == 1
            and r.get("decision") == "no_defect") / n_negatives
        if n_negatives else 0.0
    )

    by_layer = collections.Counter(r.get("stopped_at_layer") for r in recs)

    # Oracle cost — sum any L3 usage entries in trace.
    in_tok = out_tok = 0
    cache_hits = cache_misses = 0
    for r in recs:
        for t in r.get("trace", []):
            if t.get("layer") == 3:
                if "usage" in t:
                    in_tok += t["usage"].get("prompt_tokens", 0)
                    out_tok += t["usage"].get("completion_tokens", 0)
                cache = t.get("cache") or {}
                if cache.get("hit") is True:
                    cache_hits += 1
                elif cache.get("hit") is False:
                    cache_misses += 1
    cost = in_tok * PRICE_IN + out_tok * PRICE_OUT

    lats = [r.get("client_elapsed_ms", 0) for r in recs]
    return {
        "n": n,
        "n_positive_truth": tp + fn,
        "n_negative_truth": fp + tn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "uncertain": uncertain, "error": error,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_dropped_by_l1": n_dropped_l1,
        "l1_drop_rate_overall": round(n_dropped_l1 / n, 4),
        "l1_drop_rate_on_negatives": round(l1_drop_rate_on_negatives, 4),
        "stopped_at_layer": dict(by_layer),
        "tokens_in": in_tok,
        "tokens_out": out_tok,
        "oracle_cost_usd": round(cost, 6),
        "cost_per_100k_frames_usd": round(cost / n * 100_000, 2) if n else None,
        "latency_ms": {
            "mean": round(sum(lats) / n, 1),
            "p50": int(_percentile(lats, 0.50)),
            "p95": int(_percentile(lats, 0.95)),
        },
        "oracle_cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hits / (cache_hits + cache_misses), 4)
            if (cache_hits + cache_misses) else 0.0,
        },
        "class_confusion_on_defectives": dict(class_confusion),
    }


def main() -> None:
    recs = load(TRACE_PATH)
    by_track: dict[str, list[dict]] = {}
    for r in recs:
        by_track.setdefault(r.get("track", "?"), []).append(r)

    out = {
        "trace_path": str(TRACE_PATH),
        "n_total": len(recs),
        "tracks": {t: summarise_track(by_track[t]) for t in sorted(by_track)},
        "pricing": {"model": "gpt-4.1-mini", "usd_per_1m_in": 0.40, "usd_per_1m_out": 1.60},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
