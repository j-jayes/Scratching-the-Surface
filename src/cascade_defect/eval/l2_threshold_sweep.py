"""Sweep Layer-2 confidence threshold on a trace to estimate cost/F1 tradeoff.

This expects a cascade trace where L1/L2/L3 decisions are recorded per frame.
For best fidelity, generate the trace with L3 enabled and a high
``--l2-conf-threshold`` (for example ``1.01``) so every post-L1 frame includes
an L3 fallback in the trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_TRACE = Path("reports/eval_cascade_metal.jsonl")
DEFAULT_OUT = Path("reports/l2_threshold_sweep.json")
DEFAULT_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
# Matches the λ used in threshold_sweep.py (knee objective F1 - λ*escalation_rate):
# this keeps the "best threshold" heuristic aligned with existing calibration logic.
L3_CALL_RATE_PENALTY = 0.05
PRICE_PER_INPUT_TOKEN = 0.40 / 1e6
PRICE_PER_OUTPUT_TOKEN = 1.60 / 1e6


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _trace_map(rec: dict) -> dict[int, dict]:
    return {int(step.get("layer")): step for step in rec.get("trace", []) if step.get("layer")}


def _truth_positive(rec: dict) -> bool:
    return rec.get("true_polarity") == "defective"


def _oracle_cost(step3: dict | None) -> float:
    if not step3:
        return 0.0
    usage = step3.get("usage", {})
    return (
        usage.get("prompt_tokens", 0) * PRICE_PER_INPUT_TOKEN
        + usage.get("completion_tokens", 0) * PRICE_PER_OUTPUT_TOKEN
    )


def _cost_adjusted_f1(row: dict) -> float:
    return row["f1"] - L3_CALL_RATE_PENALTY * row["l3_call_rate"]


def _decision_for_threshold(rec: dict, threshold: float) -> tuple[str, float, bool]:
    """Return (decision in {defect,no_defect,uncertain,error}, usd_cost, used_l3)."""
    steps = _trace_map(rec)
    l1 = steps.get(1)
    if l1 and l1.get("decision") == "no_defect":
        return "no_defect", 0.0, False

    l2 = steps.get(2)
    if l2:
        conf = float(l2.get("confidence") or 0.0)
        cls_name = l2.get("class")
        if cls_name and conf >= threshold:
            return "defect", 0.0, False

    l3 = steps.get(3)
    if l3:
        d = l3.get("decision")
        if d == "no_defect":
            return "no_defect", _oracle_cost(l3), True
        if d == "uncertain":
            return "uncertain", _oracle_cost(l3), True
        if d:
            return "defect", _oracle_cost(l3), True
        return "error", _oracle_cost(l3), True

    # Missing L3 fallback in trace for this threshold simulation.
    return "error", 0.0, False


def _score(records: list[dict], threshold: float, uncertain_mode: str) -> dict:
    tp = fp = tn = fn = uncertain = errors = 0
    n_l3 = 0
    total_cost = 0.0
    for rec in records:
        decision, usd_cost, used_l3 = _decision_for_threshold(rec, threshold)
        total_cost += usd_cost
        n_l3 += int(used_l3)
        truth_pos = _truth_positive(rec)
        if decision == "error":
            errors += 1
            continue
        if decision == "uncertain":
            uncertain += 1
            if uncertain_mode == "error":
                errors += 1
            elif truth_pos:
                fn += 1
            else:
                tn += 1
            continue
        pred_pos = decision == "defect"
        if pred_pos and truth_pos:
            tp += 1
        elif pred_pos and (not truth_pos):
            fp += 1
        elif (not pred_pos) and truth_pos:
            fn += 1
        else:
            tn += 1

    n = len(records)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    cost_per_100k = (total_cost / n) * 100000 if n else 0.0
    return {
        "threshold": threshold,
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "uncertain": uncertain,
        "errors": errors,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "l3_calls": n_l3,
        "l3_call_rate": round((n_l3 / n), 4) if n else 0.0,
        "oracle_cost_usd": round(total_cost, 6),
        "cost_per_100k_frames_usd": round(cost_per_100k, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--track", choices=["A", "B", "C"], default="A")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument(
        "--uncertain-mode",
        choices=["truth_aware", "error"],
        default="truth_aware",
        help=(
            "How to count L3='uncertain': truth_aware counts uncertain defects as FN "
            "and uncertain normals as TN; error counts all uncertain as errors."
        ),
    )
    args = ap.parse_args()

    records = [r for r in _load(args.trace) if r.get("track") == args.track]
    rows = [_score(records, t, args.uncertain_mode) for t in args.thresholds]
    best = max(rows, key=_cost_adjusted_f1) if rows else None

    payload = {
        "trace_path": str(args.trace),
        "track": args.track,
        "thresholds": args.thresholds,
        "uncertain_mode": args.uncertain_mode,
        "l3_call_rate_penalty": L3_CALL_RATE_PENALTY,
        "results": rows,
        "best_by_f1_minus_cost": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["best_by_f1_minus_cost"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
