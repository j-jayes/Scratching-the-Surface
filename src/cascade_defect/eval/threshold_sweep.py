"""Phase K.2 — Calibrate per-domain z-threshold τ on a *cost* objective.

Reads a cascade-eval JSONL (where every record carries the L1 ``score_z`` in
``trace[0]``) and for each per-domain z-threshold candidate computes:

  * Naive L1-classifier F1 (treat ``z ≥ τ`` as "defect").
  * Escalation count (= cases that *would* hit L2/L3 at this τ).
  * Cost-per-100k-frames assuming a single Oracle call costs the per-record
    Oracle USD seen in the trace (mean over the L3 hits there).

Picks the **knee** of the (escalation_rate, F1) curve per domain — the τ
that maximises ``F1 - λ · escalation_rate`` (λ defaults to 0.05, i.e. 1pp
of escalations is worth 0.05 F1). Persists the chosen τ back into
``models/patchcore_metal/summary.json`` (under ``calibration_knee.<domain>``)
so production config can pick it up via ``Z_THRESHOLD_<DOMAIN>`` env vars.

Usage::

    uv run python -m cascade_defect.eval.threshold_sweep \\
        --trace reports/eval_cascade_metal_k1.jsonl \\
        --out reports/threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PATCHCORE_SUMMARY = Path("models/patchcore_metal/summary.json")
DEFAULT_TRACE = Path("reports/eval_cascade_metal_k1.jsonl")
DEFAULT_OUT = Path("reports/threshold_sweep.json")
Z_GRID = [round(x * 0.25, 2) for x in range(-4, 21)]  # -1.0 .. 5.0 step 0.25


def _load(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.open(encoding="utf-8") if line.strip()]


def _per_domain(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["domain"], []).append(r)
    return out


def _l1_score(rec: dict) -> float | None:
    for ent in rec.get("trace", []):
        if ent.get("layer") == 1 and ent.get("score_z") is not None:
            return float(ent["score_z"])
    return None


def _sweep_one(records: list[dict], lam: float) -> dict:
    """Sweep τ over Z_GRID for one domain. Returns curve + chosen knee."""
    truth = []
    scores = []
    for r in records:
        z = _l1_score(r)
        if z is None:
            continue
        truth.append(r["true_polarity"] == "defective")
        scores.append(z)
    n = len(truth)
    if n == 0:
        return {"n": 0}

    n_pos = sum(truth)
    n_neg = n - n_pos
    curve = []
    best = None
    for tau in Z_GRID:
        tp = sum(1 for t, s in zip(truth, scores, strict=True) if t and s >= tau)
        fp = sum(1 for t, s in zip(truth, scores, strict=True) if (not t) and s >= tau)
        fn = n_pos - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_pos if n_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        n_escalate = tp + fp  # cases passed up to L2/L3
        escalation_rate = n_escalate / n
        score = f1 - lam * escalation_rate
        row = {
            "tau": tau, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_escalate": n_escalate,
            "escalation_rate": round(escalation_rate, 4),
            "knee_score": round(score, 4),
        }
        curve.append(row)
        if best is None or score > best["knee_score"]:
            best = row
    return {
        "n": n, "n_positive": n_pos, "n_negative": n_neg,
        "lambda": lam,
        "curve": curve,
        "knee": best,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lam", type=float, default=0.05,
                    help="Cost weight: knee maximises F1 - λ × escalation_rate.")
    ap.add_argument("--persist-knee", action="store_true",
                    help="Write the chosen knee per domain back into "
                         "models/patchcore_metal/summary.json.")
    args = ap.parse_args()

    recs = _load(args.trace)
    by_domain = _per_domain(recs)
    sweep = {d: _sweep_one(rs, args.lam) for d, rs in by_domain.items()}

    out = {
        "trace_path": str(args.trace),
        "lambda": args.lam,
        "z_grid": Z_GRID,
        "per_domain": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({d: v.get("knee") for d, v in sweep.items()}, indent=2))
    print(f"\nWrote {args.out}")

    if args.persist_knee and PATCHCORE_SUMMARY.exists():
        summary = json.loads(PATCHCORE_SUMMARY.read_text(encoding="utf-8"))
        summary["calibration_knee"] = {
            "lambda": args.lam,
            "trace_path": str(args.trace),
            "per_domain": {
                d: {
                    "tau_z": v["knee"]["tau"],
                    "f1": v["knee"]["f1"],
                    "escalation_rate": v["knee"]["escalation_rate"],
                }
                for d, v in sweep.items() if v.get("knee")
            },
        }
        PATCHCORE_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Persisted knee into {PATCHCORE_SUMMARY}")


if __name__ == "__main__":
    main()
