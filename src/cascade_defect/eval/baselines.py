"""Compare cascade vs single-layer baselines on the same metal frames.

Reads four JSONL traces (cascade, L1-only, L2-only, L3-only) and emits
``reports/baselines_metal.json`` with apples-to-apples F1 / cost / latency
per system, with bootstrap 95 % CIs and a McNemar test of the cascade
against the pure-Oracle baseline on the frames they share.

Why this module exists: the v1 narrative ("6\u201320\u00d7 cheaper than pure Oracle")
was just text on the website. After this runs, it's a numerical table with
confidence intervals and a paired significance test.

Usage::

    uv run python -m cascade_defect.eval.baselines
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from cascade_defect.eval import stats as _stats

REPORTS = Path("reports")
PRICE_IN = 0.40 / 1e6  # gpt-4.1-mini, USD per token
PRICE_OUT = 1.60 / 1e6

SYSTEMS = {
    "cascade":     REPORTS / "eval_cascade_metal_k1_calibrated.jsonl",
    "l1_only":     REPORTS / "baseline_l1_only.jsonl",
    "yolo_only":   REPORTS / "baseline_yolo_only.jsonl",
    "oracle_only": REPORTS / "baseline_oracle_only_metal.jsonl",
}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _normalised_decision(rec: dict, system: str) -> str:
    """Map any baseline's idiosyncratic ``decision`` to {defect, no_defect, uncertain, error}.

    The YOLO-only baseline trace has a known quirk: when YOLO returns no boxes,
    the runner fills ``decision=defect`` / ``class=defect_unclassified`` because
    in the real cascade the absent-L1 case escalates to L3. For the *yolo_only*
    baseline we override and trust the L2 trace entry.
    """
    if system == "yolo_only":
        for t in rec.get("trace", []):
            if t.get("layer") == 2:
                return "defect" if t.get("decision") == "defect" else "no_defect"
        return "no_defect"
    d = rec.get("decision", "error")
    if d in ("error", "uncertain"):
        return d
    return "defect" if d == "defect" else "no_defect"


def _patched(recs: Sequence[dict], system: str) -> list[dict]:
    out = []
    for r in recs:
        r2 = dict(r)
        r2["decision"] = _normalised_decision(r, system)
        out.append(r2)
    return out


def _summary(recs: Sequence[dict]) -> dict:
    if not recs:
        return {"n": 0}
    f1 = _stats.bootstrap_ci(recs, _stats.f1_stat)
    p = _stats.bootstrap_ci(recs, _stats.precision_stat)
    r = _stats.bootstrap_ci(recs, _stats.recall_stat)
    cost = _stats.cost_per_100k_stat(recs, PRICE_IN, PRICE_OUT)
    lats = sorted(rec.get("client_elapsed_ms", 0) for rec in recs)
    return {
        "n": len(recs),
        "f1": round(f1.point, 4),
        "f1_ci95": [round(f1.lo, 4), round(f1.hi, 4)],
        "precision": round(p.point, 4),
        "precision_ci95": [round(p.lo, 4), round(p.hi, 4)],
        "recall": round(r.point, 4),
        "recall_ci95": [round(r.lo, 4), round(r.hi, 4)],
        "cost_per_100k_frames_usd": round(cost, 2),
        "latency_ms": {
            "p50": int(lats[len(lats) // 2]),
            "p95": int(lats[min(len(lats) - 1, int(len(lats) * 0.95))]),
        },
    }


def _mcnemar_against_oracle(cascade: Sequence[dict], oracle: Sequence[dict]) -> dict:
    """McNemar: cascade vs oracle on the *same* image, both decided correctly?"""
    by_image_oracle = {r["image"]: r for r in oracle}
    b = c = both_right = both_wrong = paired = 0
    for r in cascade:
        o = by_image_oracle.get(r["image"])
        if o is None:
            continue
        paired += 1
        truth = r.get("true_polarity") == "defective"
        cas_right = (_normalised_decision(r, "cascade") == "defect") == truth
        ora_right = (_normalised_decision(o, "oracle_only") == "defect") == truth
        if cas_right and ora_right:
            both_right += 1
        elif (not cas_right) and (not ora_right):
            both_wrong += 1
        elif cas_right and not ora_right:
            b += 1
        elif (not cas_right) and ora_right:
            c += 1
    chi, p = _stats.mcnemar(b, c)
    return {
        "n_paired": paired,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "cascade_only_correct": b,
        "oracle_only_correct": c,
        "chi_square": None if chi != chi else round(chi, 3),  # NaN check (exact test)
        "p_value": round(p, 4),
        "interpretation": (
            "no significant difference" if p >= 0.05
            else ("cascade significantly better" if b > c else "oracle significantly better")
        ),
    }


def main() -> None:
    out: dict = {"systems": {}, "by_track": {}, "missing": []}

    by_system: dict[str, list[dict]] = {}
    for system, path in SYSTEMS.items():
        recs = _load(path)
        if not recs:
            out["missing"].append({"system": system, "path": str(path)})
            continue
        recs = _patched(recs, system)
        by_system[system] = recs
        # Overall summary across all tracks the baseline ran on.
        out["systems"][system] = _summary(recs)

    # Per-track breakout (only for systems that ran on that track).
    for system, recs in by_system.items():
        for track in sorted({r.get("track", "?") for r in recs}):
            out["by_track"].setdefault(track, {})[system] = _summary(
                [r for r in recs if r.get("track") == track]
            )

    # Paired significance: cascade vs oracle-only on shared frames.
    if "cascade" in by_system and "oracle_only" in by_system:
        out["mcnemar_cascade_vs_oracle"] = _mcnemar_against_oracle(
            by_system["cascade"], by_system["oracle_only"]
        )

    OUT = REPORTS / "baselines_metal.json"
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
