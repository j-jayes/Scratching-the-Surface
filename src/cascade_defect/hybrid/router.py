"""Phase 3 — Hybrid router that combines ResNet50 (classical) and a VLM.

The router consumes the cached predictions produced by Phases 1c and 2 — it does
**not** re-call any model — and applies a configurable fusion policy to score the
hybrid system on the same 240-image benchmark used for the VLM.

Three policies are implemented:

* ``"escalate"`` — accept the ResNet50 label whenever its softmax confidence is
  above ``--threshold``, otherwise defer to the VLM. This is the operationally
  realistic mode (cheap most of the time, expensive when uncertain).
* ``"vote"`` — agree → accept; disagree → defer to whichever model has higher
  confidence.
* ``"vlm_first"`` — always trust VLM unless it returns ``"uncertain"`` /
  ``"unknown"``. Useful as an upper bound on VLM-only behaviour.

Inputs:
- ``reports/resnet50_metal.json`` (per-image predictions block — see eval.py)
- ``reports/vlm_bench_metal_traces.jsonl``

Outputs:
- ``reports/hybrid_metal.json`` — same metric schema as the other two so the
  three-way headline table in the slides can be assembled mechanically.

Run::

    uv run python -m cascade_defect.hybrid.router \
        --vlm-provider azure_openai/oracle \
        --policy escalate --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]
CLASSES = ["no_defect", "pitting", "inclusion", "scratch", "patch"]


# ── Data loading ────────────────────────────────────────────────────────────
@dataclass
class ResnetPrediction:
    image: str
    true_label: str
    pred_label: str
    confidence: float
    probs: dict[str, float]


@dataclass
class VlmPrediction:
    image: str
    true_label: str
    pred_label: str          # already normalised to canonical taxonomy
    confidence: float
    latency_s: float
    cost_usd: float


def load_resnet_predictions(path: Path) -> dict[str, ResnetPrediction]:
    payload = json.loads(path.read_text())
    out: dict[str, ResnetPrediction] = {}
    for row in payload["per_image"]:
        out[row["image"]] = ResnetPrediction(
            image=row["image"],
            true_label=row["true_label"],
            pred_label=row["pred_label"],
            confidence=float(row["confidence"]),
            probs={c: float(row["probs"][c]) for c in CLASSES},
        )
    return out


def load_vlm_predictions(traces_path: Path, provider: str) -> dict[str, VlmPrediction]:
    out: dict[str, VlmPrediction] = {}
    for line in traces_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prov_key = f"{rec.get('provider')}/{rec.get('model')}"
        if prov_key != provider and rec.get("provider") != provider:
            continue
        if rec.get("error"):
            continue
        pred = rec.get("predicted_class")
        if pred is None:
            continue
        out[rec["image_path"]] = VlmPrediction(
            image=rec["image_path"],
            true_label=rec["true_class"],
            pred_label=pred,
            confidence=float(rec.get("confidence") or 0.0),
            latency_s=float(rec["latency_s"]),
            cost_usd=float(rec.get("cost_usd") or 0.0),
        )
    return out


# ── Fusion policies ─────────────────────────────────────────────────────────
Policy = Literal["escalate", "vote", "vlm_first"]


def fuse_one(
    rn: ResnetPrediction,
    vlm: VlmPrediction | None,
    policy: Policy,
    threshold: float,
) -> tuple[str, str, bool]:
    """Return (predicted_label, source, escalated_to_vlm)."""
    if policy == "escalate":
        if vlm is None or rn.confidence >= threshold:
            return rn.pred_label, "resnet50", False
        return vlm.pred_label, "vlm", True

    if policy == "vote":
        if vlm is None:
            return rn.pred_label, "resnet50", False
        if rn.pred_label == vlm.pred_label:
            return rn.pred_label, "agree", False
        if rn.confidence >= vlm.confidence:
            return rn.pred_label, "resnet50", False
        return vlm.pred_label, "vlm", True

    if policy == "vlm_first":
        if vlm is None or vlm.pred_label in {"uncertain", "unknown"}:
            return rn.pred_label, "resnet50", False
        return vlm.pred_label, "vlm", True

    raise ValueError(f"Unknown policy: {policy}")


# ── Metrics (mirrors eval.py + bench_vlm.py) ────────────────────────────────
def per_class_prf(records: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for c in CLASSES:
        tp = sum(1 for r in records if r["pred"] == c and r["true"] == c)
        fp = sum(1 for r in records if r["pred"] == c and r["true"] != c)
        fn = sum(1 for r in records if r["pred"] != c and r["true"] == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def binary_defect_metrics(records: list[dict]) -> dict:
    tp = sum(1 for r in records if r["pred"] != "no_defect" and r["true"] != "no_defect")
    fp = sum(1 for r in records if r["pred"] != "no_defect" and r["true"] == "no_defect")
    tn = sum(1 for r in records if r["pred"] == "no_defect" and r["true"] == "no_defect")
    fn = sum(1 for r in records if r["pred"] == "no_defect" and r["true"] != "no_defect")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ── Driver ──────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--resnet-report", default="reports/resnet50_metal.json")
    p.add_argument("--vlm-traces", default="reports/vlm_bench_metal_traces.jsonl")
    p.add_argument("--vlm-provider", default="azure_openai/oracle",
                   help="Provider key as stored in trace JSONL.")
    p.add_argument("--policy", choices=["escalate", "vote", "vlm_first"],
                   default="escalate")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="ResNet50 confidence threshold for the 'escalate' policy.")
    p.add_argument("--report", default="reports/hybrid_metal.json")
    args = p.parse_args()

    rn_preds = load_resnet_predictions(ROOT / args.resnet_report)
    vlm_preds = load_vlm_predictions(ROOT / args.vlm_traces, args.vlm_provider)

    # Restrict to the intersection (the 240-image VLM bench).
    keys = sorted(set(rn_preds) & set(vlm_preds))
    print(f"ResNet50 preds: {len(rn_preds)}  VLM preds: {len(vlm_preds)}  intersection: {len(keys)}")

    fused: list[dict] = []
    n_escalated = 0
    cost_per_call = (sum(v.cost_usd for v in vlm_preds.values()) / max(1, len(vlm_preds)))
    lat_per_call = (sum(v.latency_s for v in vlm_preds.values()) / max(1, len(vlm_preds)))
    for k in keys:
        rn = rn_preds[k]
        vlm = vlm_preds.get(k)
        pred, source, escalated = fuse_one(rn, vlm, args.policy, args.threshold)
        n_escalated += int(escalated)
        fused.append({
            "image": k,
            "true": rn.true_label,
            "pred": pred,
            "source": source,
            "escalated": escalated,
        })

    correct = sum(1 for r in fused if r["pred"] == r["true"])
    accuracy = correct / max(1, len(fused))
    macro_f1 = sum(c["f1"] for c in per_class_prf(fused).values()) / len(CLASSES)
    bin_metrics = binary_defect_metrics(fused)

    n = len(fused)
    out = {
        "config": {
            "policy": args.policy,
            "threshold": args.threshold,
            "vlm_provider": args.vlm_provider,
            "n_samples": n,
            "n_escalated": n_escalated,
            "escalation_rate": n_escalated / max(1, n),
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class_prf(fused),
            "binary_defect": bin_metrics,
        },
        "estimated_cost_per_image_usd": (n_escalated / max(1, n)) * cost_per_call,
        "estimated_avg_latency_s": (
            (n - n_escalated) * 0.05 + n_escalated * lat_per_call
        ) / max(1, n),
    }

    out_path = ROOT / args.report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=2))
    print(f"accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}  "
          f"binary_f1={bin_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
