"""Phase 2 — VLM few-shot benchmark on the locked 240-image v2 subset.

Runs each configured provider through ``data/splits_metal_v2/vlm_benchmark.csv``
and writes:

* ``reports/vlm_bench_metal_traces.jsonl``  — per-image trace (resumable)
* ``reports/vlm_bench_metal.json``          — aggregate metrics per provider

Cost guardrail: ``--max-cost-usd`` aborts a provider run when the rolling spend
crosses the cap. Resumable: if the trace file already contains an
``(image, provider, model)`` triple, it is skipped on re-run.

Usage::

    uv run python scripts/bench_vlm.py
    uv run python scripts/bench_vlm.py --providers azure --max-cost-usd 1.0
    uv run python scripts/bench_vlm.py --providers openrouter --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR_DEFAULT = ROOT / "data/splits_metal_v2/seed"
BENCH_CSV_DEFAULT = ROOT / "data/splits_metal_v2/vlm_benchmark.csv"
TRACES_DEFAULT = ROOT / "reports/vlm_bench_metal_traces.jsonl"
REPORT_DEFAULT = ROOT / "reports/vlm_bench_metal.json"

# Map raw model output → 5-class taxonomy used by the v2 split.
# NEU class names sometimes leak in if the model hallucinates from training
# data (e.g. "scratches"/"patches"); we collapse them to the metal canonical.
LABEL_NORMALIZE = {
    "scratches": "scratch",
    "patches": "patch",
    "pitted_surface": "pitting",
    "rolled-in_scale": "patch",      # closest visual analog
    "crazing": "scratch",            # crazing = fine cracks → linear class
    "inclusion": "inclusion",
    "scratch": "scratch",
    "patch": "patch",
    "pitting": "pitting",
    "no_defect": "no_defect",
    "uncertain": "uncertain",
    "surface_anomaly": "surface_anomaly",
}

# Final canonical classes for the comparison (must match resnet50.CLASSES).
CANONICAL = ["no_defect", "pitting", "inclusion", "scratch", "patch"]


# ── Provider configs ────────────────────────────────────────────────────────
def _build_clients(names: list[str]) -> list:
    from cascade_defect.vlm import AzureOpenAIClient, OpenRouterClient
    out = []
    if "azure" in names:
        out.append(AzureOpenAIClient())
    if "openrouter" in names:
        out.append(OpenRouterClient(
            model="qwen/qwen3-vl-30b-a3b-instruct",
            price_in=0.13, price_out=0.52,
        ))
    if "openrouter_qwen35plus" in names:
        out.append(OpenRouterClient(
            model="qwen/qwen3.5-plus-20260420",
            price_in=0.40, price_out=2.40,
        ))
    return out


# ── Trace I/O ───────────────────────────────────────────────────────────────
def load_existing(path: Path) -> set[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    if not path.exists():
        return seen
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            seen.add((r["image_path"], r["provider"], r["model"]))
    return seen


def append_trace(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ── Metrics ─────────────────────────────────────────────────────────────────
def normalize_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    return LABEL_NORMALIZE.get(raw.strip().lower(), raw.strip().lower())


def per_class_prf1(labels: list[str], preds: list[str]) -> dict:
    classes = sorted(set(labels) | {"no_defect"})
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(labels, preds) if t == c and p == c)
        fp = sum(1 for t, p in zip(labels, preds) if t != c and p == c)
        fn = sum(1 for t, p in zip(labels, preds) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        support = sum(1 for t in labels if t == c)
        out[c] = {"precision": round(precision, 4),
                  "recall": round(recall, 4),
                  "f1": round(f1, 4),
                  "support": support}
    return out


def aggregate(traces: list[dict]) -> dict:
    by_pm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in traces:
        by_pm[(t["provider"], t["model"])].append(t)

    out: dict = {"providers": []}
    for (provider, model), recs in sorted(by_pm.items()):
        labels: list[str] = []
        preds: list[str] = []
        latencies: list[float] = []
        costs: list[float] = []
        in_toks: list[int] = []
        out_toks: list[int] = []
        parse_fail = 0
        api_fail = 0
        # Collapse predictions outside the canonical taxonomy into special bins.
        unknown_predictions: Counter = Counter()
        for r in recs:
            if r.get("error") and not r.get("predicted_class"):
                api_fail += 1
                continue
            true_lbl = r["true_class"]
            pred = r.get("predicted_class")
            if pred is None:
                parse_fail += 1
                continue
            pred_norm = pred if pred in CANONICAL else "unknown"
            if pred_norm == "unknown":
                unknown_predictions[pred] += 1
            labels.append(true_lbl)
            preds.append(pred_norm)
            latencies.append(r["latency_s"])
            if r.get("cost_usd") is not None:
                costs.append(r["cost_usd"])
            if r.get("in_tokens") is not None:
                in_toks.append(r["in_tokens"])
            if r.get("out_tokens") is not None:
                out_toks.append(r["out_tokens"])

        scored = len(labels)
        n_correct = sum(1 for t, p in zip(labels, preds) if t == p)
        n_binary_tp = sum(1 for t, p in zip(labels, preds)
                          if t != "no_defect" and p != "no_defect")
        n_binary_fn = sum(1 for t, p in zip(labels, preds)
                          if t != "no_defect" and p == "no_defect")
        n_binary_fp = sum(1 for t, p in zip(labels, preds)
                          if t == "no_defect" and p != "no_defect")
        n_binary_tn = sum(1 for t, p in zip(labels, preds)
                          if t == "no_defect" and p == "no_defect")
        bin_p = (n_binary_tp / (n_binary_tp + n_binary_fp)
                 if (n_binary_tp + n_binary_fp) else 0.0)
        bin_r = (n_binary_tp / (n_binary_tp + n_binary_fn)
                 if (n_binary_tp + n_binary_fn) else 0.0)
        bin_f1 = (2 * bin_p * bin_r / (bin_p + bin_r)
                  if (bin_p + bin_r) else 0.0)

        prf1 = per_class_prf1(labels, preds)
        macro_f1 = (sum(v["f1"] for c, v in prf1.items() if c in CANONICAL)
                    / max(1, sum(1 for c in prf1 if c in CANONICAL)))

        latencies_sorted = sorted(latencies)
        def _pct(p: float) -> float | None:
            if not latencies_sorted:
                return None
            i = int(round(p * (len(latencies_sorted) - 1)))
            return latencies_sorted[i]

        out["providers"].append({
            "provider": provider,
            "model": model,
            "n_total": len(recs),
            "n_scored": scored,
            "api_failures": api_fail,
            "json_parse_failures": parse_fail,
            "accuracy": round(n_correct / scored, 4) if scored else None,
            "macro_f1_5class": round(macro_f1, 4) if scored else None,
            "binary_defect_vs_normal": {
                "precision": round(bin_p, 4), "recall": round(bin_r, 4),
                "f1": round(bin_f1, 4),
                "tp": n_binary_tp, "fp": n_binary_fp,
                "fn": n_binary_fn, "tn": n_binary_tn,
            },
            "per_class": prf1,
            "unknown_predictions": dict(unknown_predictions),
            "latency_s": {
                "n": len(latencies),
                "mean": round(sum(latencies)/len(latencies), 3) if latencies else None,
                "p50": round(_pct(0.5), 3) if latencies else None,
                "p95": round(_pct(0.95), 3) if latencies else None,
            },
            "cost_usd": {
                "total": round(sum(costs), 4) if costs else None,
                "mean_per_image": round(sum(costs)/len(costs), 6) if costs else None,
                "per_1k": round(sum(costs)/len(costs)*1000, 4) if costs else None,
            },
            "tokens": {
                "in_mean": round(sum(in_toks)/len(in_toks), 1) if in_toks else None,
                "out_mean": round(sum(out_toks)/len(out_toks), 1) if out_toks else None,
            },
        })
    return out


# ── Driver ──────────────────────────────────────────────────────────────────
def iter_bench_rows(csv_path: Path) -> Iterable[tuple[Path, str]]:
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            yield ROOT / r["image_path"], r["label"]


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--bench-csv", default=str(BENCH_CSV_DEFAULT))
    p.add_argument("--seed-dir", default=str(SEED_DIR_DEFAULT))
    p.add_argument("--traces", default=str(TRACES_DEFAULT))
    p.add_argument("--report", default=str(REPORT_DEFAULT))
    p.add_argument("--providers", nargs="+",
                   default=["azure", "openrouter"],
                   help="Subset of {azure, openrouter, openrouter_qwen35plus}.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap images per provider (handy for smoke tests).")
    p.add_argument("--max-cost-usd", type=float, default=5.0,
                   help="Per-provider hard cap; aborts that provider when crossed.")
    p.add_argument("--reset", action="store_true",
                   help="Wipe traces file before running.")
    args = p.parse_args()

    bench_csv = Path(args.bench_csv)
    seed_dir = Path(args.seed_dir)
    traces_path = Path(args.traces)
    report_path = Path(args.report)

    if args.reset and traces_path.exists():
        traces_path.unlink()
        print(f"Wiped {traces_path}")

    rows = list(iter_bench_rows(bench_csv))
    print(f"Bench: {len(rows)} images from {bench_csv.relative_to(ROOT)}")
    print(f"Seed dir: {seed_dir.relative_to(ROOT)}  "
          f"(classes: {sorted(p.name for p in seed_dir.iterdir() if p.is_dir())})")

    clients = _build_clients(args.providers)
    if not clients:
        raise SystemExit(f"No providers built from {args.providers}")
    print(f"Providers: {[c.provider + '/' + c.model for c in clients]}")

    seen = load_existing(traces_path)
    if seen:
        print(f"Resuming — {len(seen)} (image,provider,model) triples already done")

    for client in clients:
        running_cost = 0.0
        n_done = 0
        n_skipped = 0
        per_image = []
        t_start = time.time()
        print(f"\n=== {client.provider} / {client.model} ===")
        for i, (img_path, true_lbl) in enumerate(rows, 1):
            rel = str(img_path.relative_to(ROOT))
            if (rel, client.provider, client.model) in seen:
                n_skipped += 1
                continue
            if args.limit is not None and n_done >= args.limit:
                break
            r = client.predict(img_path, seed_dir, domain="metal")
            pred = (r.prediction.defect_class if r.prediction else None)
            pred_norm = normalize_label(pred)
            rec = {
                "image_path": rel,
                "true_class": true_lbl,
                "provider": client.provider,
                "model": client.model,
                "predicted_class_raw": pred,
                "predicted_class": pred_norm,
                "confidence": (r.prediction.confidence if r.prediction else None),
                "reasoning": (r.prediction.reasoning if r.prediction else None)[:200]
                              if r.prediction and r.prediction.reasoning else None,
                "latency_s": r.latency_s,
                "in_tokens": r.in_tokens,
                "out_tokens": r.out_tokens,
                "cost_usd": r.cost_usd,
                "error": r.error,
                "raw_text": r.raw_text[:200] if r.raw_text else None,
            }
            append_trace(traces_path, rec)
            per_image.append(rec)
            n_done += 1
            if r.cost_usd:
                running_cost += r.cost_usd
            mark = "OK " if pred_norm else "ERR"
            print(f"  [{mark}] {i:3d}/{len(rows)} t={r.latency_s:5.2f}s "
                  f"true={true_lbl:10s} pred={str(pred_norm):14s} "
                  f"$cum={running_cost:.4f}  err={r.error or '-'}")
            if running_cost >= args.max_cost_usd:
                print(f"  ↳ hit cost cap ${args.max_cost_usd}, stopping.")
                break
        dt = time.time() - t_start
        print(f"  done: scored={n_done} skipped={n_skipped} "
              f"wall={dt:.1f}s cost=${running_cost:.4f}")

    # Aggregate over the (now possibly enlarged) trace file.
    all_traces: list[dict] = []
    with traces_path.open() as fh:
        for line in fh:
            if line.strip():
                all_traces.append(json.loads(line))
    agg = aggregate(all_traces)
    agg["bench_csv"] = str(bench_csv.relative_to(ROOT))
    agg["seed_dir"] = str(seed_dir.relative_to(ROOT))
    agg["n_bench_images"] = len(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(agg, indent=2))
    try:
        rel_report = report_path.resolve().relative_to(ROOT)
    except ValueError:
        rel_report = report_path
    print(f"\nWrote {rel_report}")
    for prov in agg["providers"]:
        print(f"  {prov['provider']}/{prov['model']}: "
              f"acc={prov['accuracy']} macroF1={prov['macro_f1_5class']} "
              f"binF1={prov['binary_defect_vs_normal']['f1']} "
              f"p50={prov['latency_s']['p50']}s "
              f"$/1k={prov['cost_usd']['per_1k']}")


if __name__ == "__main__":
    main()
