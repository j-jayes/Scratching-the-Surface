"""Evaluate the trained ResNet50 on the v2 test split.

Outputs:
- ``reports/resnet50_metal.json``  — per-class P/R/F1, macro-F1, accuracy,
  ECE, latency (CPU/MPS), per-image predictions count, confusion matrix.
- ``website/assets/eval/cm_resnet50.png``  — confusion-matrix heatmap.

Usage::

    uv run python -m cascade_defect.classical.eval \
        --weights models/resnet50_severstal.pt \
        --test-csv data/splits_metal_v2/test_labels.csv
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import EVAL_TFM, SeverstalCSV, make_eval_loader
from .resnet50 import CLASSES, load_checkpoint
from .train import pick_device

ROOT = Path(__file__).resolve().parents[3]


def per_class_prf1(cm: np.ndarray) -> dict:
    out = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        out[c] = {"precision": float(precision), "recall": float(recall),
                  "f1": float(f1), "support": int(cm[i, :].sum())}
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    confidences = probs.max(1)
    predictions = probs.argmax(1)
    accuracies = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def latency_benchmark(model, device, n: int = 50) -> dict:
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {"device": str(device), "n": n,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean())}


def render_confusion(cm: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # local import — optional dep
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("ResNet50 — confusion matrix (row-normalised)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            txt = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/resnet50_severstal.pt")
    p.add_argument("--test-csv",
                   default="data/splits_metal_v2/test_labels.csv")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-json", default="reports/resnet50_metal.json")
    p.add_argument("--out-cm", default="website/assets/eval/cm_resnet50.png")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Device: {device}")

    model = load_checkpoint(ROOT / args.weights, device=device)
    loader = make_eval_loader(ROOT / args.test_csv,
                              batch_size=args.batch_size, num_workers=2)

    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_paths: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.tolist())
            all_paths.extend(paths)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.array(all_labels)
    preds = probs.argmax(1)
    eval_secs = time.time() - t0

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for t, p_ in zip(labels, preds):
        cm[t, p_] += 1

    prf1 = per_class_prf1(cm)
    macro_f1 = float(np.mean([v["f1"] for v in prf1.values()]))
    accuracy = float((preds == labels).mean())
    ece = expected_calibration_error(probs, labels)

    # Binary defect/no-defect view (no_defect = 0)
    binary_pred = (preds != 0).astype(int)
    binary_true = (labels != 0).astype(int)
    tp = int(((binary_pred == 1) & (binary_true == 1)).sum())
    fp = int(((binary_pred == 1) & (binary_true == 0)).sum())
    fn = int(((binary_pred == 0) & (binary_true == 1)).sum())
    tn = int(((binary_pred == 0) & (binary_true == 0)).sum())
    bin_p = tp / (tp + fp) if (tp + fp) else 0.0
    bin_r = tp / (tp + fn) if (tp + fn) else 0.0
    bin_f1 = (2 * bin_p * bin_r / (bin_p + bin_r)) if (bin_p + bin_r) else 0.0

    lat_eval_device = latency_benchmark(model, device)
    cpu_lat = (latency_benchmark(model.cpu(), torch.device("cpu"), n=20)
               if device.type != "cpu" else None)

    report = {
        "weights": args.weights,
        "test_csv": args.test_csv,
        "n_images": int(len(labels)),
        "label_distribution": dict(Counter(CLASSES[i] for i in labels.tolist())),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": prf1,
        "binary_defect_vs_normal": {
            "precision": float(bin_p), "recall": float(bin_r),
            "f1": float(bin_f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "ece": ece,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "latency": {"eval_device": lat_eval_device, "cpu": cpu_lat},
        "eval_wall_secs": eval_secs,
        "per_image": [
            {
                "image": all_paths[i],
                "true_label": CLASSES[int(labels[i])],
                "pred_label": CLASSES[int(preds[i])],
                "confidence": float(probs[i].max()),
                "probs": {CLASSES[k]: float(probs[i][k]) for k in range(len(CLASSES))},
            }
            for i in range(len(labels))
        ],
    }
    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")

    out_cm = ROOT / args.out_cm
    render_confusion(cm, out_cm)
    print(f"Wrote {out_cm}")
    print(f"\nmacro_F1={macro_f1:.4f}  acc={accuracy:.4f}  "
          f"binary_F1={bin_f1:.4f}  ECE={ece:.4f}")


if __name__ == "__main__":
    # Avoid unused-import warning when matplotlib not available; SeverstalCSV/EVAL_TFM kept for re-exports.
    _ = (SeverstalCSV, EVAL_TFM)
    main()
