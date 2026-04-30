"""CLI demo: push one image through the in-process L1 + L2 cascade.

Self-contained — no microservices, no Azure, no GPT call. Useful for the
README walkthrough and for sanity-checking a freshly-trained banks/weights
combo on a single test image.

Usage:
    uv run python scripts/demo_cascade.py path/to/image.jpg --domain ksdd2
    uv run python scripts/demo_cascade.py path/to/image.jpg --domain severstal --json

Output is a single JSON-serialisable trace identical in shape to the
per-record output of `cascade_defect.eval.run_cascade_metal`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cascade_defect.eval.run_cascade_metal import (
    DEFAULT_PATCHCORE_DIR,
    _L1PatchCore,
    _L2Yolo,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO = ROOT / "models" / "yolo_metal" / "best.pt"

# Phase K.2 calibrated knee thresholds (also persisted into
# models/patchcore_metal/summary.json under "calibration_knee").
CALIBRATED_TAU = {"severstal": -0.5, "ksdd2": 1.0}


def run_one(image: Path, domain: str, *, yolo_weights: Path | None) -> dict:
    if domain not in {"ksdd2", "severstal"}:
        raise SystemExit(f"--domain must be one of ksdd2|severstal, got {domain!r}")
    if not image.exists():
        raise SystemExit(f"Image not found: {image}")

    l1 = _L1PatchCore(DEFAULT_PATCHCORE_DIR, z_per_domain=CALIBRATED_TAU)
    l2 = _L2Yolo(yolo_weights) if (yolo_weights and yolo_weights.exists()) else None

    trace: list[dict] = []
    t0 = time.monotonic()

    t1 = time.monotonic()
    raw, z = l1.score(image, domain)
    tau = l1.threshold_for(domain)
    l1_decision = "defect" if z >= tau else "no_defect"
    trace.append({
        "layer": 1, "decision": l1_decision,
        "score_raw": round(raw, 6), "score_z": round(z, 3),
        "z_threshold": tau, "elapsed_ms": int((time.monotonic() - t1) * 1000),
    })

    if l1_decision == "no_defect":
        return {
            "image": str(image), "domain": domain,
            "decision": "no_defect", "stopped_at_layer": 1,
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "trace": trace,
        }

    if l2 is not None:
        t2 = time.monotonic()
        cls_name, conf, n_det = l2.detect(image)
        trace.append({
            "layer": 2, "decision": "defect" if cls_name else "no_defect",
            "class": cls_name, "confidence": round(conf, 3),
            "n_detections": n_det,
            "elapsed_ms": int((time.monotonic() - t2) * 1000),
        })
        if cls_name and conf >= 0.50:
            return {
                "image": str(image), "domain": domain,
                "decision": "defect", "class": cls_name,
                "confidence": round(conf, 3),
                "stopped_at_layer": 2,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
                "trace": trace,
            }

    return {
        "image": str(image), "domain": domain,
        "decision": "uncertain",
        "stopped_at_layer": 2 if l2 is not None else 1,
        "note": "would escalate to L3 (GPT-4.1-mini) in production; skipped in demo",
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path, help="Path to a single test image (jpg/png)")
    ap.add_argument("--domain", required=True, choices=["ksdd2", "severstal"])
    ap.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO)
    ap.add_argument("--json", action="store_true", help="Emit JSON only (no pretty table)")
    args = ap.parse_args(argv)

    result = run_one(args.image, args.domain, yolo_weights=args.yolo_weights)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"\nImage:  {result['image']}")
    print(f"Domain: {result['domain']}")
    print(f"\n→ Decision: {result['decision'].upper()}")
    if result.get("class"):
        print(f"  Class:    {result['class']} (conf {result.get('confidence', 0):.2f})")
    print(f"  Stopped:  Layer {result['stopped_at_layer']}")
    print(f"  Elapsed:  {result['elapsed_ms']} ms")
    if note := result.get("note"):
        print(f"  Note:     {note}")
    print("\nTrace:")
    for t in result["trace"]:
        bits = [f"L{t['layer']}", t["decision"]]
        if "score_z" in t and t["score_z"] is not None:
            bits.append(f"z={t['score_z']:+.2f} (τ={t['z_threshold']:+.2f})")
        if "confidence" in t and t["confidence"]:
            bits.append(f"yolo={t.get('class','?')}@{t['confidence']:.2f}")
        bits.append(f"{t.get('elapsed_ms', 0)}ms")
        print("  · " + "  ".join(bits))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
