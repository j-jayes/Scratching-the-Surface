"""Export L1+L2 backbones to ONNX and benchmark CPU inference latency.

Headline number for the architecture page: "What does a single-frame inference
cost on a CPU-only edge box?" Many edge deployments don't have CUDA — proving
the gate runs in <X ms on commodity hardware is a credible production signal.

Outputs:
    models/onnx/patchcore_resnet18.onnx
    models/onnx/yolov8n.onnx                    (delegated to ultralytics)
    reports/onnx_latency.json
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

import torch

from cascade_defect.layer1_autoencoder.patchcore import _FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("export_onnx")

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "models" / "onnx"
REPORT = ROOT / "reports" / "onnx_latency.json"
YOLO_PT = ROOT / "models" / "yolo_metal" / "best.pt"

WARMUP = 5
ITERS = 30


def export_patchcore() -> Path:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    out = ONNX_DIR / "patchcore_resnet18.onnx"
    extractor = _FeatureExtractor().eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        extractor,
        dummy,
        out.as_posix(),
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Exported PatchCore backbone → %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def export_yolo() -> Path | None:
    if not YOLO_PT.exists():
        logger.warning("YOLO weights missing at %s — skipping export", YOLO_PT)
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics not installed — skipping YOLO export")
        return None
    model = YOLO(YOLO_PT.as_posix())
    exported = model.export(format="onnx", imgsz=640, opset=17, dynamic=False, simplify=False)
    src = Path(exported)
    dst = ONNX_DIR / "yolov8n.onnx"
    if src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())
    logger.info("Exported YOLOv8n → %s (%.1f MB)", dst, dst.stat().st_size / 1e6)
    return dst


def benchmark(onnx_path: Path, input_shape: tuple[int, ...], input_name: str) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping benchmark for %s", onnx_path.name)
        return {"skipped": "onnxruntime not installed"}

    import numpy as np

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        onnx_path.as_posix(), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(WARMUP):
        sess.run(None, {input_name: dummy})

    times: list[float] = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "iters": ITERS,
        "threads": 4,
        "ms_p50": round(statistics.median(times), 2),
        "ms_p95": round(sorted(times)[int(0.95 * len(times)) - 1], 2),
        "ms_mean": round(statistics.mean(times), 2),
        "ms_min": round(min(times), 2),
    }


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"models": {}}

    pc = export_patchcore()
    pc_input_name = "image"
    report["models"]["patchcore_resnet18"] = {
        "onnx_path": pc.relative_to(ROOT).as_posix(),
        "size_mb": round(pc.stat().st_size / 1e6, 2),
        "input_shape": [1, 3, 224, 224],
        "latency_cpu": benchmark(pc, (1, 3, 224, 224), pc_input_name),
    }

    yolo = export_yolo()
    if yolo is not None:
        # ultralytics ONNX uses input name "images"
        report["models"]["yolov8n"] = {
            "onnx_path": yolo.relative_to(ROOT).as_posix(),
            "size_mb": round(yolo.stat().st_size / 1e6, 2),
            "input_shape": [1, 3, 640, 640],
            "latency_cpu": benchmark(yolo, (1, 3, 640, 640), "images"),
        }

    pc_p50 = report["models"]["patchcore_resnet18"]["latency_cpu"].get("ms_p50")
    yolo_p50 = report["models"].get("yolov8n", {}).get("latency_cpu", {}).get("ms_p50")
    if pc_p50 is not None:
        report["headline"] = {
            "patchcore_p50_ms": pc_p50,
            "yolo_p50_ms": yolo_p50,
            "l1_l2_combined_p50_ms": (
                round(pc_p50 + yolo_p50, 2) if yolo_p50 is not None else None
            ),
            "hardware": "4-thread CPU (onnxruntime)",
        }

    REPORT.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", REPORT)
    print(json.dumps(report["headline"], indent=2) if "headline" in report else "")


if __name__ == "__main__":
    main()
