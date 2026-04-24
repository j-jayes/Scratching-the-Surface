"""Pure-Oracle baseline: call AOAI directly on every test image, no cascade.

Used to compare cost + accuracy vs the cascade. Reuses the same shared
client / few-shot prompt as the production Oracle endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from cascade_defect.layer3_gpt4o.oracle import predict
from dotenv import load_dotenv

load_dotenv()

SEED_DIR = Path("data/splits/seed")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEST_ROOT = Path("data/splits/test")
OUT_PATH = Path("reports/eval_oracle_only.jsonl")


def iter_test_images(root: Path):
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.glob("*.jpg")):
            yield cls_dir.name, img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    images = list(iter_test_images(TEST_ROOT))
    if not args.full and args.limit:
        per_class = max(1, args.limit // 6)
        keep, seen = [], {}
        for cls, p in images:
            if seen.get(cls, 0) < per_class:
                keep.append((cls, p))
                seen[cls] = seen.get(cls, 0) + 1
        images = keep

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Pure-Oracle baseline on %d images", len(images))

    with out.open("w", encoding="utf-8") as fh:
        for i, (cls, path) in enumerate(images, 1):
            t0 = time.monotonic()
            try:
                pred, usage = predict(path, SEED_DIR)
                rec = {
                    "true_class": cls,
                    "image": str(path),
                    "decision": "defect" if pred.defect_class != "no_defect" else "no_defect",
                    "class": pred.defect_class,
                    "confidence": pred.confidence,
                    "reasoning": pred.reasoning,
                    "usage": usage,
                    "client_elapsed_ms": int((time.monotonic() - t0) * 1000),
                }
            except Exception as e:  # noqa: BLE001
                rec = {
                    "true_class": cls,
                    "image": str(path),
                    "decision": "error",
                    "error": str(e),
                    "client_elapsed_ms": int((time.monotonic() - t0) * 1000),
                }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            log.info("[%d/%d] %s -> %s (%dms)", i, len(images), cls, rec.get("class", "-"), rec["client_elapsed_ms"])

    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
