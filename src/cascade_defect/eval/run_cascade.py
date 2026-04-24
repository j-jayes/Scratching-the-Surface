"""End-to-end evaluation harness for the cascade.

Hits the deployed router for each image in the test split, captures the
per-layer trace + timing + token usage, and writes a JSONL log to
`reports/eval_cascade.jsonl`.

Usage:
    uv run python -m cascade_defect.eval.run_cascade --limit 60
    uv run python -m cascade_defect.eval.run_cascade --full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_ROUTER = (
    "https://cascade-router.orangebush-bb39ddbf.westeurope.azurecontainerapps.io"
)
TEST_ROOT = Path("data/splits/test")
OUT_PATH = Path("reports/eval_cascade.jsonl")


def iter_test_images(root: Path):
    """Yield (true_class, path) tuples for every image in the test split."""
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.glob("*.jpg")):
            yield cls_dir.name, img


def call_router(client: httpx.Client, router: str, img: Path) -> dict:
    """POST one image to the cascade router; return parsed JSON or error dict."""
    t0 = time.monotonic()
    try:
        with img.open("rb") as fh:
            r = client.post(
                f"{router}/predict",
                files={"file": (img.name, fh, "image/jpeg")},
                timeout=120,
            )
        r.raise_for_status()
        body = r.json()
        body["client_elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        return body
    except (httpx.HTTPError, ValueError) as e:
        return {
            "decision": "error",
            "error": str(e),
            "client_elapsed_ms": int((time.monotonic() - t0) * 1000),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max images (0 = all when --full).")
    ap.add_argument("--full", action="store_true", help="Run on the entire test split.")
    ap.add_argument("--router", default=os.getenv("ROUTER_URL", DEFAULT_ROUTER))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    images = list(iter_test_images(TEST_ROOT))
    if not args.full and args.limit:
        # Stratified: take first N/6 from each class.
        per_class = max(1, args.limit // 6)
        keep: list = []
        seen: dict[str, int] = {}
        for cls, p in images:
            if seen.get(cls, 0) < per_class:
                keep.append((cls, p))
                seen[cls] = seen.get(cls, 0) + 1
        images = keep

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Evaluating %d images against %s", len(images), args.router)

    n_done = 0
    with httpx.Client() as client, out.open("w", encoding="utf-8") as fh:
        for cls, path in images:
            body = call_router(client, args.router, path)
            record = {"true_class": cls, "image": str(path), **body}
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            n_done += 1
            stopped = body.get("stopped_at_layer", "?")
            decision = body.get("decision", "?")
            pred = body.get("class", "-")
            log.info(
                "[%d/%d] %s/%s -> L%s %s/%s (%dms)",
                n_done, len(images), cls, path.name,
                stopped, decision, pred, body.get("client_elapsed_ms", -1),
            )
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
