"""Offline batch annotator — call the Oracle on the unlabelled pool to produce pseudo-labels.

Usage
-----
    # Smoke test (10 images, ~$0.01)
    uv run python -m cascade_defect.layer3_gpt4o.annotate --limit 10

    # Full run on the 1,422-image unlabelled pool (~$1)
    uv run python -m cascade_defect.layer3_gpt4o.annotate --full

Output is a JSONL file at ``data/processed/pseudo_labels.jsonl`` and is uploaded
to Blob (``processed/pseudo_labels.jsonl``) when ``--upload`` is passed.

Resumable: rows already present in the JSONL (matched by image path) are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from cascade_defect.layer3_gpt4o.oracle import predict

logger = logging.getLogger(__name__)

DEFAULT_UNLABELLED_DIR = Path("data/splits/unlabelled")
DEFAULT_SEED_DIR = Path("data/splits/seed")
DEFAULT_OUTPUT = Path("data/processed/pseudo_labels.jsonl")

# gpt-4.1-mini pricing (USD per 1M tokens, GlobalStandard / Standard, Apr 2026)
PRICE_INPUT_PER_1M = 0.40
PRICE_OUTPUT_PER_1M = 1.60


def _load_existing(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            seen.add(json.loads(line)["image_path"])
        except (json.JSONDecodeError, KeyError):
            continue
    return seen


def _iter_images(unlabelled_dir: Path):
    for class_dir in sorted(p for p in unlabelled_dir.iterdir() if p.is_dir()):
        for img in sorted(class_dir.glob("*.jpg")):
            yield img, class_dir.name  # truth label kept only for evaluation


def annotate(
    unlabelled_dir: Path = DEFAULT_UNLABELLED_DIR,
    seed_dir: Path = DEFAULT_SEED_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
) -> dict:
    """Annotate the unlabelled pool, appending one JSONL row per image.

    Returns a summary dict (counts, total cost USD).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _load_existing(output_path)
    logger.info("Resuming with %d previously-annotated images", len(seen))

    total_in = total_out = 0
    n_done = 0
    n_skipped = 0
    n_correct = 0  # only meaningful because we know the truth labels

    t0 = time.monotonic()
    with output_path.open("a", encoding="utf-8") as f:
        for img_path, true_class in _iter_images(unlabelled_dir):
            rel = img_path.as_posix()
            if rel in seen:
                n_skipped += 1
                continue
            if limit is not None and n_done >= limit:
                break
            try:
                pred, usage = predict(img_path, seed_dir)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed on %s: %s", rel, e)
                continue

            row = {
                "image_path": rel,
                "true_class": true_class,
                "pred_class": pred.defect_class,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            total_in += usage["prompt_tokens"]
            total_out += usage["completion_tokens"]
            n_done += 1
            if pred.defect_class == true_class:
                n_correct += 1
            if n_done % 5 == 0 or n_done <= 3:
                logger.info(
                    "[%d] %s → %s (conf=%.2f) tokens=%d/%d",
                    n_done, img_path.name, pred.defect_class, pred.confidence,
                    usage["prompt_tokens"], usage["completion_tokens"],
                )

    elapsed = time.monotonic() - t0
    cost_usd = (total_in / 1_000_000 * PRICE_INPUT_PER_1M) + (
        total_out / 1_000_000 * PRICE_OUTPUT_PER_1M
    )
    summary = {
        "images_annotated": n_done,
        "images_skipped_resume": n_skipped,
        "prompt_tokens": total_in,
        "completion_tokens": total_out,
        "elapsed_s": round(elapsed, 1),
        "estimated_cost_usd": round(cost_usd, 4),
        "accuracy_vs_truth": round(n_correct / n_done, 3) if n_done else None,
    }
    logger.info("Summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Batch-annotate the unlabelled pool.")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--limit", type=int, default=10, help="Annotate at most N images (default 10).")
    g.add_argument("--full", action="store_true", help="Annotate the entire unlabelled pool.")
    parser.add_argument("--unlabelled-dir", type=Path, default=DEFAULT_UNLABELLED_DIR)
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    limit = None if args.full else args.limit
    summary = annotate(args.unlabelled_dir, args.seed_dir, args.output, limit=limit)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
