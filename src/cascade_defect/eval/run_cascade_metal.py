"""Local cascade evaluator for the metal-surface refit (Phase J / J.1 / J.2).

Runs the full L1 (PatchCore) → L2 (YOLO, optional) → L3 (Oracle, optional)
cascade *in-process* against ``data/splits_metal/cascade_test/`` and writes a
JSONL trace + a summary block per evaluation track:

    Track A — Severstal in-domain   (positives + negatives)
    Track B — KSDD2 in-domain       (AE/PatchCore + Oracle, no YOLO involvement)
    Track C — KSDD2 defectives via Severstal-trained YOLO (OOD generalisation)

Why local? The deployed cascade is identical (same code paths) but each
endpoint costs an ACA cold-start + AOAI tokens. Running in-process is
~100× cheaper and gives us deterministic timings on the dev box. The
production smoke-test against the live router stays as ``run_cascade.py``.

Usage::

    # No-API smoke test (L1 only, both domains, fast):
    uv run python -m cascade_defect.eval.run_cascade_metal --layers l1 \\
        --limit-per-track 50

    # Full Track A with YOLO + Oracle (costs AOAI tokens):
    uv run python -m cascade_defect.eval.run_cascade_metal \\
        --tracks A --layers l1 l2 l3 --limit-per-track 200

    # Everything, capped at 100/track to keep the bill bounded:
    uv run python -m cascade_defect.eval.run_cascade_metal \\
        --layers l1 l2 l3 --limit-per-track 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CASCADE_TEST_ROOT = Path("data/splits_metal/cascade_test")
SEED_DIR = Path("data/splits_metal/seed")  # falls back to data/splits/seed if absent
LEGACY_SEED_DIR = Path("data/splits/seed")
DEFAULT_PATCHCORE_DIR = Path("models/patchcore_metal")
DEFAULT_YOLO_WEIGHTS = Path("models/yolo_metal/best.pt")
DEFAULT_OUT = Path("reports/eval_cascade_metal.jsonl")

Domain = Literal["severstal", "ksdd2"]
Polarity = Literal["normal", "defective"]
Track = Literal["A", "B", "C"]


# ─────────────────────────────────────────────────────────────────────────────
# Test-set iteration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TestCase:
    track: Track
    domain: Domain
    polarity: Polarity
    image: Path


def iter_track_a(root: Path) -> Iterator[TestCase]:
    """Severstal in-domain — both polarities."""
    for pol in ("normal", "defective"):
        for img in sorted((root / "severstal" / pol).glob("*.jpg")):
            yield TestCase("A", "severstal", pol, img)


def iter_track_b(root: Path) -> Iterator[TestCase]:
    """KSDD2 in-domain — both polarities, AE+Oracle path."""
    for pol in ("normal", "defective"):
        for img in sorted((root / "ksdd2" / pol).glob("*.png")):
            yield TestCase("B", "ksdd2", pol, img)


def iter_track_c(root: Path) -> Iterator[TestCase]:
    """KSDD2 defectives only — stress-test Severstal YOLO on OOD defects."""
    for img in sorted((root / "ksdd2" / "defective").glob("*.png")):
        yield TestCase("C", "ksdd2", "defective", img)


def _stratified_cap(cases: list[TestCase], cap: int, seed: int) -> list[TestCase]:
    """Cap total cases at ``cap`` while keeping per-(track, polarity) balance."""
    if cap <= 0 or len(cases) <= cap:
        return cases
    by_bucket: dict[tuple[Track, Polarity], list[TestCase]] = {}
    for c in cases:
        by_bucket.setdefault((c.track, c.polarity), []).append(c)
    rng = random.Random(seed)
    per_bucket = max(1, cap // len(by_bucket))
    out: list[TestCase] = []
    for bucket in by_bucket.values():
        rng.shuffle(bucket)
        out.extend(bucket[:per_bucket])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — PatchCore (in-process)
# ─────────────────────────────────────────────────────────────────────────────
class _L1PatchCore:
    """Lazy-loaded per-domain PatchCore banks + z-score scoring.

    ``z_threshold`` is the global default. ``z_per_domain`` (Phase K.2) lets
    us override τ per domain — the PatchCore Severstal/KSDD2 score scales are
    very different, so a single τ would either over- or under-trigger one
    side. When a domain is missing from the override map the global default
    is used.
    """

    def __init__(
        self,
        bank_dir: Path,
        z_threshold: float = 3.0,
        z_per_domain: dict[str, float] | None = None,
    ) -> None:
        from cascade_defect.layer1_autoencoder import patchcore  # lazy

        self._patchcore = patchcore
        self._bank_dir = bank_dir
        self._z = z_threshold
        self._z_per_domain = z_per_domain or {}
        self._extractor = patchcore._FeatureExtractor()
        self._banks: dict[str, tuple] = {}  # domain -> (bank_tensor, calibration)
        for d in ("severstal", "ksdd2"):
            try:
                self._banks[d] = patchcore.load_bank(bank_dir, d)
            except FileNotFoundError:
                logger.warning("No PatchCore bank for domain=%s in %s", d, bank_dir)

    def threshold_for(self, domain: str) -> float:
        return self._z_per_domain.get(domain, self._z)

    @property
    def z_threshold(self) -> float:
        return self._z

    def score(self, image_path: Path, domain: Domain) -> tuple[float, float]:
        """Return (raw_score, z_score). Raises if domain bank not loaded."""
        from PIL import Image

        if domain not in self._banks:
            raise RuntimeError(f"No PatchCore bank loaded for domain={domain}")
        bank, calib = self._banks[domain]
        img = Image.open(image_path)
        raw = self._patchcore.score_image(self._extractor, bank, img, device="cpu")
        std = calib.score_std if calib.score_std > 1e-9 else 1.0
        z = (raw - calib.score_mean) / std
        return raw, z


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Ultralytics YOLO (in-process, optional)
# ─────────────────────────────────────────────────────────────────────────────
class _L2Yolo:
    """Lazy-loaded YOLO; returns top detection per image."""

    def __init__(self, weights: Path, conf: float = 0.25) -> None:
        from ultralytics import YOLO  # lazy — heavy import

        self._model = YOLO(str(weights))
        self._conf = conf
        self.class_names: list[str] = list(self._model.names.values())  # type: ignore[arg-type]

    def detect(self, image_path: Path) -> tuple[str | None, float, int]:
        """Return (top_class_name, top_confidence, n_detections)."""
        results = self._model.predict(
            source=str(image_path), conf=self._conf, verbose=False
        )
        if not results:
            return None, 0.0, 0
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0, 0
        confs = boxes.conf.tolist()
        cls_ids = [int(c) for c in boxes.cls.tolist()]
        top_idx = int(max(range(len(confs)), key=lambda i: confs[i]))
        return self.class_names[cls_ids[top_idx]], float(confs[top_idx]), len(boxes)


# ─────────────────────────────────────────────────────────────────────────────
# Single-case evaluator
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_seed_dir() -> Path:
    return SEED_DIR if SEED_DIR.exists() else LEGACY_SEED_DIR


def _evaluate_one(
    case: TestCase,
    *,
    l1: _L1PatchCore | None,
    l2: _L2Yolo | None,
    l3_enabled: bool,
    seed_dir: Path,
    use_cache: bool = False,
) -> dict:
    record: dict = {
        "track": case.track,
        "domain": case.domain,
        "true_polarity": case.polarity,
        "image": str(case.image),
        "trace": [],
    }
    t_start = time.monotonic()

    # Layer 1 — PatchCore gate
    if l1 is None:
        # No L1 — assume all frames pass through to L2/L3
        l1_decision = "defect"
        record["trace"].append({"layer": 1, "decision": l1_decision, "score_z": None})
    else:
        try:
            t1 = time.monotonic()
            raw, z = l1.score(case.image, case.domain)
            l1_ms = int((time.monotonic() - t1) * 1000)
            tau = l1.threshold_for(case.domain)
            l1_decision = "defect" if z >= tau else "no_defect"
            record["trace"].append({
                "layer": 1, "decision": l1_decision,
                "score_raw": round(raw, 6), "score_z": round(z, 3),
                "z_threshold": tau, "elapsed_ms": l1_ms,
            })
        except Exception as e:  # noqa: BLE001
            record["trace"].append({"layer": 1, "decision": "error", "error": str(e)})
            record["decision"] = "error"
            record["stopped_at_layer"] = 1
            record["client_elapsed_ms"] = int((time.monotonic() - t_start) * 1000)
            return record

    if l1_decision == "no_defect":
        record["decision"] = "no_defect"
        record["class"] = "no_defect"
        record["stopped_at_layer"] = 1
        record["client_elapsed_ms"] = int((time.monotonic() - t_start) * 1000)
        return record

    # Layer 2 — YOLO (optional, skip on Track B by design)
    l2_ran = False
    if l2 is not None and case.track != "B":
        t2 = time.monotonic()
        cls_name, conf, n_det = l2.detect(case.image)
        l2_ms = int((time.monotonic() - t2) * 1000)
        l2_ran = True
        record["trace"].append({
            "layer": 2, "decision": "defect" if cls_name else "no_defect",
            "class": cls_name, "confidence": round(conf, 3),
            "n_detections": n_det, "elapsed_ms": l2_ms,
        })
        if cls_name and conf >= 0.50:
            # YOLO confident — short-circuit, no Oracle needed
            record["decision"] = "defect"
            record["class"] = cls_name
            record["confidence"] = conf
            record["stopped_at_layer"] = 2
            record["client_elapsed_ms"] = int((time.monotonic() - t_start) * 1000)
            return record

    # Layer 3 — Oracle (optional, costs AOAI tokens)
    if l3_enabled:
        if use_cache:
            from cascade_defect.layer3_gpt4o.cache import cached_predict as _predict_fn
        else:
            from cascade_defect.layer3_gpt4o.oracle import predict  # lazy

            def _predict_fn(image_path, seed_dir, *, domain=None):
                p, u = predict(image_path, seed_dir, domain=domain)
                return p, u, {"hit": False}

        try:
            t3 = time.monotonic()
            pred, usage, cache_info = _predict_fn(case.image, seed_dir, domain="metal")
            l3_ms = int((time.monotonic() - t3) * 1000)
            record["trace"].append({
                "layer": 3, "decision": pred.defect_class,
                "class": pred.defect_class, "confidence": pred.confidence,
                "reasoning": pred.reasoning, "usage": usage,
                "cache": cache_info, "elapsed_ms": l3_ms,
            })
            if pred.defect_class == "no_defect":
                record["decision"] = "no_defect"
                record["class"] = "no_defect"
            elif pred.defect_class == "uncertain":
                record["decision"] = "uncertain"
                record["class"] = "uncertain"
            else:
                record["decision"] = "defect"
                record["class"] = pred.defect_class
                record["confidence"] = pred.confidence
            record["stopped_at_layer"] = 3
        except Exception as e:  # noqa: BLE001
            record["trace"].append({"layer": 3, "decision": "error", "error": str(e)})
            record["decision"] = "error"
            record["stopped_at_layer"] = 3
    else:
        # No L3 configured — escalation still counts as "defect" for L1 gate stats
        record["decision"] = "defect"
        record["class"] = "defect_unclassified"
        record["stopped_at_layer"] = 2 if l2_ran else 1

    record["client_elapsed_ms"] = int((time.monotonic() - t_start) * 1000)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    ap.add_argument("--layers", nargs="+", choices=["l1", "l2", "l3"], default=["l1"])
    ap.add_argument("--patchcore-dir", type=Path, default=DEFAULT_PATCHCORE_DIR)
    ap.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    ap.add_argument("--z-threshold", type=float, default=float(os.getenv("Z_THRESHOLD", "3.0")))
    ap.add_argument("--z-severstal", type=float, default=None,
                    help="Per-domain override for Severstal (Phase K.2 calibrated knee).")
    ap.add_argument("--z-ksdd2", type=float, default=None,
                    help="Per-domain override for KSDD2 (Phase K.2 calibrated knee).")
    ap.add_argument("--use-cache", action="store_true",
                    help="Wrap L3 in the perceptual-hash cache (Phase K.4).")
    ap.add_argument("--limit-per-track", type=int, default=0,
                    help="Cap cases per track (stratified by polarity). 0 = no cap.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    layers = set(args.layers)

    # Build per-track case lists with independent caps so Track C (small) isn't
    # drowned out by Track A (large).
    track_cases: dict[Track, list[TestCase]] = {}
    if "A" in args.tracks:
        track_cases["A"] = list(iter_track_a(CASCADE_TEST_ROOT))
    if "B" in args.tracks:
        track_cases["B"] = list(iter_track_b(CASCADE_TEST_ROOT))
    if "C" in args.tracks:
        track_cases["C"] = list(iter_track_c(CASCADE_TEST_ROOT))

    if args.limit_per_track > 0:
        for t, cases in track_cases.items():
            track_cases[t] = _stratified_cap(cases, args.limit_per_track, args.seed)

    all_cases = [c for cs in track_cases.values() for c in cs]
    logger.info(
        "Evaluating %d cases (%s) across layers %s",
        len(all_cases),
        ", ".join(f"{t}={len(cs)}" for t, cs in track_cases.items()),
        sorted(layers),
    )

    # Lazy-init layers
    z_per_domain = {}
    if args.z_severstal is not None:
        z_per_domain["severstal"] = args.z_severstal
    if args.z_ksdd2 is not None:
        z_per_domain["ksdd2"] = args.z_ksdd2
    l1 = (
        _L1PatchCore(args.patchcore_dir, z_threshold=args.z_threshold,
                     z_per_domain=z_per_domain or None)
        if "l1" in layers else None
    )
    l2 = None
    if "l2" in layers:
        if not args.yolo_weights.exists():
            logger.warning("YOLO weights %s not found — skipping L2", args.yolo_weights)
        else:
            l2 = _L2Yolo(args.yolo_weights)
    l3_enabled = "l3" in layers
    seed_dir = _resolve_seed_dir()
    use_cache = args.use_cache and l3_enabled

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for case in all_cases:
            rec = _evaluate_one(
                case, l1=l1, l2=l2, l3_enabled=l3_enabled,
                seed_dir=seed_dir, use_cache=use_cache,
            )
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_done += 1
            logger.info(
                "[%d/%d] track=%s domain=%s true=%s -> L%s %s/%s (%dms)",
                n_done, len(all_cases), case.track, case.domain, case.polarity,
                rec.get("stopped_at_layer", "?"), rec.get("decision", "?"),
                rec.get("class", "-"), rec.get("client_elapsed_ms", -1),
            )

    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
