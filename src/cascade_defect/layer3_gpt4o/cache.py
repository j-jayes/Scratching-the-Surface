"""Perceptual-hash cache in front of the Oracle (Phase K.4).

Severstal coil frames within a single roll are near-duplicates: the steel
strip is moving past a fixed camera, neighbouring frames overlap heavily.
On the cascade test set ~30 % of escalations to L3 are visually
indistinguishable from a frame the Oracle has already classified.

This module wraps :func:`cascade_defect.layer3_gpt4o.oracle.predict` with a
**dHash** keyed cache (8×8 difference hash → 64-bit fingerprint, Hamming
distance ≤ ``DHASH_HAMMING_THRESHOLD``). dHash is the cheapest perceptual
hash that still survives mild brightness / framing jitter; pHash would be
slightly more robust but pulls in extra deps.

The cache is in-process (Python dict) — sufficient for the eval harness and
for an ACA app whose lifetime is a single revision. For multi-revision
persistence wire it through Redis behind the same interface.

Usage::

    from cascade_defect.layer3_gpt4o.cache import cached_predict
    pred, usage, cache_info = cached_predict(image_path, seed_dir, domain="metal")
    # cache_info["hit"] is True iff the Oracle was skipped
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .oracle import DefectPrediction, predict

logger = logging.getLogger(__name__)

DHASH_SIZE = 8  # 8×8 ⇒ 64-bit fingerprint
DHASH_HAMMING_THRESHOLD = 6  # ≤6/64 bits differ → treat as duplicate (~9 %)


def dhash(image_path: Path | str, size: int = DHASH_SIZE) -> int:
    """Compute the 64-bit difference hash of an image.

    Returns an int so we can XOR + ``bit_count()`` for Hamming distance.
    """
    img = Image.open(image_path).convert("L").resize(
        (size + 1, size), Image.BILINEAR
    )
    px = img.load()
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | (1 if px[x, y] > px[x + 1, y] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(slots=True)
class _CacheEntry:
    fingerprint: int
    prediction: DefectPrediction
    usage: dict
    source_image: str


@dataclass(slots=True)
class OracleCache:
    """Per-domain dHash cache. Hits skip the AOAI round-trip entirely."""

    threshold: int = DHASH_HAMMING_THRESHOLD
    entries: dict[str, list[_CacheEntry]] = field(default_factory=dict)
    n_hits: int = 0
    n_misses: int = 0
    saved_in_tokens: int = 0
    saved_out_tokens: int = 0

    def lookup(self, fingerprint: int, domain: str) -> _CacheEntry | None:
        bucket = self.entries.get(domain or "_default", [])
        for ent in bucket:
            if hamming(ent.fingerprint, fingerprint) <= self.threshold:
                return ent
        return None

    def insert(
        self,
        fingerprint: int,
        domain: str,
        prediction: DefectPrediction,
        usage: dict,
        source_image: str,
    ) -> None:
        self.entries.setdefault(domain or "_default", []).append(
            _CacheEntry(fingerprint, prediction, usage, source_image)
        )

    def predict(
        self,
        image_path: Path,
        seed_dir: Path,
        *,
        domain: str | None = None,
    ) -> tuple[DefectPrediction, dict, dict]:
        """Return (prediction, usage, cache_info)."""
        fp = dhash(image_path)
        d = (domain or "_default").lower()
        hit = self.lookup(fp, d)
        if hit is not None:
            self.n_hits += 1
            self.saved_in_tokens += hit.usage.get("prompt_tokens", 0)
            self.saved_out_tokens += hit.usage.get("completion_tokens", 0)
            logger.debug("Oracle cache HIT (domain=%s, fp=%016x)", d, fp)
            # Return a shallow-copied usage with zeros — no real tokens spent
            zero_usage = {k: 0 for k in hit.usage}
            return hit.prediction, zero_usage, {
                "hit": True,
                "fingerprint": f"{fp:016x}",
                "matched_source": hit.source_image,
            }
        self.n_misses += 1
        prediction, usage = predict(image_path, seed_dir, domain=domain)
        self.insert(fp, d, prediction, usage, str(image_path))
        return prediction, usage, {"hit": False, "fingerprint": f"{fp:016x}"}

    def stats(self) -> dict:
        total = self.n_hits + self.n_misses
        return {
            "n_hits": self.n_hits,
            "n_misses": self.n_misses,
            "hit_rate": round(self.n_hits / total, 4) if total else 0.0,
            "saved_prompt_tokens": self.saved_in_tokens,
            "saved_completion_tokens": self.saved_out_tokens,
            "cache_size": sum(len(v) for v in self.entries.values()),
            "threshold_hamming_bits": self.threshold,
        }


# Module-level singleton convenience for callers that don't want to thread state.
_default_cache = OracleCache()


def cached_predict(
    image_path: Path,
    seed_dir: Path,
    *,
    domain: str | None = None,
) -> tuple[DefectPrediction, dict, dict]:
    return _default_cache.predict(image_path, seed_dir, domain=domain)


def cache_stats() -> dict:
    return _default_cache.stats()


def reset_cache() -> None:
    global _default_cache
    _default_cache = OracleCache()
