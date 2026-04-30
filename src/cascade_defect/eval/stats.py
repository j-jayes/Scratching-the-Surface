"""Bootstrap CIs + paired McNemar's test for cascade evaluation.

Tiny zero-dep helpers (``random.Random`` only — no scipy, no numpy needed) so
they can run inside the same uv environment as the rest of the eval pipeline.

The two questions these answer:

* ``bootstrap_ci``: "How tight is my F1 / precision / recall estimate?"
* ``mcnemar``: "Is the cascade *significantly* worse than the Oracle on the
  same frames, or just within sampling noise?"
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CI:
    point: float
    lo: float
    hi: float

    def fmt(self, ndigits: int = 3) -> str:
        return f"{self.point:.{ndigits}f} [{self.lo:.{ndigits}f}, {self.hi:.{ndigits}f}]"


def bootstrap_ci(
    records: Sequence[dict],
    statistic: Callable[[Sequence[dict]], float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CI:
    """Percentile bootstrap confidence interval for an arbitrary record statistic.

    ``records`` is a list of per-frame eval rows; ``statistic`` is any
    function that takes a sample (with replacement) of those rows and returns
    a scalar (e.g. F1). Returns the point estimate plus the (alpha/2,
    1-alpha/2) percentile bounds over ``n_resamples`` bootstrap samples.
    """
    if not records:
        return CI(0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(records)
    point = statistic(records)
    samples: list[float] = []
    for _ in range(n_resamples):
        resample = [records[rng.randrange(n)] for _ in range(n)]
        try:
            samples.append(statistic(resample))
        except Exception:  # noqa: BLE001 — degenerate resamples (e.g. all-negative) get skipped
            continue
    if not samples:
        return CI(point, point, point)
    samples.sort()
    lo_idx = max(0, int(len(samples) * (alpha / 2)))
    hi_idx = min(len(samples) - 1, int(len(samples) * (1 - alpha / 2)))
    return CI(point, samples[lo_idx], samples[hi_idx])


def mcnemar(b: int, c: int, *, continuity: bool = True) -> tuple[float, float]:
    """McNemar's exact-style test on a paired 2x2 disagreement table.

    ``b`` = #frames where system A is correct AND system B is wrong.
    ``c`` = #frames where system A is wrong AND system B is correct.

    Returns (chi-square statistic, two-sided p-value). Uses the continuity
    correction by default. For tiny ``b+c`` (< 25) the exact binomial p-value
    is returned instead.

    Implementation note: the binomial sf is computed with ``math.comb`` to
    keep this dependency-free.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < 25:
        # Exact two-sided binomial: P(X <= min(b,c)) under p=0.5, doubled.
        k = min(b, c)
        cum = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * cum)
        return float("nan"), p
    diff = abs(b - c) - (1 if continuity else 0)
    chi = (diff * diff) / n
    # chi-square (df=1) survival function = erfc(sqrt(chi)/sqrt(2))
    p = math.erfc(math.sqrt(chi) / math.sqrt(2))
    return chi, p


# ─── ready-made record statistics for cascade JSONL traces ───────────────────
def _binary(rec: dict) -> str:
    d = rec.get("decision", "error")
    if d in ("error", "uncertain"):
        return d
    return "defect" if d == "defect" else "no_defect"


def f1_stat(recs: Sequence[dict]) -> float:
    tp = fp = fn = 0
    for r in recs:
        truth = r.get("true_polarity") == "defective"
        pred = _binary(r) == "defect"
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif (not truth) and pred:
            fp += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def precision_stat(recs: Sequence[dict]) -> float:
    tp = fp = 0
    for r in recs:
        truth = r.get("true_polarity") == "defective"
        pred = _binary(r) == "defect"
        if truth and pred:
            tp += 1
        elif (not truth) and pred:
            fp += 1
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall_stat(recs: Sequence[dict]) -> float:
    tp = fn = 0
    for r in recs:
        truth = r.get("true_polarity") == "defective"
        pred = _binary(r) == "defect"
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
    return tp / (tp + fn) if (tp + fn) else 0.0


def cost_per_100k_stat(recs: Sequence[dict], price_in: float, price_out: float) -> float:
    in_tok = out_tok = 0
    for r in recs:
        for t in r.get("trace", []):
            if t.get("layer") == 3 and "usage" in t:
                in_tok += t["usage"].get("prompt_tokens", 0)
                out_tok += t["usage"].get("completion_tokens", 0)
    cost = in_tok * price_in + out_tok * price_out
    return cost / len(recs) * 100_000 if recs else 0.0
