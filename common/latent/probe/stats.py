"""Small statistics helpers shared by the probe modules (ours; the bootstrap mirrors latent.eval.summarize: seeded
percentile bootstrap over examples)."""
from __future__ import annotations

import math

import numpy as np


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of the mean over the first axis (examples). Returns (mean, lo, hi); NaNs for empty."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    a = (1.0 - level) / 2.0
    lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_stat_ci(n_items: int, stat, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of an arbitrary statistic stat(index_array) resampling items (examples)."""
    if n_items == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(stat(np.arange(n_items)))
    draws = np.array([stat(rng.integers(0, n_items, size=n_items)) for _ in range(n_boot)], dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    a = (1.0 - level) / 2.0
    lo, hi = np.percentile(draws, [100 * a, 100 * (1 - a)])
    return point, float(lo), float(hi)


def paired_diff_ci(a, b, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Paired example bootstrap of mean(a - b) (registration 7.9: "paired example-bootstrap interval")."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bootstrap_mean_ci(a - b, n_boot=n_boot, seed=seed, level=level)


def binomial_sigma(p: float, n: int) -> float:
    """sqrt(p (1 - p) / n): the binomial standard error over n examples (registration 7.9 "2 sigma")."""
    if n <= 0 or not math.isfinite(p):
        return float("nan")
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


def f1_score(y_true, y_pred) -> float:
    """Binary F1 (positive class 1); 0.0 when there are no positives in truth and prediction."""
    t, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = float((t & p).sum())
    denom = float(t.sum() + p.sum())
    return 0.0 if denom == 0 else 2.0 * tp / denom


__all__ = ["bootstrap_mean_ci", "bootstrap_stat_ci", "paired_diff_ci", "binomial_sigma", "f1_score"]
