"""Statistics for the pre-registered tests: bootstrap CIs, paired Wilcoxon, Holm correction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``stat(row_indices)``, resampling whole clusters.

    Votes on the same item are not independent, so items (not votes) are resampled.
    """
    codes, uniq = pd.factorize(np.asarray(clusters))
    rows_of = [np.flatnonzero(codes == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(codes)))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws[b] = stat(np.concatenate([rows_of[c] for c in pick]))
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided p-value of the Wilcoxon signed-rank test on paired settings ``a - b``."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(diff, 0):
        return 1.0
    return float(sps.wilcoxon(diff).pvalue)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm–Bonferroni step-down: which hypotheses are rejected at family-wise level ``alpha``."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    reject, still = {}, True
    for i, name in enumerate(order):
        still = still and pvalues[name] <= alpha / (m - i)
        reject[name] = still
    return reject


def minimum_detectable_effect(
    seed_sd: float, n_seeds: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest mean difference detectable given the seed-to-seed SD (normal approximation)."""
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return float(z * seed_sd * np.sqrt(2 / n_seeds))
