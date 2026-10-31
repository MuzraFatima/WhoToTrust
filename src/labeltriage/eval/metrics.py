"""Evaluation metrics shared by every method.

All functions take NumPy arrays: ``probs`` is ``(n, C)`` predicted probabilities, ``labels`` is
``(n,)`` integer classes and ``human`` is ``(n, C)`` human vote distributions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins (15 by default)."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def _normalise(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    return p / np.maximum(p.sum(axis=1, keepdims=True), EPS)


def kl(human: np.ndarray, probs: np.ndarray) -> float:
    """Mean KL(human || model), in nats."""
    p, q = _normalise(human), _normalise(probs)
    terms = np.where(p > 0, p * (np.log(np.maximum(p, EPS)) - np.log(np.maximum(q, EPS))), 0.0)
    return float(terms.sum(axis=1).mean())


def js(human: np.ndarray, probs: np.ndarray) -> float:
    """Mean Jensen–Shannon divergence, in nats (bounded by ln 2)."""
    p, q = _normalise(human), _normalise(probs)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def manhattan(human: np.ndarray, probs: np.ndarray) -> float:
    """Mean L1 distance between distributions (LeWiDi 2025 soft metric)."""
    return float(np.abs(_normalise(human) - _normalise(probs)).sum(axis=1).mean())


def wasserstein(human: np.ndarray, probs: np.ndarray) -> float:
    """Mean 1-Wasserstein distance with classes placed at 0..C-1 (LeWiDi 2025 soft metric)."""
    cdf_gap = np.cumsum(_normalise(human) - _normalise(probs), axis=1)
    return float(np.abs(cdf_gap).sum(axis=1).mean())


def auroc(is_error: np.ndarray, score: np.ndarray) -> float:
    is_error = np.asarray(is_error, dtype=bool)
    if is_error.all() or not is_error.any():
        return float("nan")
    return float(roc_auc_score(is_error, score))


def auprc(is_error: np.ndarray, score: np.ndarray) -> float:
    is_error = np.asarray(is_error, dtype=bool)
    if not is_error.any():
        return float("nan")
    return float(average_precision_score(is_error, score))


def classification_report(
    probs: np.ndarray, labels: np.ndarray, human: np.ndarray | None = None, prefix: str = ""
) -> dict[str, float]:
    """Every classifier metric the study reports, keyed with an optional prefix."""
    out = {
        "acc": accuracy(probs, labels),
        "ece": ece(probs, labels),
        "nll": nll(probs, labels),
    }
    if human is not None:
        out |= {
            "kl": kl(human, probs),
            "js": js(human, probs),
            "manhattan": manhattan(human, probs),
            "wasserstein": wasserstein(human, probs),
        }
    return {f"{prefix}{k}": v for k, v in out.items()}
