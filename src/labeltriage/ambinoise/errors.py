"""The four AmbiNoise error processes.

Each process maps a vote's original label to a wrong label, and says which votes it can corrupt
at all (a swap cannot touch a class outside its pairs; a lazy annotator cannot be wrong on an item
whose true vote already is the default class). Every corrupted label differs from the original.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Disjoint class pairs swapped by the "swap" process. CIFAR-10 pairs visually confusable
# classes; NLI swaps entailment and contradiction and leaves neutral untouched.
SWAP_PAIRS: dict[int, tuple[tuple[int, int], ...]] = {
    10: ((3, 5), (1, 9), (4, 7), (0, 8), (2, 6)),  # cat-dog, car-truck, deer-horse, ...
    3: ((0, 2),),  # entailment-contradiction
}

ERROR_TYPES = ("spammer", "swap", "lazy", "plausible_wrong")


@dataclass(frozen=True)
class ErrorContext:
    num_classes: int
    lazy_class: int  # the most common class in the dataset
    ranking: np.ndarray | None = None  # (n_items, C) injection-model probabilities (A1)


def swap_map(num_classes: int) -> np.ndarray:
    """``m[c]`` is the class ``c`` swaps to, or -1 if ``c`` is not in any pair."""
    m = np.full(num_classes, -1)
    # Other class counts fall back to pairing neighbours: 0-1, 2-3, ...
    default = tuple((c, c + 1) for c in range(0, num_classes - 1, 2))
    for a, b in SWAP_PAIRS.get(num_classes, default):
        m[a], m[b] = b, a
    return m


def eligible(kind: str, labels: np.ndarray, ctx: ErrorContext) -> np.ndarray:
    """Which votes ``kind`` is able to corrupt."""
    if kind in ("spammer", "plausible_wrong"):
        return np.ones(len(labels), dtype=bool)
    if kind == "swap":
        return swap_map(ctx.num_classes)[labels] >= 0
    if kind == "lazy":
        return labels != ctx.lazy_class
    raise KeyError(f"unknown error type {kind!r}")


def corrupt(
    kind: str,
    labels: np.ndarray,
    item_rows: np.ndarray,
    ctx: ErrorContext,
    rng: np.random.Generator,
) -> np.ndarray:
    """Wrong labels for the given votes. ``item_rows`` indexes ``ctx.ranking`` per vote."""
    c = ctx.num_classes
    if kind == "spammer":
        # Uniform over the other C-1 classes.
        return (labels + rng.integers(1, c, size=len(labels))) % c
    if kind == "swap":
        out = swap_map(c)[labels]
        if (out < 0).any():
            raise ValueError("swap asked to corrupt a class outside its pairs")
        return out
    if kind == "lazy":
        if (labels == ctx.lazy_class).any():
            raise ValueError("lazy asked to corrupt a vote that already is the default class")
        return np.full(len(labels), ctx.lazy_class)
    if kind == "plausible_wrong":
        if ctx.ranking is None:
            raise ValueError("plausible_wrong needs the injection model's ranking (amendment A1)")
        scores = ctx.ranking[item_rows].astype(float).copy()
        scores[np.arange(len(labels)), labels] = -np.inf
        return scores.argmax(axis=1)
    raise KeyError(f"unknown error type {kind!r}")
