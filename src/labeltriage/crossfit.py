"""Cross-fitting: score every item with a model that never saw it.

Fold assignment depends only on the item ID and a seed, so the same item lands in the same fold
for every feature group, the probe models and the injection model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator

import numpy as np


def fold_ids(item_ids, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Stable fold index per item, from a hash of ``(seed, item_id)``."""
    out = np.empty(len(item_ids), dtype=np.int64)
    for i, item in enumerate(item_ids):
        h = hashlib.blake2b(f"{seed}:{item}".encode(), digest_size=8).digest()
        out[i] = int.from_bytes(h, "little") % n_folds
    return out


def splits(
    item_ids, n_folds: int = 5, seed: int = 0
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold, train_mask, held_out_mask)`` for each fold."""
    folds = fold_ids(item_ids, n_folds, seed)
    for f in range(n_folds):
        yield f, folds != f, folds == f


def cross_predict(
    item_ids,
    fit_predict: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Stack held-out predictions: ``fit_predict(fold, train_mask, held_out_mask)`` returns the
    predictions for the held-out rows, in order."""
    out = None
    for f, train, held in splits(item_ids, n_folds, seed):
        pred = np.asarray(fit_predict(f, train, held))
        if out is None:
            out = np.zeros((len(item_ids), *pred.shape[1:]), dtype=pred.dtype)
        out[held] = pred
    return out
