"""Counterfactual training sets for the harm analysis (blueprint section 8.5, H3).

Using AmbiNoise's known error labels, every item gets one of four targets, all built from the same
sampled votes, so data, size and seeds match across counterfactuals:

- ``neither``: the observed vote shares (errors and ambiguity both present);
- ``errors_removed``: shares of the non-error votes only;
- ``ambiguity_collapsed``: error votes kept, but all non-error votes moved to the item's clean
  majority class, so genuine disagreement disappears while errors stay;
- ``both``: one-hot on the clean majority.

The accuracy and ECE differences from ``neither`` attribute harm to each disagreement category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeltriage.targets import TargetSet

COUNTERFACTUALS = ("neither", "errors_removed", "ambiguity_collapsed", "both")


def harm_targets(votes: pd.DataFrame, num_classes: int, which: str) -> TargetSet:
    if which not in COUNTERFACTUALS:
        raise KeyError(f"unknown counterfactual {which!r}")
    codes, item_ids = pd.factorize(votes["item_id"], sort=True)
    labels = votes["label"].to_numpy()
    err = votes["is_error"].to_numpy(dtype=bool)
    n = len(item_ids)

    all_counts = np.zeros((n, num_classes))
    np.add.at(all_counts, (codes, labels), 1)
    clean = np.zeros((n, num_classes))
    np.add.at(clean, (codes[~err], labels[~err]), 1)
    errs = all_counts - clean
    has_clean = clean.sum(axis=1) > 0
    # Items whose every vote is an error fall back to their observed votes in every variant.
    clean_major = np.where(has_clean, clean.argmax(axis=1), all_counts.argmax(axis=1))
    onehot = np.eye(num_classes)[clean_major]

    if which == "neither":
        counts = all_counts
    elif which == "errors_removed":
        counts = np.where(has_clean[:, None], clean, all_counts)
    elif which == "ambiguity_collapsed":
        counts = errs + onehot * clean.sum(axis=1, keepdims=True)
        counts = np.where(has_clean[:, None], counts, all_counts)
    else:
        counts = onehot

    dist = counts / counts.sum(axis=1, keepdims=True)
    return TargetSet(np.asarray(item_ids), dist, np.ones(n))
