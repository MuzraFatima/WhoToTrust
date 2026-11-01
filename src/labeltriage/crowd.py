"""Dawid–Skene (1979): EM over per-annotator confusion matrices and latent true labels.

Used twice: as G2's annotator-reliability signal and as the Dawid–Skene aggregation baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DawidSkene:
    item_ids: np.ndarray
    annotator_ids: np.ndarray
    posterior: np.ndarray  # (n_items, C) p(true class | votes)
    confusion: np.ndarray  # (n_annotators, C, C) p(vote = l | true = c)
    prior: np.ndarray  # (C,)

    @property
    def annotator_accuracy(self) -> pd.Series:
        """Expected probability that each annotator's vote equals the true class."""
        acc = np.einsum("c,acc->a", self.prior, self.confusion)
        return pd.Series(acc, index=self.annotator_ids, name="ds_accuracy")


def dawid_skene(
    votes: pd.DataFrame,
    num_classes: int,
    max_iter: int = 50,
    tol: float = 1e-5,
    smoothing: float = 0.01,
) -> DawidSkene:
    """Fit Dawid–Skene by EM, initialised from vote shares (soft majority vote)."""
    i_codes, item_ids = pd.factorize(votes["item_id"], sort=True)
    a_codes, ann_ids = pd.factorize(votes["annotator_id"], sort=True)
    labels = votes["label"].to_numpy()
    n_i, n_a, c = len(item_ids), len(ann_ids), num_classes

    post = np.zeros((n_i, c))
    np.add.at(post, (i_codes, labels), 1.0)
    post /= post.sum(axis=1, keepdims=True)

    log_like_prev = -np.inf
    for _ in range(max_iter):
        # M-step: class prior and confusion matrices from soft assignments.
        prior = post.mean(axis=0) + 1e-12
        conf = np.full((n_a, c, c), smoothing)
        np.add.at(conf, (a_codes, slice(None), labels), post[i_codes])
        conf /= conf.sum(axis=2, keepdims=True)

        # E-step: item posteriors given every vote on the item.
        log_post = np.tile(np.log(prior), (n_i, 1))
        np.add.at(log_post, i_codes, np.log(conf[a_codes, :, labels]))
        log_norm = np.logaddexp.reduce(log_post, axis=1)
        post = np.exp(log_post - log_norm[:, None])

        log_like = log_norm.sum()
        if abs(log_like - log_like_prev) < tol * max(1.0, abs(log_like)):
            break
        log_like_prev = log_like

    return DawidSkene(np.asarray(item_ids), np.asarray(ann_ids), post, conf, prior)
