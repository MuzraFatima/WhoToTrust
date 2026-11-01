"""Aggregation baselines (blueprint section 10). Each returns a :class:`TargetSet`.

Confident Learning and CROWDLAB need out-of-sample class probabilities. Every baseline gets the
same ones: a logistic-regression probe on the frozen embeddings G4 uses, cross-fitted over item
folds and trained on majority labels, so no item is scored by a model that saw its votes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from labeltriage import crossfit, targets
from labeltriage.crowd import dawid_skene
from labeltriage.data import store


def oof_pred_probs(
    item_ids: np.ndarray,
    embeddings: pd.DataFrame,
    labels: np.ndarray,
    num_classes: int,
    seed: int = 0,
    n_folds: int = 5,
) -> np.ndarray:
    """Out-of-fold class probabilities for ``item_ids`` from a probe on ``embeddings``."""
    x = StandardScaler().fit_transform(embeddings.loc[item_ids].to_numpy(dtype=np.float32))

    def fit_predict(fold, train, held):
        clf = LogisticRegression(max_iter=1000, C=0.5, random_state=seed + fold)
        clf.fit(x[train], labels[train])
        p = np.zeros((held.sum(), num_classes))
        p[:, clf.classes_] = clf.predict_proba(x[held])
        return p

    probs = crossfit.cross_predict(item_ids, fit_predict, n_folds, seed)
    probs = np.clip(probs, 1e-6, None)
    return probs / probs.sum(axis=1, keepdims=True)


def dawid_skene_targets(votes: pd.DataFrame, num_classes: int) -> targets.TargetSet:
    ds = dawid_skene(votes, num_classes)
    onehot = np.eye(num_classes)[ds.posterior.argmax(axis=1)]
    return targets.TargetSet(ds.item_ids, onehot, np.ones(len(ds.item_ids)))


def entropy_weight_targets(
    votes: pd.DataFrame, num_classes: int, rng: np.random.Generator
) -> targets.TargetSet:
    """Majority label, sample weight ``1 - H(votes) / log C``: the naive "disagreement means
    unreliable" rule, with no learning."""
    mv = targets.majority(votes, num_classes, rng)
    dist = store.vote_matrix(votes, num_classes).distribution
    w = 1.0 - targets.entropy(dist) / np.log(num_classes)
    return targets.TargetSet(mv.item_ids, mv.targets, w)


def confident_learning_targets(
    votes: pd.DataFrame, num_classes: int, pred_probs: np.ndarray, rng: np.random.Generator
) -> targets.TargetSet:
    """Majority labels with Confident Learning's likely issues pruned (weight 0).

    ``pred_probs`` rows follow the sorted item order of :func:`store.vote_matrix`.
    """
    from cleanlab.filter import find_label_issues

    mv = targets.majority(votes, num_classes, rng)
    labels = mv.targets.argmax(axis=1)
    issues = find_label_issues(labels, pred_probs, filter_by="prune_by_noise_rate", n_jobs=1)
    return targets.TargetSet(mv.item_ids, mv.targets, (~issues).astype(float))


def crowdlab_targets(
    votes: pd.DataFrame, num_classes: int, pred_probs: np.ndarray
) -> targets.TargetSet:
    """CROWDLAB consensus labels; items Confident Learning flags against them are filtered."""
    from cleanlab.filter import find_label_issues
    from cleanlab.multiannotator import get_label_quality_multiannotator

    matrix = votes.pivot_table(
        index="item_id", columns="annotator_id", values="label", aggfunc="first"
    ).sort_index()
    out = get_label_quality_multiannotator(
        matrix, pred_probs, verbose=False, return_annotator_stats=False
    )
    consensus = out["label_quality"]["consensus_label"].to_numpy(dtype=int)
    issues = find_label_issues(consensus, pred_probs, filter_by="prune_by_noise_rate", n_jobs=1)
    onehot = np.eye(num_classes)[consensus]
    return targets.TargetSet(matrix.index.to_numpy(), onehot, (~issues).astype(float))
