"""Training targets built from votes: what the classifier is asked to fit for each item.

A :class:`TargetSet` pairs every item with a target distribution over classes and a sample
weight. Majority vote is a one-hot target, soft labels are the vote shares. The LabelTriage
router (Phase 4) produces the same structure, so every method shares one trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from labeltriage.data import store


@dataclass(frozen=True)
class TargetSet:
    item_ids: np.ndarray
    targets: np.ndarray  # (n, C), rows sum to 1
    weights: np.ndarray  # (n,)
    routes: np.ndarray | None = None  # (n,) route names, set by the router

    def __post_init__(self):
        n = len(self.item_ids)
        if self.targets.shape[0] != n or self.weights.shape != (n,):
            raise ValueError("item_ids, targets and weights must have matching lengths")
        if not np.allclose(self.targets.sum(axis=1), 1.0):
            raise ValueError("every target row must sum to 1")

    def __len__(self) -> int:
        return len(self.item_ids)

    def subset(self, mask: np.ndarray) -> TargetSet:
        return self._take(mask)

    def _take(self, idx) -> TargetSet:
        routes = None if self.routes is None else self.routes[idx]
        return TargetSet(self.item_ids[idx], self.targets[idx], self.weights[idx], routes)

    def align(self, item_ids) -> TargetSet:
        """Reorder rows to follow ``item_ids``; every requested item must be present."""
        pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        missing = pd.Index(item_ids).difference(pos.index)
        if len(missing):
            raise KeyError(f"{len(missing)} items have no target, e.g. {list(missing[:3])}")
        return self._take(pos.loc[list(item_ids)].to_numpy())


def majority(
    votes: pd.DataFrame, num_classes: int, rng: np.random.Generator | None = None
) -> TargetSet:
    """One-hot majority vote. Ties are broken uniformly at random, not by class order."""
    vm = store.vote_matrix(votes, num_classes)
    rng = rng or np.random.default_rng(0)
    top = vm.counts == vm.counts.max(axis=1, keepdims=True)
    noise = rng.random(top.shape) * top
    winner = noise.argmax(axis=1)
    onehot = np.eye(num_classes)[winner]
    return TargetSet(vm.item_ids, onehot, np.ones(len(vm.item_ids)))


def soft(votes: pd.DataFrame, num_classes: int) -> TargetSet:
    """Raw vote shares as the target distribution."""
    vm = store.vote_matrix(votes, num_classes)
    return TargetSet(vm.item_ids, vm.distribution, np.ones(len(vm.item_ids)))


CLEAR, AMBIGUOUS, UNRELIABLE = "clear", "ambiguous", "unreliable"


def entropy(p: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy in nats."""
    logp = np.log(np.where(p > 0, p, 1.0))
    return -(p * logp).sum(axis=1)


def route(
    votes: pd.DataFrame,
    num_classes: int,
    p_error: np.ndarray,
    nu: float = 1.5,
    h0: float = 0.3,
    hard_drop: float | None = None,
) -> TargetSet:
    """Build per-item targets from votes weighted by ``1 - p(error)`` (blueprint section 8.4).

    With surviving weight ``n_i`` and cleaned distribution ``q_i``:

    - **unreliable** if ``n_i < nu``: target ``q_i``, sample weight ``n_i / K_i``;
    - **clear** if ``H(q_i) <= h0``: one-hot ``argmax q_i``, weight 1;
    - **ambiguous** otherwise: target ``q_i``, weight 1.

    ``hard_drop`` replaces soft weighting with ``w = 1[e <= hard_drop]`` (an ablation). Passing
    the true ``is_error`` column as ``p_error`` gives the oracle triage baseline.
    """
    e = np.asarray(p_error, dtype=float)
    if e.shape != (len(votes),):
        raise ValueError("p_error must have one entry per vote")
    w = (e <= hard_drop).astype(float) if hard_drop is not None else 1.0 - e

    item_codes, item_ids = pd.factorize(votes["item_id"], sort=True)
    weighted = np.zeros((len(item_ids), num_classes))
    np.add.at(weighted, (item_codes, votes["label"].to_numpy()), w)
    k = np.bincount(item_codes, minlength=len(item_ids)).astype(float)
    n = weighted.sum(axis=1)

    q = np.full_like(weighted, 1.0 / num_classes)
    alive = n > 0
    q[alive] = weighted[alive] / n[alive, None]

    unreliable = n < nu
    clear = ~unreliable & (entropy(q) <= h0)
    targets = np.where(clear[:, None], np.eye(num_classes)[q.argmax(axis=1)], q)
    weights = np.where(unreliable, n / k, 1.0)
    routes = np.where(unreliable, UNRELIABLE, np.where(clear, CLEAR, AMBIGUOUS))
    return TargetSet(np.asarray(item_ids), targets, weights, routes)


def build(
    kind: str, votes: pd.DataFrame, num_classes: int, rng: np.random.Generator | None = None
) -> TargetSet:
    if kind == "hard":
        return majority(votes, num_classes, rng)
    if kind == "soft":
        return soft(votes, num_classes)
    raise KeyError(f"unknown target kind {kind!r}")
