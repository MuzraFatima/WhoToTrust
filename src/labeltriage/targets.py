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

    def __post_init__(self):
        n = len(self.item_ids)
        if self.targets.shape[0] != n or self.weights.shape != (n,):
            raise ValueError("item_ids, targets and weights must have matching lengths")
        if not np.allclose(self.targets.sum(axis=1), 1.0):
            raise ValueError("every target row must sum to 1")

    def __len__(self) -> int:
        return len(self.item_ids)

    def subset(self, mask: np.ndarray) -> TargetSet:
        return TargetSet(self.item_ids[mask], self.targets[mask], self.weights[mask])

    def align(self, item_ids) -> TargetSet:
        """Reorder rows to follow ``item_ids``; every requested item must be present."""
        pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        missing = pd.Index(item_ids).difference(pos.index)
        if len(missing):
            raise KeyError(f"{len(missing)} items have no target, e.g. {list(missing[:3])}")
        idx = pos.loc[list(item_ids)].to_numpy()
        return TargetSet(self.item_ids[idx], self.targets[idx], self.weights[idx])


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


def build(
    kind: str, votes: pd.DataFrame, num_classes: int, rng: np.random.Generator | None = None
) -> TargetSet:
    if kind == "hard":
        return majority(votes, num_classes, rng)
    if kind == "soft":
        return soft(votes, num_classes)
    raise KeyError(f"unknown target kind {kind!r}")
