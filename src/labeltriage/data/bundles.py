"""Assemble what one training run needs: train inputs, train votes and named evaluation sets.

Evaluation sets are built lazily and each declares which split it reads, so the stage guard
(:func:`labeltriage.utils.runs.require_stage`) refuses test-only sets outside ``stage=final``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from labeltriage.data import images, prepare, splits, store
from labeltriage.utils.runs import require_stage


@dataclass
class EvalSet:
    inputs: Any
    labels: np.ndarray
    human: np.ndarray | None = None


@dataclass
class EvalSpec:
    split: str  # "val" (readable any time) or "test" (stage=final only)
    build: Callable[[], EvalSet]


@dataclass
class Bundle:
    domain: str  # "image" or "text"
    num_classes: int
    train_inputs: Any  # aligned with train_item_ids
    train_item_ids: np.ndarray
    train_votes: pd.DataFrame
    evals: dict[str, EvalSpec] = field(default_factory=dict)
    # Train and val votes together (never test votes), with a split column. Triage tuning uses the
    # val part.
    votes: pd.DataFrame | None = None
    # Inputs for any item of the base dataset, in the order of the IDs given.
    inputs_for: Callable[[np.ndarray], Any] | None = None

    def eval_set(self, name: str, stage: str) -> EvalSet:
        spec = self.evals[name]
        require_stage(stage, spec.split)
        return spec.build()


class LazyArray:
    """Array stand-in that builds the real array on first indexing (e.g. after a download)."""

    def __init__(self, build: Callable[[], np.ndarray], length: int):
        self._build, self._length, self._array = build, length, None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        if self._array is None:
            self._array = self._build()
        return self._array[idx]


def _split_votes(votes: pd.DataFrame, items: pd.DataFrame, split: str):
    ids = items.index[items["split"] == split].to_numpy()
    return ids, votes[votes["item_id"].isin(set(ids))]


def _dev_votes(votes: pd.DataFrame) -> pd.DataFrame:
    return votes[votes["split"].isin(["train", "val"])].reset_index(drop=True)


def _pairs(items: pd.DataFrame, ids) -> tuple[list[str], list[str]]:
    return items.loc[ids, "premise"].tolist(), items.loc[ids, "hypothesis"].tolist()


def _human(votes: pd.DataFrame, item_ids: np.ndarray, num_classes: int) -> np.ndarray:
    return store.vote_matrix(votes, num_classes, item_order=item_ids).distribution


def _clean_cifar10_train(data_dir, split: str, subset: str | None = None) -> EvalSet:
    x, y = images.cifar10(data_dir, train=True)
    directory = splits.splits_dir() / subset if subset else None
    clean = splits.load_split("cifar10_train_clean", directory)
    idx = clean.index[clean == split].astype(int).to_numpy()
    return EvalSet(x[idx], y[idx])


def _cifar10_c_mean(data_dir) -> EvalSet:
    """All corruptions at all severities, stacked; the caller averages accuracy over them."""
    xs, ys = [], []
    for c in images.CIFAR10_C_CORRUPTIONS:
        for s in range(1, 6):
            x, y = images.cifar10_c(data_dir, c, s)
            xs.append(x)
            ys.append(y)
    return EvalSet(np.concatenate(xs), np.concatenate(ys))


def cifar10h(data_dir: str | Path, subset: str | None = None) -> Bundle:
    votes, items, meta = prepare.load_prepared("cifar10h", data_dir, subset)
    k = len(meta["classes"])
    cache: dict[str, np.ndarray] = {}

    def test_images() -> np.ndarray:
        if "x" not in cache:
            cache["x"], _ = images.cifar10(data_dir, train=False)
        return cache["x"]

    def human_set(split: str) -> Callable[[], EvalSet]:
        def build() -> EvalSet:
            ids, v = _split_votes(votes, items, split)
            idx = items.loc[ids, "image_index"].to_numpy()
            return EvalSet(test_images()[idx], items.loc[ids, "gold"].to_numpy(), _human(v, ids, k))

        return build

    train_ids, train_votes = _split_votes(votes, items, "train")

    def inputs_for(ids) -> np.ndarray:
        return test_images()[items.loc[ids, "image_index"].to_numpy()]

    return Bundle(
        domain="image",
        num_classes=k,
        train_inputs=LazyArray(lambda: inputs_for(train_ids), len(train_ids)),
        train_item_ids=train_ids,
        train_votes=train_votes,
        votes=_dev_votes(votes),
        inputs_for=inputs_for,
        evals={
            "val": EvalSpec("val", human_set("val")),
            "clean_dev": EvalSpec("val", lambda: _clean_cifar10_train(data_dir, "val", subset)),
            "human_test": EvalSpec("test", human_set("test")),
            "clean_test": EvalSpec("test", lambda: _clean_cifar10_train(data_dir, "test", subset)),
            "cifar10_1": EvalSpec("test", lambda: EvalSet(*images.cifar10_1(data_dir))),
            "cifar10_c": EvalSpec("test", lambda: _cifar10_c_mean(data_dir)),
        },
    )


def cifar10n(data_dir: str | Path, subset: str | None = None) -> Bundle:
    votes, items, meta = prepare.load_prepared("cifar10n", data_dir, subset)
    k = len(meta["classes"])
    x_train, _ = images.cifar10(data_dir, train=True)
    train_ids, train_votes = _split_votes(votes, items, "train")

    def val() -> EvalSet:
        # Tuning must not see clean labels, so validation scores against the noisy majority.
        ids, v = _split_votes(votes, items, "val")
        human = _human(v, ids, k)
        return EvalSet(x_train[items.loc[ids, "image_index"].to_numpy()], human.argmax(1), human)

    return Bundle(
        domain="image",
        num_classes=k,
        train_inputs=x_train[items.loc[train_ids, "image_index"].to_numpy()],
        train_item_ids=train_ids,
        train_votes=train_votes,
        votes=_dev_votes(votes),
        inputs_for=lambda ids: x_train[items.loc[ids, "image_index"].to_numpy()],
        evals={
            "val": EvalSpec("val", val),
            "cifar10_test": EvalSpec(
                "test", lambda: EvalSet(*images.cifar10(data_dir, train=False))
            ),
            "cifar10_1": EvalSpec("test", lambda: EvalSet(*images.cifar10_1(data_dir))),
        },
    )


def chaosnli(data_dir: str | Path, subset: str | None = None) -> Bundle:
    votes, items, meta = prepare.load_prepared("chaosnli", data_dir, subset)
    k = len(meta["classes"])

    def text_set(split: str) -> Callable[[], EvalSet]:
        def build() -> EvalSet:
            ids, v = _split_votes(votes, items, split)
            human = _human(v, ids, k)
            # The dataset's own majority label; argmax would break its ties toward entailment.
            return EvalSet(_pairs(items, ids), items.loc[ids, "gold"].to_numpy(), human)

        return build

    train_ids, train_votes = _split_votes(votes, items, "train")
    return Bundle(
        domain="text",
        num_classes=k,
        train_inputs=_pairs(items, train_ids),
        train_item_ids=train_ids,
        train_votes=train_votes,
        votes=_dev_votes(votes),
        inputs_for=lambda ids: _pairs(items, ids),
        evals={"val": EvalSpec("val", text_set("val")), "test": EvalSpec("test", text_set("test"))},
    )


def ambinoise(base: Bundle, spec, ranking: pd.DataFrame | None = None) -> Bundle:
    """Replace a base bundle's train and val votes with an AmbiNoise draw over them.

    Evaluation sets are untouched: they score against clean labels and full human distributions.
    """
    from labeltriage.ambinoise.injector import inject

    noisy = inject(base.votes, base.num_classes, spec, ranking)
    train = noisy[noisy["item_id"].isin(set(base.train_item_ids))].reset_index(drop=True)
    return dataclasses.replace(base, train_votes=train, votes=noisy)


LOADERS: dict[str, Callable[..., Bundle]] = {
    "cifar10h": cifar10h,
    "cifar10n": cifar10n,
    "chaosnli": chaosnli,
}


def load(name: str, data_dir: str | Path, subset: str | None = None) -> Bundle:
    """``subset`` (e.g. ``"mini"``) restricts the bundle to a frozen subset of items."""
    return LOADERS[name](data_dir, subset)
