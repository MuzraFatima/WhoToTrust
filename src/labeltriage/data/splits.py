"""Deterministic, stratified train / val / test splits.

Splits are drawn once with NumPy's ``Generator`` (stable across library versions), stratified by
each item's majority vote, and saved as CSV files under ``conf/splits`` so every run reads the
exact same assignment.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from labeltriage.data import store

SPLIT_SEED = 20261031


def stratified_split(
    item_ids: np.ndarray,
    strata: np.ndarray,
    sizes: dict[str, float | int],
    seed: int = SPLIT_SEED,
) -> pd.Series:
    """Assign each item to a split, keeping every stratum's proportions.

    ``sizes`` maps split name to either a fraction (all values < 1, summing to 1) or an absolute
    count (summing to ``len(item_ids)``). Rounding leftovers go to the largest split.
    """
    n = len(item_ids)
    names = list(sizes)
    values = np.array([sizes[k] for k in names], dtype=float)
    fractions = values / n if values.sum() > 1 else values
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"split sizes must cover all {n} items, got {sizes}")

    targets = np.floor(fractions * n).astype(int)
    targets[np.argmax(targets)] += n - targets.sum()

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    # Interleave strata so taking a prefix of the order keeps class proportions.
    strata = np.asarray(strata)[order]
    rank = np.empty(n)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rank[idx] = (np.arange(len(idx)) + rng.random()) / len(idx)
    order = order[np.argsort(rank, kind="stable")]

    assignment = np.empty(n, dtype=object)
    bounds = np.concatenate([[0], np.cumsum(targets)])
    for name, lo, hi in zip(names, bounds[:-1], bounds[1:], strict=True):
        assignment[order[lo:hi]] = name
    return pd.Series(assignment, index=pd.Index(item_ids, name="item_id"), name="split")


def split_by_majority(
    votes: pd.DataFrame, num_classes: int, sizes: dict[str, float | int], seed: int = SPLIT_SEED
) -> pd.Series:
    vm = store.vote_matrix(votes, num_classes)
    return stratified_split(vm.item_ids, vm.majority, sizes, seed)


def assign(votes: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Attach a ``split`` column to the vote table. Every item must have a split."""
    out = votes.drop(columns="split", errors="ignore")
    out = out.assign(split=out["item_id"].map(split))
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "item_id"].nunique()
        raise ValueError(f"{missing} items have no split")
    return store.validate(out)


def splits_dir() -> Path:
    return Path(str(resources.files("labeltriage") / "conf" / "splits"))


def save_split(split: pd.Series, name: str, directory: Path | None = None) -> Path:
    path = (directory or splits_dir()) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    split.sort_index().to_csv(path, header=True, lineterminator="\n")
    return path


def load_split(name: str, directory: Path | None = None) -> pd.Series:
    path = (directory or splits_dir()) / f"{name}.csv"
    df = pd.read_csv(path, dtype={"item_id": str, "split": str})
    return df.set_index("item_id")["split"]
