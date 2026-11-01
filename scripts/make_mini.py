"""Freeze the mini subsets used by the local CPU profile (``profile=mini``, amendment A15).

    python scripts/make_mini.py                 # write conf/splits/mini/*.csv
    python scripts/make_mini.py --seed-cache    # also copy matching rows of cache/ to cache_mini/

Each subset is drawn inside each item's original split and stratified by the item's majority
vote, so the mini data keeps the class balance and the train / val / test boundaries of the full
data. Nothing under data/ is changed. The draw is deterministic; the CSVs are committed so every
machine uses the same items.

``--seed-cache`` reuses frozen-model outputs already computed for the full data (the ChaosNLI and
VariErr embeddings, for example), so the mini profile does not recompute them on a CPU.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from labeltriage.data import images, prepare, splits, store
from labeltriage.experiments import embed

SUBSET = "mini"
MINI_SEED = 20261101

SIZES: dict[str, dict[str, int]] = {
    "cifar10h": {"train": 1000, "val": 200, "test": 300},
    "chaosnli": {"train": 800, "val": 150, "test": 250},
}
# Clean CIFAR-10 train images used as clean_dev (val) and clean_test (test) on CIFAR-10H.
CLEAN_SIZES = {"val": 500, "test": 1000}


def take(ids: np.ndarray, strata: np.ndarray, n: int) -> np.ndarray:
    """``n`` of ``ids``, keeping the strata's proportions."""
    if n >= len(ids):
        return ids
    split = splits.stratified_split(ids, strata, {"keep": n, "drop": len(ids) - n}, MINI_SEED)
    return split.index[split == "keep"].to_numpy()


def mini_split(votes: pd.DataFrame, items: pd.DataFrame, k: int, sizes: dict) -> pd.Series:
    vm = store.vote_matrix(votes, k)
    majority = pd.Series(vm.majority, index=vm.item_ids)
    parts = []
    for name, n in sizes.items():
        ids = items.index[items["split"] == name].to_numpy()
        kept = take(ids, majority.loc[ids].to_numpy(), n)
        parts.append(pd.Series(name, index=pd.Index(kept, name="item_id"), name="split"))
    return pd.concat(parts)


def clean_split(data_dir: str) -> pd.Series:
    _, labels = images.cifar10(data_dir, train=True)
    full = splits.load_split("cifar10_train_clean")
    parts = []
    for name, n in CLEAN_SIZES.items():
        ids = full.index[full == name].to_numpy()
        kept = take(ids, labels[ids.astype(int)], n)
        parts.append(pd.Series(name, index=pd.Index(kept, name="item_id"), name="split"))
    return pd.concat(parts)


def seed_cache(cache: Path, mini_cache: Path) -> list[str]:
    """Copy frozen-model outputs for the mini items; VariErr is small and copied whole."""
    copied = []
    for base in SIZES:
        keep = prepare.subset_ids(base, SUBSET)
        for kind, path in embed.cache_paths(cache, base).items():
            if not path.exists():
                continue
            frame = embed.load(path)
            if not keep.isin(frame.index).all():
                continue
            embed.save(keep, frame.loc[keep].to_numpy(), embed.cache_paths(mini_cache, base)[kind])
            copied.append(f"{kind}/{base}")
    for path in embed.cache_paths(cache, "varierr").values():
        if path.exists():
            out = mini_cache / path.relative_to(cache)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)
            copied.append(path.relative_to(cache).as_posix())
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed-cache", action="store_true")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--mini-cache", default="cache_mini")
    args = parser.parse_args(argv)

    out = splits.splits_dir() / SUBSET
    for name, sizes in SIZES.items():
        votes, items, meta = prepare.load_prepared(name, args.data_dir)
        split = mini_split(votes, items, len(meta["classes"]), sizes)
        splits.save_split(split, name, out)
        print(f"{name}: {split.value_counts().to_dict()} of {len(items)} items")
    split = clean_split(args.data_dir)
    splits.save_split(split, "cifar10_train_clean", out)
    print(f"cifar10_train_clean: {split.value_counts().to_dict()}")
    # VariErr is already small (500 items, transfer only), so the mini profile keeps all of it.
    splits.save_split(splits.load_split("varierr"), "varierr", out)

    if args.seed_cache:
        for line in seed_cache(Path(args.cache), Path(args.mini_cache)):
            print(f"cache_mini <- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
