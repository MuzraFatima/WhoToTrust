"""Turn raw downloads into processed, split datasets under ``<data_dir>/processed/<name>/``.

    python -m labeltriage.data.prepare cifar10h cifar10n chaosnli varierr --data-dir data

Each dataset directory holds ``votes.parquet`` (the annotation store with a ``split`` column),
``items.parquet`` and ``meta.json``. VariErr is a transfer-only set and gets no split column.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from labeltriage.data import download, sources, splits, store

log = logging.getLogger(__name__)

SPLIT_SIZES: dict[str, dict[str, float | int]] = {
    "cifar10h": {"train": 7000, "val": 1000, "test": 2000},
    # CIFAR-10N's test sets are the clean CIFAR-10 test set and CIFAR-10.1, so its own votes
    # only need a validation hold-out for tuning.
    "cifar10n": {"train": 45000, "val": 5000},
    "chaosnli": {"train": 0.7, "val": 0.1, "test": 0.2},
}


def chaosnli_paths(data_dir: str | Path) -> list[Path]:
    """Best available ChaosNLI files: the official SNLI + MNLI jsonl files, else the official zip,
    else the MNLI-only Hugging Face mirror (development fallback, with a warning).

    alphaNLI (``chaosNLI_alphanli.jsonl``) is a different task (pick the likelier of two
    hypotheses, labels 1/2) and is deliberately not loaded.
    """
    try:
        return [
            download.fetch("chaosnli_snli", data_dir),
            download.fetch("chaosnli_mnli_m", data_dir),
        ]
    except FileNotFoundError:
        pass
    try:
        return [download.fetch("chaosnli", data_dir)]
    except FileNotFoundError as err:
        log.warning("%s; falling back to the MNLI-only mirror for development", err)
        return [download.fetch("chaosnli_mnli_mirror", data_dir)]


def load_source(name: str, data_dir: str | Path) -> sources.Source:
    if name == "cifar10h":
        return sources.cifar10h(download.fetch("cifar10h", data_dir))
    if name == "cifar10n":
        return sources.cifar10n(
            download.fetch("cifar10n_labels", data_dir),
            download.fetch("cifar10n_side_info", data_dir),
        )
    if name == "varierr":
        return sources.varierr(download.fetch("varierr", data_dir))
    if name == "chaosnli":
        return sources.chaosnli(chaosnli_paths(data_dir))
    raise KeyError(f"unknown dataset {name!r}")


def finalise(
    src: sources.Source, split: pd.Series | None, meta: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Attach ``split`` (if any) to votes and items, and describe the result in ``meta``."""
    votes, items = src.votes, src.items.copy()
    if split is not None:
        votes = splits.assign(votes, split)
        items["split"] = items.index.map(split)
    per_item = votes.groupby("item_id").size()
    meta = {
        "name": src.name,
        "classes": list(src.classes),
        "num_items": len(items),
        "num_votes": len(votes),
        "num_annotators": int(votes["annotator_id"].nunique()),
        "votes_per_item": {"min": int(per_item.min()), "median": float(per_item.median())},
        **({"split_counts": items["split"].value_counts().to_dict()} if split is not None else {}),
        **(meta or {}),
    }
    return votes, items, meta


def resolve_split(name: str, src: sources.Source, out_dir: Path) -> pd.Series:
    """Use the committed split file if there is one, otherwise draw it and save it locally."""
    try:
        split = splits.load_split(name)
    except FileNotFoundError:
        split = splits.split_by_majority(src.votes, src.num_classes, SPLIT_SIZES[name])
        path = splits.save_split(split, name, out_dir)
        log.warning("no committed split for %s; drew one at %s (commit it to freeze)", name, path)
    missing = set(src.items.index) - set(split.index)
    if missing:
        raise ValueError(f"split for {name} is missing {len(missing)} items; was it redrawn?")
    return split.loc[list(src.items.index)]


def prepare(name: str, data_dir: str | Path) -> Path:
    out_dir = Path(data_dir) / "processed" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    src = load_source(name, data_dir)
    extra: dict = {}

    if name == "chaosnli":
        overlap = sources.nli_overlap(src.items, load_source("varierr", data_dir).items)
        src = sources.drop_items(src, overlap)
        extra["varierr_overlap_removed"] = len(overlap)
        extra["parts"] = src.items["source"].value_counts().to_dict()

    split = None if name == "varierr" else resolve_split(name, src, out_dir)
    votes, items, meta = finalise(src, split, extra)

    store.save(votes, out_dir / "votes.parquet")
    items.to_parquet(out_dir / "items.parquet")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=int))
    log.info("prepared %s: %s", name, meta)
    return out_dir


def load_prepared(name: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = Path(data_dir) / "processed" / name
    meta = json.loads((d / "meta.json").read_text())
    votes = store.load(d / "votes.parquet", num_classes=len(meta["classes"]))
    return votes, pd.read_parquet(d / "items.parquet"), meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=["cifar10h", "cifar10n", "chaosnli", "varierr"])
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in args.names:
        prepare(name, args.data_dir)


if __name__ == "__main__":
    main()
