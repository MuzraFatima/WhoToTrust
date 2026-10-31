"""The annotation store: one long table with a row per (item, annotator, label) vote.

An annotator normally casts one vote per item, but some datasets (VariErr) let an annotator pick
several labels for the same item, so uniqueness is on the label too.

Every stage of the pipeline reads and writes this table, so its schema is the contract between
modules. Optional columns are filled in by later stages (``is_error`` by the AmbiNoise injector,
``split`` by the splitter, and so on).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: dict[str, str] = {
    "dataset": "string",
    "item_id": "string",
    "annotator_id": "string",
    "label": "int64",
}

OPTIONAL_COLUMNS: dict[str, str] = {
    "split": "string",
    "is_error": "boolean",
    "error_type": "string",
}

SPLITS = ("train", "val", "test")


class SchemaError(ValueError):
    """Raised when a table does not follow the annotation-store schema."""


def validate(df: pd.DataFrame, num_classes: int | None = None) -> pd.DataFrame:
    """Check ``df`` against the schema and return it with canonical dtypes.

    Raises :class:`SchemaError` on missing columns, null keys, negative or out-of-range labels,
    unknown split names, or the same (item, annotator, label) vote appearing twice.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            raise SchemaError(f"column {col!r} has null values")

    dtypes = {c: t for c, t in {**REQUIRED_COLUMNS, **OPTIONAL_COLUMNS}.items() if c in df}
    out = df.astype(dtypes)

    if (out["label"] < 0).any():
        raise SchemaError("labels must be non-negative class indices")
    if num_classes is not None and (out["label"] >= num_classes).any():
        raise SchemaError(f"labels must be < num_classes={num_classes}")

    if "split" in out:
        bad = set(out["split"].dropna().unique()) - set(SPLITS)
        if bad:
            raise SchemaError(f"unknown split names: {sorted(bad)}")

    dup = out.duplicated(subset=["dataset", "item_id", "annotator_id", "label"])
    if dup.any():
        raise SchemaError(
            f"{int(dup.sum())} duplicate (dataset, item_id, annotator_id, label) rows"
        )

    return out.reset_index(drop=True)


@dataclass(frozen=True)
class VoteMatrix:
    """Per-item vote counts, aligned with ``item_ids``."""

    item_ids: np.ndarray
    counts: np.ndarray  # shape (n_items, num_classes)

    @property
    def distribution(self) -> np.ndarray:
        totals = self.counts.sum(axis=1, keepdims=True)
        return self.counts / np.maximum(totals, 1)

    @property
    def majority(self) -> np.ndarray:
        """Majority class per item; ties go to the lowest class index."""
        return self.counts.argmax(axis=1)


def vote_matrix(df: pd.DataFrame, num_classes: int, item_order=None) -> VoteMatrix:
    """Count votes per item and class. ``item_order`` fixes the row order if given."""
    counts = pd.crosstab(df["item_id"], df["label"]).reindex(
        columns=range(num_classes), fill_value=0
    )
    if item_order is not None:
        counts = counts.reindex(index=pd.Index(item_order), fill_value=0)
    return VoteMatrix(item_ids=counts.index.to_numpy(), counts=counts.to_numpy(dtype=np.int64))


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate(df).to_parquet(path, index=False)
    return path


def load(path: str | Path, num_classes: int | None = None) -> pd.DataFrame:
    return validate(pd.read_parquet(path), num_classes=num_classes)
