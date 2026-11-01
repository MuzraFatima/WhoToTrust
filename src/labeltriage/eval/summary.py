"""Collect finished runs into one flat table: a row per run, a column per metric."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from labeltriage.utils.runs import DONE, RESULT


def collect(root: str | Path) -> pd.DataFrame:
    rows = []
    for done in sorted(Path(root).rglob(DONE)):
        result = json.loads((done.parent / RESULT).read_text())
        cfg = result.get("config", {})
        rows.append(
            {
                "experiment": cfg.get("experiment", {}).get("name"),
                "data": cfg.get("data", {}).get("name"),
                "method": cfg.get("method", {}).get("name"),
                **{
                    f"method.{k}": cfg["method"][k]
                    for k in ("tag", "nu", "h0", "hard_drop", "model")
                    if k in cfg.get("method", {})
                },
                "seed": cfg.get("seed"),
                "stage": cfg.get("stage"),
                "train_seconds": result.get("train_seconds"),
                "device": result.get("device"),
                **result.get("metrics", {}),
            }
        )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, by=("experiment", "data", "method")) -> pd.DataFrame:
    """Mean and standard deviation over seeds for every numeric column."""
    numeric = df.select_dtypes("number").columns.drop("seed", errors="ignore")
    grouped = df.groupby(list(by))[list(numeric)]
    out = grouped.agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.join(df.groupby(list(by)).size().rename("n_seeds")).reset_index()
