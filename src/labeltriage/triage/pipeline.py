"""Score every train and val vote with LabelTriage, without any vote scoring itself.

- Train votes get out-of-fold scores: five triage models, each fitted on the other folds' train
  items, so the router never acts on a score from a model that saw that vote's error label.
- Val items are split in half by item: one half calibrates (isotonic), the other half measures
  AUROC and calibration error, so the Phase 4 gate is not judged on the calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from labeltriage import crossfit
from labeltriage.eval.metrics import auprc, auroc
from labeltriage.triage.model import TriageModel, calibration_error


@dataclass
class TriageResult:
    p_error: np.ndarray  # aligned with the input table; train votes out-of-fold
    model: TriageModel  # fitted on all train votes, calibrated on the val calibration half
    metrics: dict[str, float]


def score(
    table: pd.DataFrame,
    columns: list[str],
    kind: str = "lgbm",
    n_folds: int = 5,
    seed: int = 0,
) -> TriageResult:
    split = table["split"].to_numpy()
    y = table["is_error"].to_numpy(dtype=bool)
    items = table["item_id"].to_numpy()
    train = split == "train"
    val = split == "val"
    half = crossfit.fold_ids(items, 2, seed + 7)
    calib, held = val & (half == 0), val & (half == 1)

    p = np.full(len(table), np.nan)
    train_idx = np.flatnonzero(train)
    folds = crossfit.fold_ids(items[train_idx], n_folds, seed)
    for f in range(n_folds):
        fit_rows = train_idx[folds != f]
        out_rows = train_idx[folds == f]
        m = TriageModel(columns, kind, seed + f).fit(table.iloc[fit_rows], y[fit_rows])
        m.calibrate(table[calib], y[calib])
        p[out_rows] = m.predict(table.iloc[out_rows])

    full = (
        TriageModel(columns, kind, seed)
        .fit(table[train], y[train])
        .calibrate(table[calib], y[calib])
    )
    p[val] = full.predict(table[val])

    metrics = {
        "triage/val_auroc": auroc(y[held], p[held]),
        "triage/val_auprc": auprc(y[held], p[held]),
        "triage/val_calibration_error": calibration_error(y[held].astype(float), p[held]),
        "triage/train_oof_auroc": auroc(y[train], p[train]),
    }
    if "g1_entropy" in table:
        metrics["triage/val_entropy_only"] = auroc(y[held], table.loc[held, "g1_entropy"])
        metrics["triage/val_minority_share_only"] = auroc(y[held], -table.loc[held, "g1_share"])
    return TriageResult(p, full, metrics)


def ambiguity_recall(
    votes: pd.DataFrame, routes: pd.Series, num_classes: int, h0: float = 0.3
) -> float:
    """Share of truly ambiguous items (clean votes still split, entropy > h0) routed to soft.

    "Truly ambiguous" is judged on the votes that are not errors, using the same entropy
    threshold as the router.
    """
    from labeltriage.targets import entropy

    clean = votes[~votes["is_error"].to_numpy(dtype=bool)]
    counts = pd.crosstab(clean["item_id"], clean["label"]).reindex(
        columns=range(num_classes), fill_value=0
    )
    dist = counts.to_numpy() / np.maximum(counts.to_numpy().sum(axis=1, keepdims=True), 1)
    ambiguous = pd.Series(entropy(dist) > h0, index=counts.index)
    amb_ids = ambiguous.index[ambiguous]
    hits = routes.reindex(amb_ids) == "ambiguous"
    return float(hits.mean()) if len(amb_ids) else float("nan")
