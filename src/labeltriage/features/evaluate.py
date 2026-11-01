"""Phase 3 gate: how well each feature group detects errors on its own, plus a leakage test.

A LightGBM detector is fitted on train-split votes and scored by AUROC on val-split votes. The
leakage test refits with the train labels shuffled, several times. Leakage can only push AUROC
above 0.5, so the test is one-sided (amendment A10): it fails when the mean shuffled AUROC is
significantly above 0.5. Means somewhat below 0.5 reflect how a noise-fitted model behaves in
sparse feature regions and do not indicate leakage.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from labeltriage.eval.metrics import auroc
from labeltriage.features.build import GROUPS, columns_of

DETECTOR_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


def detector_auroc(
    features: pd.DataFrame,
    is_error: np.ndarray,
    split: np.ndarray,
    columns: list[str],
    seed: int = 0,
    shuffle_labels: bool = False,
) -> float:
    train, val = split == "train", split == "val"
    y = np.asarray(is_error, dtype=int)
    y_fit = y[train]
    if shuffle_labels:
        y_fit = np.random.default_rng(seed).permutation(y_fit)
    model = lgb.LGBMClassifier(**DETECTOR_PARAMS, random_state=seed)
    model.fit(features.loc[train, columns], y_fit)
    return auroc(y[val], model.predict_proba(features.loc[val, columns])[:, 1])


def gate_report(
    features: pd.DataFrame,
    votes: pd.DataFrame,
    seed: int = 0,
    threshold: float = 0.60,
    n_shuffles: int = 10,
) -> dict[str, float]:
    """Per-group and all-group AUROC, the leakage AUROC, and pass flags."""
    is_error = votes["is_error"].to_numpy(dtype=bool)
    split = votes["split"].to_numpy()
    out: dict[str, float] = {}
    present = [g for g in GROUPS if columns_of(features, g)]
    for g in present:
        out[f"auroc/{g}"] = detector_auroc(features, is_error, split, columns_of(features, g), seed)
    all_cols = list(features.columns)
    out["auroc/all"] = detector_auroc(features, is_error, split, all_cols, seed)
    shuffled = np.array(
        [
            detector_auroc(features, is_error, split, all_cols, seed + r, shuffle_labels=True)
            for r in range(n_shuffles)
        ]
    )
    out["auroc/leakage_shuffled"] = float(shuffled.mean())
    out["auroc/leakage_shuffled_sd"] = float(shuffled.std(ddof=1)) if n_shuffles > 1 else 0.0
    # H1 is judged against item vote entropy alone; the minority-share rule is a second
    # untrained reference.
    val = split == "val"
    out["auroc/entropy_only"] = auroc(is_error[val], features.loc[val, "g1_entropy"])
    out["auroc/minority_share_only"] = auroc(is_error[val], -features.loc[val, "g1_share"])
    out["gate/groups_pass"] = float(all(out[f"auroc/{g}"] >= threshold for g in present))
    se = out["auroc/leakage_shuffled_sd"] / np.sqrt(n_shuffles)
    out["gate/leakage_pass"] = float(out["auroc/leakage_shuffled"] < 0.5 + max(0.02, 2 * se))
    return out
