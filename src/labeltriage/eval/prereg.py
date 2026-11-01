"""Pilot analysis for the pre-registration: seed spread, minimum detectable effects, gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeltriage.eval.stats import minimum_detectable_effect


def pooled_seed_sd(runs: pd.DataFrame, metric: str, by=("data", "method")) -> float:
    """Pooled within-group standard deviation of ``metric`` across seeds."""
    groups = runs.groupby(list(by))[metric]
    var = groups.var(ddof=1)
    dof = groups.count() - 1
    keep = dof > 0
    if not keep.any():
        return float("nan")
    return float(np.sqrt((var[keep] * dof[keep]).sum() / dof[keep].sum()))


def mde_report(runs: pd.DataFrame, hypotheses: dict) -> dict[str, float]:
    """Compare each pre-registered effect size with what the planned seeds can detect.

    Accuracy-type metrics are in [0, 1] in results, so H2's point thresholds are divided by 100.
    """
    n_head = hypotheses["seeds"]["headline"]
    out: dict[str, float] = {}

    sd_acc = pooled_seed_sd(runs, "clean_dev/acc")
    out["H2/seed_sd"] = sd_acc
    out["H2/mde"] = minimum_detectable_effect(sd_acc, n_head)
    out["H2/threshold"] = hypotheses["H2"]["gain_over_best_global_min_points"] / 100
    out["H2/detectable"] = float(out["H2/threshold"] > out["H2/mde"])

    if "clean_dev/ece" in runs:
        sd = pooled_seed_sd(runs, "clean_dev/ece")
        rel = 0.20 * runs["clean_dev/ece"].mean()
        out["H3/seed_sd"], out["H3/mde"], out["H3/threshold"] = (
            sd,
            minimum_detectable_effect(sd, n_head),
            rel,
        )
        out["H3/detectable"] = float(rel > out["H3/mde"])

    if "val/kl" in runs:
        sd = pooled_seed_sd(runs, "val/kl")
        mv = runs.loc[runs["method"] == "majority_vote", "val/kl"].mean()
        rel = 0.30 * mv
        out["H4/seed_sd"], out["H4/mde"], out["H4/threshold"] = (
            sd,
            minimum_detectable_effect(sd, n_head),
            rel,
        )
        out["H4/detectable"] = float(rel > out["H4/mde"])

    if "triage/val_auroc" in runs:
        sd = pooled_seed_sd(runs.dropna(subset=["triage/val_auroc"]), "triage/val_auroc")
        out["H1/seed_sd"] = sd
        out["H1/mde"] = minimum_detectable_effect(sd, n_head)
        out["H1/threshold"] = hypotheses["H1"]["gain_over_entropy_min"]
        out["H1/detectable"] = float(out["H1/threshold"] > out["H1/mde"])

    flags = [v for k, v in out.items() if k.endswith("/detectable")]
    out["gate/all_detectable"] = float(bool(flags) and all(flags))
    return out
