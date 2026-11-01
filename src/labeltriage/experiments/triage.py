"""Phase 4: the triage gate on AmbiNoise, and the transfer test onto VariErr's real errors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from labeltriage.data import prepare
from labeltriage.eval.metrics import auprc, auroc
from labeltriage.eval.stats import cluster_bootstrap
from labeltriage.experiments.features import build_or_load, feature_columns, feature_inputs
from labeltriage.experiments.train_eval import load_bundle
from labeltriage.features import build as fb
from labeltriage.tasks import task
from labeltriage.triage.model import TriageModel
from labeltriage.triage.pipeline import score


@task("triage")
def triage(cfg: DictConfig) -> dict[str, Any]:
    """Gate: calibrated val AUROC >= 0.80 and calibration error < 0.05."""
    table = build_or_load(cfg, load_bundle(cfg).votes)
    cols = feature_columns(table)
    res = score(table, cols, kind=cfg.method.get("model", "lgbm"), seed=cfg.seed)
    m = res.metrics
    m["gate/auroc_pass"] = float(m["triage/val_auroc"] >= cfg.experiment.min_auroc)
    # Amendment A11: the bar never drops below what this many held-out votes can measure.
    bar = max(cfg.experiment.max_calibration_error, m["triage/val_calibration_noise_floor"])
    m["gate/calibration_bar"] = bar
    m["gate/calibration_pass"] = float(m["triage/val_calibration_error"] < bar)
    imp = res.model.importance()
    return {
        "metrics": m,
        "importance": {} if imp is None else imp.head(15).round(4).to_dict(),
    }


def training_table(cfg: DictConfig) -> pd.DataFrame:
    """Feature tables of every AmbiNoise-NLI draw the transfer model learns from, stacked."""
    parts = []
    for k in cfg.experiment.train_k:
        for etype in cfg.experiment.train_error_types:
            c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            c.data.noise.k = int(k)
            c.data.noise.error_type = str(etype)
            parts.append(build_or_load(c, load_bundle(c).votes).assign(draw=f"k{k}_{etype}"))
    return pd.concat(parts, ignore_index=True)


def pair_scores(votes: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Collapse vote-level scores to VariErr's primary unit, the (item, label) pair (A2)."""
    df = votes[["item_id", "label", "is_error"]].assign(p=p)
    return df.groupby(["item_id", "label"], as_index=False).agg(
        p=("p", "mean"), is_error=("is_error", "first")
    )


@task("transfer")
def transfer(cfg: DictConfig) -> dict[str, Any]:
    """Train on AmbiNoise-NLI, score VariErr without ever training or tuning on it."""
    table = training_table(cfg)
    cols = feature_columns(table)
    y = table["is_error"].to_numpy(dtype=bool)
    train = (table["split"] == "train").to_numpy()
    val = (table["split"] == "val").to_numpy()
    model = (
        TriageModel(cols, cfg.method.get("model", "lgbm"), cfg.seed)
        .fit(table[train], y[train])
        .calibrate(table[val], y[val])
    )

    v, _, _ = prepare.load_prepared("varierr", cfg.paths.data_dir)
    feats = fb.build(
        v,
        3,
        feature_inputs(cfg.paths.cache_dir, "varierr"),
        groups=tuple(cfg.experiment.get("groups", fb.GROUPS)),
        seed=cfg.seed,
        probe_epochs=cfg.experiment.get("probe_epochs", 15),
    )
    vt = pd.concat([v.reset_index(drop=True), feats], axis=1)
    pairs = pair_scores(vt, model.predict(vt[cols]))
    pairs["entropy"] = pair_scores(vt, vt["g1_entropy"].to_numpy())["p"]
    pairs["minority"] = pair_scores(vt, -vt["g1_share"].to_numpy())["p"]
    err = pairs["is_error"].to_numpy(dtype=bool)

    def diff(idx: np.ndarray) -> float:
        return auroc(err[idx], pairs["p"].to_numpy()[idx]) - auroc(
            err[idx], pairs["minority"].to_numpy()[idx]
        )

    d, lo, hi = cluster_bootstrap(
        diff, pairs["item_id"].to_numpy(), cfg.experiment.n_boot, cfg.seed
    )
    return {
        "metrics": {
            "transfer/auroc": auroc(err, pairs["p"]),
            "transfer/auprc": auprc(err, pairs["p"]),
            "transfer/entropy_only": auroc(err, pairs["entropy"]),
            "transfer/minority_share_only": auroc(err, pairs["minority"]),
            "transfer/gain_over_minority": d,
            "transfer/gain_over_minority_lo": lo,
            "transfer/gain_over_minority_hi": hi,
        },
        "n_pairs": len(pairs),
        "n_errors": int(err.sum()),
        "train_votes": int(train.sum()),
    }
