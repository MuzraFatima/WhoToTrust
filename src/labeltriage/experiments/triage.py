"""Phase 4: the triage gate on AmbiNoise, and the transfer test onto VariErr's real errors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from labeltriage.data import prepare, splits
from labeltriage.eval.metrics import auprc, auroc
from labeltriage.eval.stats import cluster_bootstrap
from labeltriage.experiments import features as fx
from labeltriage.experiments.features import feature_columns, feature_inputs
from labeltriage.experiments.train_eval import load_bundle
from labeltriage.features import build as fb
from labeltriage.tasks import task
from labeltriage.triage.model import TriageModel
from labeltriage.triage.pipeline import score
from labeltriage.utils.runs import require_stage


@task("triage")
def triage(cfg: DictConfig) -> dict[str, Any]:
    """Gate: calibrated val AUROC >= 0.80 and calibration error < 0.05."""
    table = fx.build_or_load(cfg, load_bundle(cfg).votes)
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
            parts.append(fx.build_or_load(c, load_bundle(c).votes).assign(draw=f"k{k}_{etype}"))
    return pd.concat(parts, ignore_index=True)


def source_cfg(cfg: DictConfig, base: str, k: int, rate: float, etype: str) -> DictConfig:
    """A copy of ``cfg`` whose data section is an AmbiNoise draw over ``base``.

    Lets a run on a dataset without error labels (CIFAR-10N) build its triage model from
    AmbiNoise without composing a second Hydra config inside a running job.
    """
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    prefix = {"cifar10h": "ambinoise_10h", "chaosnli": "ambinoise_nli"}[base]
    c.data.base = base
    c.data.name = f"{prefix}_k{k}_r{rate}_{etype}"
    c.data.noise = {
        "k": int(k),
        "rate": float(rate),
        "error_type": etype,
        "num_annotators": 200,
        "concentration": 2.0,
        "native_cutoff": 0.10,
        "seed": int(cfg.seed),
        "ranking_path": f"{cfg.paths.cache_dir}/injection/{base}.parquet",
    }
    return c


def source_model(cfg: DictConfig, columns: list[str]) -> TriageModel:
    """Triage model trained on AmbiNoise draws named by ``method.transfer_source``."""
    src = cfg.method.transfer_source
    parts = []
    for k in src.k:
        for etype in src.error_types:
            c = source_cfg(cfg, src.base, k, src.rate, etype)
            parts.append(fx.build_or_load(c, load_bundle(c).votes))
    table = pd.concat(parts, ignore_index=True)
    y = table["is_error"].to_numpy(dtype=bool)
    train = (table["split"] == "train").to_numpy()
    val = (table["split"] == "val").to_numpy()
    return (
        TriageModel(columns, cfg.method.get("model", "lgbm"), cfg.seed)
        .fit(table[train], y[train])
        .calibrate(table[val], y[val])
    )


def pair_scores(votes: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Collapse vote-level scores to VariErr's primary unit, the (item, label) pair (A2)."""
    df = votes[["item_id", "label", "is_error"]].assign(p=p)
    return df.groupby(["item_id", "label"], as_index=False).agg(
        p=("p", "mean"), is_error=("is_error", "first")
    )


@task("transfer")
def transfer(cfg: DictConfig) -> dict[str, Any]:
    """Train on AmbiNoise-NLI and score VariErr, which is never trained on.

    Only the dev half informs design choices (A13). The test half is reported with stage=final.
    """
    table = training_table(cfg)
    groups = tuple(cfg.experiment.get("groups", fb.GROUPS))
    cols = [c for c in feature_columns(table) if c.split("_")[0] in groups]
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
        groups=groups,
        seed=cfg.seed,
        probe_epochs=cfg.experiment.get("probe_epochs", 15),
    )
    vt = pd.concat([v.reset_index(drop=True), feats], axis=1)
    pairs = pair_scores(vt, model.predict(vt[cols]))
    pairs["entropy"] = pair_scores(vt, vt["g1_entropy"].to_numpy())["p"]
    pairs["minority"] = pair_scores(vt, -vt["g1_share"].to_numpy())["p"]
    pairs["split"] = pairs["item_id"].map(splits.load_split("varierr"))

    # Amendment A13: the dev half may guide design; the test half opens only with stage=final.
    metrics: dict[str, float] = {}
    halves = {"dev": "val"} | ({"test": "test"} if cfg.stage == "final" else {})
    for name, split in halves.items():
        require_stage(cfg.stage, split)
        metrics |= transfer_metrics(
            pairs[pairs["split"] == split], f"transfer/{name}_", cfg.experiment.n_boot, cfg.seed
        )
    return {
        "metrics": metrics,
        "n_pairs": {k: int((pairs["split"] == v).sum()) for k, v in halves.items()},
        "train_votes": int(train.sum()),
    }


def transfer_metrics(pairs: pd.DataFrame, prefix: str, n_boot: int, seed: int) -> dict[str, float]:
    err = pairs["is_error"].to_numpy(dtype=bool)
    p, minority = pairs["p"].to_numpy(), pairs["minority"].to_numpy()

    def diff(idx: np.ndarray) -> float:
        return auroc(err[idx], p[idx]) - auroc(err[idx], minority[idx])

    d, lo, hi = cluster_bootstrap(diff, pairs["item_id"].to_numpy(), n_boot, seed)
    return {
        f"{prefix}auroc": auroc(err, p),
        f"{prefix}auprc": auprc(err, p),
        f"{prefix}entropy_only": auroc(err, pairs["entropy"]),
        f"{prefix}minority_share_only": auroc(err, minority),
        f"{prefix}gain_over_minority": d,
        f"{prefix}gain_over_minority_lo": lo,
        f"{prefix}gain_over_minority_hi": hi,
    }
