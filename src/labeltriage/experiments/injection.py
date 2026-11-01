"""The plausible-wrong injection model (amendment A1).

A classifier is cross-fitted on majority labels over every item of the base dataset, and each item
gets the class probabilities from the fold model that never saw it. AmbiNoise's plausible-wrong
errors pick the highest-ranked class other than the true vote. Seeds are disjoint from every
other run, and the images use a ResNet-18 while G3 uses CLIP, so the injector and the triage
features do not share a model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from labeltriage import crossfit, targets
from labeltriage.data import bundles, download, prepare
from labeltriage.eval import metrics
from labeltriage.experiments.train_eval import TRAINERS
from labeltriage.tasks import task


def ranking_path(cfg: DictConfig, base: str) -> Path:
    return Path(cfg.paths.cache_dir) / "injection" / f"{base}.parquet"


@task("injection_model")
def injection_model(cfg: DictConfig) -> dict[str, Any]:
    base = cfg.data.get("base", cfg.data.name)
    votes, items, meta = prepare.load_prepared(base, cfg.paths.data_dir)
    k = len(meta["classes"])
    bundle = bundles.load(base, cfg.paths.data_dir)
    inj = cfg.experiment.injection

    ts = targets.majority(votes, k, np.random.default_rng(inj.seed))
    ids = ts.item_ids

    fold_cfg = OmegaConf.to_container(cfg, resolve=True)
    if bundle.domain == "text":
        fold_cfg["data"]["train"]["model_name"] = inj.text_model

    def fit_predict(fold: int, train: np.ndarray, held: np.ndarray) -> np.ndarray:
        c = OmegaConf.create({**fold_cfg, "seed": int(inj.seed) + fold})
        sub = dataclasses.replace(
            bundle,
            train_inputs=bundle.inputs_for(ids[train]),
            train_item_ids=ids[train],
            train_votes=votes[votes["item_id"].isin(set(ids[train]))],
        )
        predict, _, _ = TRAINERS[bundle.domain](c, sub, ts.subset(train))
        return predict(bundle.inputs_for(ids[held]))

    probs = crossfit.cross_predict(ids, fit_predict, n_folds=inj.folds, seed=inj.seed)

    out = ranking_path(cfg, base)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(probs, index=pd.Index(ids, name="item_id"))
    frame.columns = frame.columns.astype(str)
    frame.to_parquet(out)

    majority = ts.targets.argmax(axis=1)
    return {
        "metrics": {
            "heldout_acc_vs_majority": metrics.accuracy(probs, majority),
            "heldout_ece": metrics.ece(probs, majority),
        },
        "ranking_path": str(out),
        "ranking_sha256": download.file_digest(out),
        "n_items": len(ids),
    }
