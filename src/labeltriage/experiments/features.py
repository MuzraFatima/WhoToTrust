"""Build and cache the feature table for one AmbiNoise draw, and report the Phase 3 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from labeltriage.experiments import embed
from labeltriage.experiments.train_eval import load_bundle
from labeltriage.features import build as fb
from labeltriage.features.evaluate import gate_report
from labeltriage.tasks import task


def feature_inputs(cache_dir: str | Path, base: str) -> fb.FeatureInputs:
    paths = embed.cache_paths(cache_dir, base)
    return fb.FeatureInputs(
        plausibility=embed.load(paths["plausibility"]),
        embeddings=embed.load(paths["embeddings"]),
    )


# Bumped whenever cached tables could be stale: v2 dropped g2_n_votes; v3 guarantees every cached
# table holds all five groups (v2 tables built by the transfer task could lack G3); v4 is the
# switch from the MNLI-only ChaosNLI mirror to the official SNLI + MNLI files.
FEATURE_VERSION = 4


def feature_cache_path(cfg: DictConfig) -> Path:
    root = Path(cfg.paths.cache_dir) / f"features_v{FEATURE_VERSION}"
    return root / cfg.data.name / f"seed{cfg.seed}.parquet"


def build_or_load(cfg: DictConfig, votes: pd.DataFrame) -> pd.DataFrame:
    """Votes with all five feature groups attached, cached per (dataset draw, seed).

    The cache always holds every group; callers pick columns afterwards. A cache that depended
    on which groups the first caller asked for would silently serve incomplete tables later.
    """
    path = feature_cache_path(cfg)
    if path.exists():
        return pd.read_parquet(path)
    base = cfg.data.get("base", cfg.data.name)
    feats = fb.build(
        votes,
        cfg.data.num_classes,
        feature_inputs(cfg.paths.cache_dir, base),
        groups=fb.GROUPS,
        seed=cfg.seed,
        probe_epochs=cfg.experiment.get("probe_epochs", 15),
    )
    table = pd.concat([votes.reset_index(drop=True), feats], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path)
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c[:3] in {f"{g}_" for g in fb.GROUPS}]


@task("features")
def features(cfg: DictConfig) -> dict[str, Any]:
    bundle = load_bundle(cfg)
    table = build_or_load(cfg, bundle.votes)
    groups = tuple(cfg.experiment.get("groups", fb.GROUPS))
    cols = [c for c in feature_columns(table) if c.split("_")[0] in groups]
    return {
        "metrics": gate_report(table[cols], table, seed=cfg.seed),
        "n_votes": len(table),
        "feature_path": str(feature_cache_path(cfg)),
        "columns": cols,
    }
