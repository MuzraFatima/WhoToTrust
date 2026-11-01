"""Expand a sweep spec into individual runs and execute the ones not yet finished.

A sweep spec is a YAML mapping of config keys to lists of values, plus optional fixed overrides::

    name: smoke_grid
    fixed: [experiment=smoke]
    axes:
      method: [majority_vote, soft_labels]
      seed: [0, 1, 2]
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from pathlib import Path

from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from labeltriage import run

log = logging.getLogger(__name__)


def expand(spec: DictConfig | dict) -> Iterator[list[str]]:
    """Yield one override list per point in the Cartesian product of ``spec.axes``."""
    spec = OmegaConf.create(spec) if isinstance(spec, dict) else spec
    fixed = list(spec.get("fixed", []))
    axes = spec.get("axes", {})
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        yield fixed + [f"{k}={v}" for k, v in zip(keys, values, strict=True)]


def run_dir_for(cfg: DictConfig, root: str | Path) -> Path:
    method = cfg.method.name + (f"-{cfg.method.tag}" if cfg.method.get("tag") else "")
    return Path(root) / cfg.experiment.name / cfg.data.name / method / f"seed{cfg.seed}"


def run_grid(spec: DictConfig | dict, root: str | Path, extra: list[str] | None = None) -> dict:
    """Run every point of ``spec`` under ``root``. Returns counts of ran / skipped / failed."""
    counts = {"ran": 0, "skipped": 0, "failed": 0}
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        for overrides in expand(spec):
            cfg = compose(config_name="config", overrides=overrides + list(extra or []))
            target = run_dir_for(cfg, root)
            try:
                result = run.execute(cfg, target)
            except Exception:
                log.exception("run failed: %s", overrides)
                counts["failed"] += 1
                continue
            counts["ran" if result is not None else "skipped"] += 1
    return counts
