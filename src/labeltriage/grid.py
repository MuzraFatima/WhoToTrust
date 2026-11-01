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
import time
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
    """``root/stage/experiment/data/method/seed``. The stage is part of the path so a dev run
    (e.g. the VariErr dev-half re-measure) can never mark its final-stage twin as done."""
    method = cfg.method.name + (f"-{cfg.method.tag}" if cfg.method.get("tag") else "")
    return Path(root) / cfg.stage / cfg.experiment.name / cfg.data.name / method / f"seed{cfg.seed}"


def shard_points(points: list[list[str]], shard: tuple[int, int] | None) -> list[list[str]]:
    """Shard ``i`` of ``n`` (1-based) takes every n-th point, so shards are balanced and stable."""
    if shard is None:
        return points
    i, n = shard
    if not 1 <= i <= n:
        raise ValueError(f"shard must be i/n with 1 <= i <= n, got {i}/{n}")
    return points[i - 1 :: n]


def run_grid(
    spec: DictConfig | dict,
    root: str | Path,
    extra: list[str] | None = None,
    shard: tuple[int, int] | None = None,
    max_hours: float | None = None,
) -> dict:
    """Run every point of ``spec`` (or one shard of it) under ``root``.

    Finished runs are skipped, so a restarted session resumes where it stopped. With
    ``max_hours``, no new run starts once that much time has passed; the rest are counted as
    ``deferred`` and run on the next session. Returns counts of ran / skipped / failed / deferred.
    """
    counts = {"ran": 0, "skipped": 0, "failed": 0, "deferred": 0}
    start = time.monotonic()
    points = shard_points(list(expand(spec)), shard)
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        for k, overrides in enumerate(points):
            if max_hours is not None and time.monotonic() - start > max_hours * 3600:
                counts["deferred"] = len(points) - k
                log.warning(
                    "time budget of %.1f h used; %d runs deferred", max_hours, len(points) - k
                )
                break
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
