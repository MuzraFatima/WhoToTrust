"""Single entry point: ``python -m labeltriage.run experiment=... data=... method=... seed=...``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import labeltriage.experiments  # noqa: F401  (registers experiment tasks)
from labeltriage.tasks import TASKS, task
from labeltriage.utils import provenance, runs, tracking
from labeltriage.utils.seed import seed_everything

log = logging.getLogger(__name__)


@task("smoke")
def _smoke(cfg: DictConfig) -> dict[str, Any]:
    """End-to-end plumbing check: draw toy votes and report their agreement."""
    rng = seed_everything(cfg.seed)
    votes = rng.integers(
        0, cfg.data.num_classes, size=(cfg.data.num_items, cfg.data.votes_per_item)
    )
    agree = (votes == votes[:, :1]).all(axis=1).mean()
    return {"items": int(cfg.data.num_items), "full_agreement": float(agree)}


def execute(cfg: DictConfig, run_dir: str | Path) -> dict[str, Any] | None:
    """Run ``cfg`` into ``run_dir`` unless it already finished. Returns the result, or None."""
    run_dir = Path(run_dir)
    if runs.is_done(run_dir):
        log.info("skipping finished run %s", run_dir)
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    result = TASKS[cfg.experiment.task](cfg)
    # Tasks may hand back tables (e.g. scored votes for the explorer); they are saved beside the
    # result rather than inlined into result.json.
    for name, table in result.pop("_artifacts", {}).items():
        table.to_parquet(run_dir / f"{name}.parquet", index=False)
    result["config"] = OmegaConf.to_container(cfg, resolve=True)
    result["provenance"] = provenance.current()
    runs.write_result(run_dir, result)
    tracking.log_result(cfg, result)
    return result


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    result = execute(cfg, HydraConfig.get().runtime.output_dir)
    if result is not None:
        log.info("finished: %s", {k: v for k, v in result.items() if k != "config"})


if __name__ == "__main__":
    main()
