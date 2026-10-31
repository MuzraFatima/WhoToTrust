"""Optional Weights & Biases logging. Everything here is a no-op unless ``tracking.wandb`` is on."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}/"))
        elif isinstance(v, int | float | bool | str) or v is None:
            out[key] = v
    return out


def log_result(cfg: DictConfig, result: dict[str, Any]) -> bool:
    """Send one finished run to W&B. Returns True if anything was logged."""
    if not cfg.tracking.wandb:
        return False
    import wandb

    run = wandb.init(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        name=f"{cfg.experiment.name}/{cfg.data.name}/{cfg.method.name}/seed{cfg.seed}",
        group=f"{cfg.experiment.name}/{cfg.data.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
    run.summary.update(_flatten({k: v for k, v in result.items() if k != "config"}))
    run.finish()
    return True
