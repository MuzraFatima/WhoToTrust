"""The shared training loop: minibatch SGD / AdamW on weighted soft targets."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from labeltriage.train.loss import weighted_soft_ce


@dataclass
class LoopConfig:
    epochs: int
    batch_size: int
    optimizer: str = "sgd"  # "sgd" or "adamw"
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    warmup_frac: float = 0.0
    amp: bool = True


def device_default() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _optimizer(model: torch.nn.Module, cfg: LoopConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise KeyError(cfg.optimizer)


def _schedule(opt, total: int, warmup: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up, then cosine decay to zero."""

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


Batch = tuple[object, torch.Tensor, torch.Tensor]


def fit(
    model: torch.nn.Module,
    n: int,
    get_batch: Callable[[np.ndarray, bool], Batch],
    forward: Callable[[torch.nn.Module, object], torch.Tensor],
    cfg: LoopConfig,
    rng: np.random.Generator,
    device: torch.device,
    on_epoch: Callable[[int, torch.nn.Module], None] | None = None,
) -> list[float]:
    """Train ``model`` for ``cfg.epochs`` epochs and return the mean loss of each epoch.

    ``get_batch(indices, train)`` returns ``(inputs, targets, weights)`` on ``device`` and
    ``forward(model, inputs)`` returns logits. ``on_epoch`` lets callers record training
    dynamics (Phase 3 uses it for data maps and AUM).
    """
    model.to(device)
    opt = _optimizer(model, cfg)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    sched = _schedule(opt, total, int(cfg.warmup_frac * total))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, cfg.batch_size):
            inputs, targets, weights = get_batch(order[start : start + cfg.batch_size], True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = forward(model, inputs)
            loss = weighted_soft_ce(logits.float(), targets, weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        if on_epoch is not None:
            on_epoch(epoch, model)
    return history


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    n: int,
    get_inputs: Callable[[np.ndarray], object],
    forward: Callable[[torch.nn.Module, object], torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Softmax probabilities for items ``0..n-1``, in order."""
    model.to(device).eval()
    out = []
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(n, start + batch_size))
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = forward(model, get_inputs(idx))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)
