"""The shared training loop: minibatch SGD / AdamW on weighted soft targets."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from labeltriage.train.loss import soft_ce_per_item, weighted_soft_ce


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
    peer: torch.nn.Module | None = None,
    forget_rate: Callable[[int], float] | None = None,
) -> list[float]:
    """Train ``model`` for ``cfg.epochs`` epochs and return the mean loss of each epoch.

    ``get_batch(indices, train)`` returns ``(inputs, targets, weights)`` on ``device`` and
    ``forward(model, inputs)`` returns logits. ``on_epoch`` lets callers record training
    dynamics (Phase 3 uses it for data maps and AUM).

    With a ``peer`` the loop runs Co-teaching (Han et al., 2018): each network keeps the
    ``1 - forget_rate(epoch)`` share of the batch it finds easiest, and its peer trains on that.
    """
    models = [model] if peer is None else [model, peer]
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    use_amp = cfg.amp and device.type == "cuda"
    opts, scheds, scalers = [], [], []
    for m in models:
        m.to(device)
        opts.append(_optimizer(m, cfg))
        scheds.append(_schedule(opts[-1], total, int(cfg.warmup_frac * total)))
        scalers.append(torch.amp.GradScaler("cuda", enabled=use_amp))

    history = []
    for epoch in range(cfg.epochs):
        for m in models:
            m.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, cfg.batch_size):
            inputs, targets, weights = get_batch(order[start : start + cfg.batch_size], True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = [forward(m, inputs).float() for m in models]
            if peer is None:
                batch_losses = [weighted_soft_ce(logits[0], targets, weights)]
            else:
                keep = max(1, math.ceil((1 - forget_rate(epoch)) * len(targets)))
                with torch.no_grad():
                    easy = [torch.argsort(soft_ce_per_item(lg, targets))[:keep] for lg in logits]
                # Each network learns from the items its peer found easy.
                batch_losses = [
                    weighted_soft_ce(
                        logits[i][easy[1 - i]], targets[easy[1 - i]], weights[easy[1 - i]]
                    )
                    for i in range(2)
                ]
            for loss, opt, sched, scaler in zip(batch_losses, opts, scheds, scalers, strict=True):
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            losses.append(batch_losses[0].item())
        history.append(float(np.mean(losses)))
        if on_epoch is not None:
            on_epoch(epoch, model)
    return history


def coteaching_schedule(tau: float, ramp_epochs: int) -> Callable[[int], float]:
    """Forget rate rising linearly from 0 to ``tau`` over ``ramp_epochs`` epochs."""
    return lambda epoch: tau * min(1.0, epoch / max(1, ramp_epochs))


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
