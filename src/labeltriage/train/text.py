"""Training and prediction for NLI premise / hypothesis pairs with a Hugging Face encoder."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from labeltriage.targets import TargetSet
from labeltriage.train import loop


def _default_loop() -> loop.LoopConfig:
    return loop.LoopConfig(
        epochs=4, batch_size=32, optimizer="adamw", lr=2e-5, weight_decay=0.01, warmup_frac=0.1
    )


@dataclass
class TextConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 128
    loop: loop.LoopConfig = field(default_factory=_default_loop)
    eval_batch_size: int = 128


Encodings = dict[str, torch.Tensor]


def tokenize(premises, hypotheses, tokenizer, max_length: int) -> Encodings:
    enc = tokenizer(
        list(premises),
        list(hypotheses),
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}


def _forward(model, inputs):
    return model(**inputs).logits


def _slice(enc: Encodings, idx, device) -> Encodings:
    t = torch.as_tensor(idx)
    return {k: v[t].to(device) for k, v in enc.items()}


def _length(enc: Encodings) -> int:
    return len(next(iter(enc.values())))


def fit(
    model: torch.nn.Module,
    encodings: Encodings,
    ts: TargetSet,
    cfg: TextConfig,
    seed: int,
    device: torch.device | None = None,
    on_epoch=None,
    peer: torch.nn.Module | None = None,
    coteach: tuple[float, int] | None = None,
) -> tuple[torch.nn.Module, list[float]]:
    """Fine-tune ``model``; with ``peer`` and ``coteach=(tau, ramp_epochs)``, via Co-teaching."""
    n = _length(encodings)
    if n != len(ts):
        raise ValueError("encodings and targets must be aligned")
    device = device or loop.device_default()
    torch.manual_seed(seed)
    targets = torch.as_tensor(ts.targets, dtype=torch.float32, device=device)
    weights = torch.as_tensor(ts.weights, dtype=torch.float32, device=device)

    def get_batch(idx, train):
        return _slice(encodings, idx, device), targets[idx], weights[idx]

    forget = loop.coteaching_schedule(*coteach) if coteach is not None else None
    rng = np.random.default_rng(seed)
    history = loop.fit(model, n, get_batch, _forward, cfg.loop, rng, device, on_epoch, peer, forget)
    return model, history


def predict(
    model: torch.nn.Module,
    encodings: Encodings,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or loop.device_default()

    def get_inputs(idx):
        return _slice(encodings, idx, device)

    return loop.predict(model, _length(encodings), get_inputs, _forward, batch_size, device)
