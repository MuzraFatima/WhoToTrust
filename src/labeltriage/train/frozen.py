"""A small MLP head on frozen embeddings: the mini profile's trainer, fast enough for a laptop CPU.

Inputs are encoded once by a frozen model (DINOv2 for images, a sentence encoder for NLI pairs)
and cached under ``<cache_dir>/frozen/`` by a hash of their content, so every later run on the
same inputs trains in seconds. Only the head learns; it goes through the shared loop, so soft
targets, weights and Co-teaching behave as they do for the full backbones.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from labeltriage.targets import TargetSet
from labeltriage.train import loop


class MLPHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def content_key(domain: str, inputs: Any) -> str:
    """A hash that changes whenever the inputs do: image bytes, or the text of every pair."""
    h = hashlib.sha256(domain.encode())
    if domain == "image":
        arr = np.ascontiguousarray(inputs)
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    else:
        premises, hypotheses = inputs
        for p, q in zip(premises, hypotheses, strict=True):
            h.update(f"{p}\x1f{q}\x1e".encode())
    return h.hexdigest()[:32]


@functools.lru_cache(maxsize=4)
def _encoder(domain: str, model_name: str):
    from labeltriage.experiments import embed

    if domain == "image":
        return embed.dinov2_embed(model_name, batch_size=32)
    run = embed.sentence_embed(model_name, batch_size=64)
    return lambda inputs: run(list(zip(*inputs, strict=True)))


def encode(domain: str, inputs: Any, model_name: str, cache_dir: str | Path) -> np.ndarray:
    """Frozen embeddings for ``inputs``, computed once and then read from the cache."""
    slug = model_name.replace("/", "--")
    path = Path(cache_dir) / "frozen" / slug / f"{content_key(domain, inputs)}.npy"
    if path.exists():
        return np.load(path, allow_pickle=False)
    feats = np.asarray(_encoder(domain, model_name)(inputs), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, feats, allow_pickle=False)
    tmp.replace(path)
    return feats


def _forward(model, inputs):
    return model(inputs)


def fit(
    feats: np.ndarray,
    ts: TargetSet,
    num_classes: int,
    cfg: loop.LoopConfig,
    seed: int,
    device: torch.device | None = None,
    hidden: int = 256,
    coteach: tuple[float, int] | None = None,
) -> tuple[nn.Module, list[float]]:
    """Train a head on ``feats`` aligned row by row with ``ts``; ``coteach`` as in the image
    trainer."""
    if len(feats) != len(ts):
        raise ValueError("features and targets must be aligned")
    device = device or loop.device_default()
    torch.manual_seed(seed)
    model = MLPHead(feats.shape[1], num_classes, hidden)
    x = torch.as_tensor(feats, dtype=torch.float32, device=device)
    targets = torch.as_tensor(ts.targets, dtype=torch.float32, device=device)
    weights = torch.as_tensor(ts.weights, dtype=torch.float32, device=device)

    def get_batch(idx, train):
        t = torch.as_tensor(idx, device=device)
        return x[t], targets[t], weights[t]

    peer, forget = None, None
    if coteach is not None:
        torch.manual_seed(seed + 7919)
        peer = MLPHead(feats.shape[1], num_classes, hidden)
        forget = loop.coteaching_schedule(*coteach)

    rng = np.random.default_rng(seed)
    history = loop.fit(model, len(x), get_batch, _forward, cfg, rng, device, None, peer, forget)
    return model, history


def predict(
    model: nn.Module, feats: np.ndarray, batch_size: int = 1024, device: torch.device | None = None
) -> np.ndarray:
    device = device or loop.device_default()
    x = torch.as_tensor(feats, dtype=torch.float32)

    def get_inputs(idx):
        return x[torch.as_tensor(idx)].to(device)

    return loop.predict(model, len(x), get_inputs, _forward, batch_size, device)
