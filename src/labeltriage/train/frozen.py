"""A small MLP head on frozen embeddings: the mini profile's trainer, fast enough for a laptop CPU.

Inputs are encoded by a frozen model (DINOv2 for images, a sentence encoder for NLI pairs) and
cached row by row under ``<cache_dir>/frozen/`` by a hash of each row's content, so every image
or pair is encoded once and every later run trains in seconds. Only the head learns; it goes
through the shared loop, so soft targets, weights and Co-teaching behave as they do for the full
backbones.
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


def row_keys(domain: str, inputs: Any) -> list[str]:
    """One hash per input row: the image's bytes, or the premise / hypothesis text."""
    if domain == "image":
        arr = np.ascontiguousarray(inputs)
        return [hashlib.sha256(row.tobytes()).hexdigest()[:32] for row in arr]
    premises, hypotheses = inputs
    return [
        hashlib.sha256(f"{p}{q}".encode()).hexdigest()[:32]
        for p, q in zip(premises, hypotheses, strict=True)
    ]


def _rows(domain: str, inputs: Any, idx: list[int]) -> Any:
    if domain == "image":
        return np.ascontiguousarray(inputs)[idx]
    premises, hypotheses = inputs
    return [premises[i] for i in idx], [hypotheses[i] for i in idx]


@functools.lru_cache(maxsize=4)
def _encoder(domain: str, model_name: str):
    from labeltriage.experiments import embed

    if domain == "image":
        return embed.dinov2_embed(model_name, batch_size=32)
    run = embed.sentence_embed(model_name, batch_size=64)
    return lambda inputs: run(list(zip(*inputs, strict=True)))


def _load_store(directory: Path) -> dict[str, np.ndarray]:
    store: dict[str, np.ndarray] = {}
    for shard in sorted(directory.glob("shard-*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            store.update(zip(z["keys"].tolist(), z["feats"], strict=True))
    return store


def encode(domain: str, inputs: Any, model_name: str, cache_dir: str | Path) -> np.ndarray:
    """Frozen embeddings for ``inputs``. Each row is encoded once, ever: rows are cached by
    content, so overlapping sets (cross-fitting folds, a subset of earlier inputs) cost nothing."""
    directory = Path(cache_dir) / "frozen" / model_name.replace("/", "--")
    keys = row_keys(domain, inputs)
    store = _load_store(directory)
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k not in store:
            first.setdefault(k, i)
    if first:
        idx = list(first.values())
        feats = np.asarray(_encoder(domain, model_name)(_rows(domain, inputs, idx)), np.float32)
        new_keys = np.array(list(first), dtype="U32")
        directory.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256("".join(new_keys).encode()).hexdigest()[:16]
        tmp = directory / f"tmp-{name}.npz"
        np.savez(tmp, keys=new_keys, feats=feats)
        tmp.replace(directory / f"shard-{name}.npz")
        store.update(zip(new_keys.tolist(), feats, strict=True))
    return np.stack([store[k] for k in keys]).astype(np.float32)


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
