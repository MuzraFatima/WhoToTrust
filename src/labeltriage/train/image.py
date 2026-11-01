"""Training and prediction for CIFAR images held as uint8 NHWC arrays.

Augmentation (random crop with 4-pixel reflect padding, horizontal flip) runs on whole batches on
the device, which keeps a T4 busy without a DataLoader.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from labeltriage.targets import TargetSet
from labeltriage.train import loop
from labeltriage.train.models import FrozenDinoHead, ResNet18

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _default_loop() -> loop.LoopConfig:
    return loop.LoopConfig(epochs=60, batch_size=128, lr=0.1)


@dataclass
class ImageConfig:
    loop: loop.LoopConfig = field(default_factory=_default_loop)
    width: int = 64
    augment: bool = True
    backbone: str = "resnet18"  # or "dinov2_head" (frozen DINOv2 + MLP head, an ablation)
    eval_batch_size: int = 512


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop from a 4-pixel reflect-padded image, then a random horizontal flip."""
    n, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dy = torch.randint(0, 9, (n,), generator=generator).tolist()
    dx = torch.randint(0, 9, (n,), generator=generator).tolist()
    out = torch.stack(
        [padded[i, :, y : y + h, z : z + w] for i, (y, z) in enumerate(zip(dy, dx, strict=True))]
    )
    flip = (torch.rand(n, generator=generator) < 0.5).to(x.device)
    return torch.where(flip.view(n, 1, 1, 1), out.flip(3), out)


def make_model(cfg: ImageConfig, num_classes: int) -> torch.nn.Module:
    if cfg.backbone == "resnet18":
        return ResNet18(num_classes, cfg.width)
    if cfg.backbone == "dinov2_head":
        return FrozenDinoHead(num_classes)
    raise KeyError(f"unknown image backbone {cfg.backbone!r}")


def _forward(model, inputs):
    return model(inputs)


def fit(
    images: np.ndarray,
    ts: TargetSet,
    num_classes: int,
    cfg: ImageConfig,
    seed: int,
    device: torch.device | None = None,
    on_epoch=None,
    coteach: tuple[float, int] | None = None,
) -> tuple[torch.nn.Module, list[float]]:
    """Train a ResNet-18 on ``images`` aligned row by row with ``ts``.

    ``coteach=(tau, ramp_epochs)`` trains a second, differently initialised ResNet-18 alongside
    it with Co-teaching, and returns the first network.
    """
    if len(images) != len(ts):
        raise ValueError("images and targets must be aligned")
    device = device or loop.device_default()
    torch.manual_seed(seed)
    model = make_model(cfg, num_classes)
    targets = torch.as_tensor(ts.targets, dtype=torch.float32, device=device)
    weights = torch.as_tensor(ts.weights, dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)

    def get_batch(idx, train):
        x = to_tensor(images[idx], device)
        if train and cfg.augment:
            x = augment(x, gen)
        return x, targets[idx], weights[idx]

    peer, forget = None, None
    if coteach is not None:
        torch.manual_seed(seed + 7919)
        peer = make_model(cfg, num_classes)
        forget = loop.coteaching_schedule(*coteach)

    rng = np.random.default_rng(seed)
    history = loop.fit(
        model, len(images), get_batch, _forward, cfg.loop, rng, device, on_epoch, peer, forget
    )
    return model, history


def predict(
    model: torch.nn.Module,
    images: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or loop.device_default()

    def get_inputs(idx):
        return to_tensor(images[idx], device)

    return loop.predict(model, len(images), get_inputs, _forward, batch_size, device)
