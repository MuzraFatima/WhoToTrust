"""Image arrays for the CIFAR family, as ``uint8`` NHWC arrays plus integer labels.

CIFAR-10 itself comes through torchvision (which checks its own MD5). CIFAR-10.1 and CIFAR-10-C
ship as plain ``.npy`` files and are loaded with ``allow_pickle=False``.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from labeltriage.data import download

CIFAR10_C_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


def cifar10(data_dir: str | Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 train (50,000) or test (10,000) images with their original labels."""
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=Path(data_dir) / "raw" / "cifar10", train=train, download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def cifar10_1(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10.1 v6: 2,000 new test images drawn to match CIFAR-10's distribution."""
    x = np.load(download.fetch("cifar10_1_data", data_dir), allow_pickle=False)
    y = np.load(download.fetch("cifar10_1_labels", data_dir), allow_pickle=False)
    return x.astype(np.uint8), y.astype(np.int64)


def _cifar10_c_dir(data_dir: str | Path) -> Path:
    out = Path(data_dir) / "raw" / "cifar10_c" / "CIFAR-10-C"
    if not (out / "labels.npy").exists():
        tar_path = download.fetch("cifar10_c", data_dir)
        with tarfile.open(tar_path) as tar:
            tar.extractall(out.parent, filter="data")
    return out


def cifar10_c(
    data_dir: str | Path, corruption: str, severity: int
) -> tuple[np.ndarray, np.ndarray]:
    """One corruption at one severity (1–5) of CIFAR-10-C: 10,000 images."""
    if corruption not in CIFAR10_C_CORRUPTIONS:
        raise KeyError(f"unknown corruption {corruption!r}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be in 1..5")
    d = _cifar10_c_dir(data_dir)
    lo, hi = (severity - 1) * 10_000, severity * 10_000
    x = np.load(d / f"{corruption}.npy", mmap_mode="r", allow_pickle=False)[lo:hi]
    y = np.load(d / "labels.npy", allow_pickle=False)[lo:hi]
    return np.ascontiguousarray(x, dtype=np.uint8), y.astype(np.int64)
