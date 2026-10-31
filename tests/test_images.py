import numpy as np
import pytest

from labeltriage.data import images


def _fake_cifar10_c(tmp_path):
    d = tmp_path / "raw" / "cifar10_c" / "CIFAR-10-C"
    d.mkdir(parents=True)
    labels = np.tile(np.arange(10), 5000).astype(np.uint8)
    np.save(d / "labels.npy", labels)
    x = np.zeros((50_000, 2, 2, 3), dtype=np.uint8)
    for s in range(5):
        x[s * 10_000 : (s + 1) * 10_000] = s + 1
    np.save(d / "fog.npy", x)


def test_cifar10_c_slices_by_severity(tmp_path):
    _fake_cifar10_c(tmp_path)
    x, y = images.cifar10_c(tmp_path, "fog", 3)
    assert x.shape == (10_000, 2, 2, 3)
    assert (x == 3).all()
    assert y[:3].tolist() == [0, 1, 2]


def test_cifar10_c_rejects_bad_arguments(tmp_path):
    with pytest.raises(KeyError):
        images.cifar10_c(tmp_path, "rain", 1)
    with pytest.raises(ValueError):
        images.cifar10_c(tmp_path, "fog", 6)
