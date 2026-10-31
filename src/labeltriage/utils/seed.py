from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch, and return a fresh NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code still reads the legacy global state
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
