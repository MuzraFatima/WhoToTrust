"""Sanity checks that AmbiNoise keeps real ambiguity intact (amendment A8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeltriage.ambinoise.injector import NoiseSpec, inject
from labeltriage.data import store


def clean_pool_distribution(pool: pd.DataFrame, num_classes: int, cutoff: float) -> pd.DataFrame:
    """Each item's full-pool distribution with native-error classes removed and renormalised."""
    vm = store.vote_matrix(pool, num_classes)
    p = vm.distribution.copy()
    p[p < cutoff] = 0.0
    p /= p.sum(axis=1, keepdims=True)
    return pd.DataFrame(p, index=vm.item_ids)


def sampling_kl(
    pool: pd.DataFrame,
    num_classes: int,
    k: int = 5,
    draws: int = 20,
    cutoff: float = 0.10,
    seed: int = 0,
    eps: float = 1e-3,
) -> pd.Series:
    """Per-item KL(clean pool || mean clean-vote distribution over ``draws`` AmbiNoise draws).

    A single draw of K=5 votes cannot match a 50-vote distribution, so the check is on the
    sampler: averaged over independent draws, the clean votes it hands out must reproduce each
    item's clean pool. Smoothing ``eps`` keeps the KL finite for classes never drawn.
    """
    ref = clean_pool_distribution(pool, num_classes, cutoff)
    counts = pd.DataFrame(0.0, index=ref.index, columns=range(num_classes))
    for d in range(draws):
        out = inject(
            pool,
            num_classes,
            NoiseSpec(k=k, rate=0.0, seed=seed + d, num_annotators=max(50, 2 * k)),
        )
        clean = out[out["error_type"] != "native"]
        counts = counts.add(
            pd.crosstab(clean["item_id"], clean["original_label"]).reindex(
                index=ref.index, columns=range(num_classes), fill_value=0
            ),
            fill_value=0,
        )
    est = (counts.to_numpy() + eps) / (counts.to_numpy() + eps).sum(axis=1, keepdims=True)
    p = ref.to_numpy()
    kl = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0) / est), 0.0).sum(axis=1)
    return pd.Series(kl, index=ref.index, name="kl")


def summarise(votes: pd.DataFrame) -> dict[str, float]:
    """Headline statistics for a dataset card."""
    per_type = votes["error_type"].value_counts(normalize=True).to_dict()
    return {
        "votes": len(votes),
        "items": int(votes["item_id"].nunique()),
        "annotators": int(votes["annotator_id"].nunique()),
        "error_share": float(votes["is_error"].mean()),
        **{f"share_{k}": float(v) for k, v in per_type.items()},
    }
