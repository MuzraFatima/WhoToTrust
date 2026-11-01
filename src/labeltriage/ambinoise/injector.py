"""Build an AmbiNoise vote table from a many-annotator dataset.

For every item, K real votes are drawn without replacement from its full human pool, so the
ambiguity is real. Each drawn vote is credited to a simulated annotator, and annotators have their
own error propensities. Exactly ``round(rate * N)`` of the eligible votes are then corrupted by
one error process, chosen in proportion to their annotators' propensities. Real votes for a class
that got under ``native_cutoff`` of the item's full pool are labelled native errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from labeltriage.ambinoise import errors
from labeltriage.data import store


@dataclass(frozen=True)
class NoiseSpec:
    k: int = 5
    rate: float = 0.2
    error_type: str = "spammer"
    num_annotators: int = 200
    # Beta concentration for annotator error propensities: lower means more heterogeneous.
    concentration: float = 2.0
    native_cutoff: float = 0.10
    seed: int = 0


def native_error_mask(pool: pd.DataFrame, num_classes: int, cutoff: float) -> pd.Series:
    """Per (item, label): True when the label got under ``cutoff`` of the item's full pool."""
    vm = store.vote_matrix(pool, num_classes)
    share = pd.DataFrame(vm.distribution, index=vm.item_ids)
    return share.stack().rename("share").lt(cutoff)


def _annotator_propensity(spec: NoiseSpec, rng: np.random.Generator) -> np.ndarray:
    mean = min(max(spec.rate, 1e-3), 0.999)
    a = mean * spec.concentration
    b = (1 - mean) * spec.concentration
    return rng.beta(a, b, size=spec.num_annotators)


def _gumbel_top_k(weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``k`` draws without replacement, with probability proportional to ``weights``."""
    keys = np.log(np.maximum(weights, 1e-12)) + rng.gumbel(size=len(weights))
    return np.argpartition(-keys, k - 1)[:k] if k > 0 else np.array([], dtype=int)


def inject(
    pool: pd.DataFrame,
    num_classes: int,
    spec: NoiseSpec,
    ranking: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the AmbiNoise vote table for ``pool``.

    ``pool`` is a full annotation-store table (every human vote per item). ``ranking`` is the
    injection model's per-item class probabilities, indexed by ``item_id``, required for the
    plausible-wrong error type. Extra columns in the output:

    - ``original_label``: the real human vote before any corruption;
    - ``is_error`` / ``error_type``: injected errors carry their process name, native errors
      ``"native"``, and everything else (variation) ``"none"``;
    - ``annotator_propensity``: the simulated annotator's error propensity (for analysis only;
      never a feature).
    """
    if spec.error_type not in errors.ERROR_TYPES:
        raise KeyError(f"unknown error type {spec.error_type!r}")
    rng = np.random.default_rng(spec.seed)

    # 1. Draw K real votes per item, without replacement.
    shuffled = pool.sample(frac=1.0, random_state=rng.integers(2**31))
    drawn = (
        shuffled.groupby("item_id", sort=True).head(spec.k).sort_values("item_id", kind="stable")
    )
    short = drawn.groupby("item_id").size()
    if (short < spec.k).any():
        raise ValueError(f"{int((short < spec.k).sum())} items have fewer than K={spec.k} votes")
    drawn = drawn.reset_index(drop=True)
    n = len(drawn)

    # 2. Credit votes to simulated annotators; an annotator labels an item at most once.
    propensity = _annotator_propensity(spec, rng)
    n_items = n // spec.k
    ann = np.stack(
        [rng.choice(spec.num_annotators, size=spec.k, replace=False) for _ in range(n_items)]
    ).ravel()

    # 3. Corrupt exactly round(rate * n) eligible votes, weighted by annotator propensity.
    labels = drawn["label"].to_numpy()
    ctx = errors.ErrorContext(
        num_classes=num_classes,
        lazy_class=int(np.bincount(pool["label"], minlength=num_classes).argmax()),
        ranking=None if ranking is None else ranking.to_numpy(),
    )
    item_ids = drawn["item_id"].to_numpy()
    item_rows = None
    if ranking is not None:
        missing = pd.Index(np.unique(item_ids)).difference(ranking.index)
        if len(missing):
            raise KeyError(f"ranking is missing {len(missing)} items")
        item_rows = ranking.index.get_indexer(item_ids)

    ok = np.flatnonzero(errors.eligible(spec.error_type, labels, ctx))
    target = int(round(spec.rate * n))
    picked = ok[_gumbel_top_k(propensity[ann[ok]], min(target, len(ok)), rng)]

    new_labels = labels.copy()
    new_labels[picked] = errors.corrupt(
        spec.error_type,
        labels[picked],
        None if item_rows is None else item_rows[picked],
        ctx,
        rng,
    )

    # 4. Native errors: real votes for a class the full pool barely supports.
    native = native_error_mask(pool, num_classes, spec.native_cutoff)
    is_native = native.reindex(pd.MultiIndex.from_arrays([item_ids, labels])).to_numpy(bool)

    injected = np.zeros(n, dtype=bool)
    injected[picked] = True
    error_type = np.where(injected, spec.error_type, np.where(is_native, "native", "none"))

    out = pd.DataFrame(
        {
            "dataset": f"ambinoise-{pool['dataset'].iloc[0]}",
            "item_id": item_ids,
            "annotator_id": np.char.add("s", ann.astype(str)),
            "label": new_labels,
            "original_label": labels,
            "is_error": injected | is_native,
            "error_type": error_type,
            "annotator_propensity": propensity[ann],
        }
    )
    if "split" in drawn:
        out.insert(4, "split", drawn["split"].to_numpy())
    # Two corrupted votes from different annotators can coincide; that is fine. The same
    # annotator never votes twice on an item, so the store's uniqueness rule holds.
    return store.validate(out, num_classes=num_classes)


def realised_rate(votes: pd.DataFrame, error_type: str) -> float:
    return float((votes["error_type"] == error_type).mean())
