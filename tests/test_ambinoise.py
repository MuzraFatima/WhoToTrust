import numpy as np
import pandas as pd
import pytest

from labeltriage.ambinoise import errors
from labeltriage.ambinoise.injector import NoiseSpec, inject, native_error_mask


def _pool(n_items=300, votes=20, num_classes=10, seed=0):
    """Synthetic many-vote pool: each item has a main class and some genuine runner-up votes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        main, alt = rng.choice(num_classes, 2, replace=False)
        share = rng.uniform(0.5, 1.0)
        labels = np.where(rng.random(votes) < share, main, alt)
        rows += [(f"i{i:04d}", f"h{j}", int(c)) for j, c in enumerate(labels)]
    df = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label"])
    df.insert(0, "dataset", "toy")
    return df


def _ranking(pool, num_classes=10, seed=1):
    ids = np.sort(pool["item_id"].unique())
    p = np.random.default_rng(seed).dirichlet(np.ones(num_classes), size=len(ids))
    return pd.DataFrame(p, index=ids)


@pytest.mark.parametrize("kind", errors.ERROR_TYPES)
@pytest.mark.parametrize("rate", [0.1, 0.2, 0.4])
def test_injected_rate_is_exact_and_labels_change(kind, rate):
    pool = _pool()
    out = inject(pool, 10, NoiseSpec(k=5, rate=rate, error_type=kind), ranking=_ranking(pool))
    injected = out["error_type"] == kind
    assert injected.mean() == pytest.approx(rate, abs=0.01)
    assert (out.loc[injected, "label"] != out.loc[injected, "original_label"]).all()
    assert (out.loc[~injected, "label"] == out.loc[~injected, "original_label"]).all()
    assert out.groupby("item_id").size().eq(5).all()


def test_swap_only_touches_its_pairs():
    pool = _pool()
    out = inject(pool, 10, NoiseSpec(rate=0.2, error_type="swap"))
    hit = out[out["error_type"] == "swap"]
    m = errors.swap_map(10)
    assert (m[hit["original_label"].to_numpy()] == hit["label"].to_numpy()).all()


def test_lazy_always_answers_the_most_common_class():
    pool = _pool()
    mode = np.bincount(pool["label"]).argmax()
    out = inject(pool, 10, NoiseSpec(rate=0.2, error_type="lazy"))
    assert out.loc[out["error_type"] == "lazy", "label"].eq(mode).all()


def test_plausible_wrong_uses_the_injection_ranking():
    pool = _pool()
    ranking = _ranking(pool)
    out = inject(pool, 10, NoiseSpec(rate=0.2, error_type="plausible_wrong"), ranking=ranking)
    hit = out[out["error_type"] == "plausible_wrong"]
    scores = ranking.loc[hit["item_id"]].to_numpy().copy()
    scores[np.arange(len(hit)), hit["original_label"].to_numpy()] = -1
    assert (scores.argmax(1) == hit["label"].to_numpy()).all()


def test_plausible_wrong_without_ranking_fails_loudly():
    with pytest.raises(ValueError, match="A1"):
        inject(_pool(), 10, NoiseSpec(error_type="plausible_wrong"))


def test_errors_concentrate_in_high_propensity_annotators():
    out = inject(_pool(n_items=1000), 10, NoiseSpec(rate=0.2, error_type="spammer"))
    per_ann = out.groupby("annotator_id").agg(
        p=("annotator_propensity", "first"), err=("error_type", lambda s: (s == "spammer").mean())
    )
    assert per_ann["p"].corr(per_ann["err"], method="spearman") > 0.5


def test_annotators_never_repeat_within_an_item():
    out = inject(_pool(), 10, NoiseSpec(k=10))
    assert not out.duplicated(["item_id", "annotator_id"]).any()


def test_native_errors_follow_the_pool_cutoff():
    pool = pd.DataFrame(
        {
            "dataset": "toy",
            "item_id": ["a"] * 20,
            "annotator_id": [f"h{j}" for j in range(20)],
            "label": [0] * 19 + [1],
        }
    )
    mask = native_error_mask(pool, 3, 0.10)
    assert bool(mask[("a", 1)]) and not bool(mask[("a", 0)])
    out = inject(pool, 3, NoiseSpec(k=20, rate=0.0, num_annotators=30))
    native = out[out["error_type"] == "native"]
    assert native["label"].tolist() == [1]
    assert out["is_error"].sum() == 1


def test_same_seed_same_benchmark():
    pool = _pool()
    a = inject(pool, 10, NoiseSpec(seed=3))
    b = inject(pool, 10, NoiseSpec(seed=3))
    c = inject(pool, 10, NoiseSpec(seed=4))
    pd.testing.assert_frame_equal(a, b)
    assert not a["label"].equals(c["label"])


def test_split_column_is_carried_through():
    pool = _pool().assign(split="train")
    assert inject(pool, 10, NoiseSpec())["split"].eq("train").all()


def test_items_with_too_few_votes_are_rejected():
    with pytest.raises(ValueError, match="fewer than K"):
        inject(_pool(votes=3), 10, NoiseSpec(k=5))
