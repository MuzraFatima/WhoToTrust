import numpy as np
import pandas as pd
import pytest

from labeltriage import baselines

pytest.importorskip("cleanlab")


def _crowd(n=300, seed=0):
    """Items near class centres; annotators u0-u7 mostly right, u8-u9 random."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n)
    ids = np.array([f"i{i:04d}" for i in range(n)])
    rows = []
    for i, t in zip(ids, truth, strict=True):
        for a in rng.choice(10, 5, replace=False):
            bad = a >= 8
            lab = rng.integers(0, 3) if bad else (t if rng.random() < 0.9 else rng.integers(0, 3))
            rows.append((i, f"u{a}", int(lab)))
    votes = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label"])
    votes.insert(0, "dataset", "t")
    centres = rng.standard_normal((3, 6)) * 3
    emb = pd.DataFrame(centres[truth] + rng.standard_normal((n, 6)), index=ids)
    return votes, truth, emb


def test_dawid_skene_targets_are_one_hot_and_accurate():
    votes, truth, _ = _crowd()
    ts = baselines.dawid_skene_targets(votes, 3)
    assert set(np.unique(ts.targets)) <= {0.0, 1.0}
    assert (ts.targets.argmax(1) == truth).mean() > 0.95


def test_entropy_weight_downweights_split_items():
    votes = pd.DataFrame(
        {
            "dataset": "t",
            "item_id": ["a", "a", "b", "b"],
            "annotator_id": ["u", "v", "u", "v"],
            "label": [0, 0, 0, 1],
        }
    )
    ts = baselines.entropy_weight_targets(votes, 2, np.random.default_rng(0))
    assert ts.weights.tolist() == pytest.approx([1.0, 0.0])


def test_oof_probs_are_valid_and_informative():
    votes, truth, emb = _crowd()
    ids = np.sort(votes["item_id"].unique())
    p = baselines.oof_pred_probs(ids, emb, truth, 3)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert (p.argmax(1) == truth).mean() > 0.9


def test_confident_learning_and_crowdlab_prune_some_items():
    votes, truth, emb = _crowd()
    ids = np.sort(votes["item_id"].unique())
    p = baselines.oof_pred_probs(ids, emb, truth, 3)
    rng = np.random.default_rng(0)
    cl = baselines.confident_learning_targets(votes, 3, p, rng)
    cr = baselines.crowdlab_targets(votes, 3, p)
    for ts in (cl, cr):
        assert list(ts.item_ids) == list(ids)
        assert 0.5 < ts.weights.mean() <= 1.0
    assert (cr.targets.argmax(1) == truth).mean() > 0.95
