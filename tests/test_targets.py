import numpy as np
import pandas as pd
import pytest

from labeltriage import targets


def _votes(rows):
    item, ann, lab = zip(*rows, strict=True)
    return pd.DataFrame({"dataset": "t", "item_id": item, "annotator_id": ann, "label": lab})


def test_majority_is_one_hot():
    v = _votes([("a", "1", 2), ("a", "2", 2), ("a", "3", 0), ("b", "1", 1)])
    ts = targets.majority(v, 3)
    np.testing.assert_array_equal(ts.targets, [[0, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(ts.weights, [1, 1])


def test_majority_breaks_ties_randomly():
    v = _votes([(f"i{k}", u, c) for k in range(400) for u, c in (("1", 0), ("2", 1))])
    winners = targets.majority(v, 2, np.random.default_rng(0)).targets.argmax(axis=1)
    assert 0.4 < winners.mean() < 0.6


def test_soft_is_vote_share():
    v = _votes([("a", "1", 0), ("a", "2", 0), ("a", "3", 1), ("a", "4", 2)])
    np.testing.assert_allclose(targets.soft(v, 3).targets, [[0.5, 0.25, 0.25]])


def test_targetset_validates_shapes_and_rows():
    with pytest.raises(ValueError, match="sum to 1"):
        targets.TargetSet(np.array(["a"]), np.array([[0.5, 0.2]]), np.ones(1))
    with pytest.raises(ValueError, match="matching"):
        targets.TargetSet(np.array(["a", "b"]), np.array([[1.0, 0.0]]), np.ones(2))


def test_build_dispatch():
    v = _votes([("a", "1", 0)])
    assert targets.build("soft", v, 2).targets.tolist() == [[1.0, 0.0]]
    with pytest.raises(KeyError):
        targets.build("nope", v, 2)
