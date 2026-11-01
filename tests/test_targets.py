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


def test_align_reorders_and_rejects_missing():
    v = _votes([("b", "1", 1), ("a", "1", 0), ("c", "1", 2)])
    ts = targets.soft(v, 3).align(["c", "a"])
    assert ts.item_ids.tolist() == ["c", "a"]
    np.testing.assert_array_equal(ts.targets.argmax(1), [2, 0])
    with pytest.raises(KeyError, match="no target"):
        targets.soft(v, 3).align(["z"])


def test_router_sends_items_down_each_route():
    v = _votes(
        [("clear", u, 0) for u in "abcd"]
        + [("amb", "a", 0), ("amb", "b", 0), ("amb", "c", 1), ("amb", "d", 1)]
        + [("unrel", "a", 0), ("unrel", "b", 1)]
    )
    e = np.where(v["item_id"] == "unrel", 0.6, 0.0)
    ts = targets.route(v, 2, e)
    by = dict(zip(ts.item_ids, ts.routes, strict=True))
    assert by == {"amb": "ambiguous", "clear": "clear", "unrel": "unreliable"}
    t = dict(zip(ts.item_ids, ts.targets.tolist(), strict=True))
    assert t["clear"] == [1.0, 0.0]
    assert t["amb"] == [0.5, 0.5]
    w = dict(zip(ts.item_ids, ts.weights, strict=True))
    assert w["unrel"] == pytest.approx(0.8 / 2)


def test_oracle_routing_removes_known_errors():
    v = _votes([("x", "a", 2), ("x", "b", 2), ("x", "c", 2), ("x", "d", 0)])
    ts = targets.route(v, 3, np.array([0, 0, 0, 1]))
    assert ts.routes.tolist() == ["clear"]
    assert ts.targets.tolist() == [[0.0, 0.0, 1.0]]


def test_all_votes_dropped_gives_zero_weight_uniform_target():
    v = _votes([("x", "a", 0), ("x", "b", 1)])
    ts = targets.route(v, 2, np.array([0.95, 0.95]), hard_drop=0.8)
    assert ts.weights.tolist() == [0.0]
    assert ts.targets.tolist() == [[0.5, 0.5]]


def test_routes_survive_align():
    v = _votes([("b", "1", 1), ("a", "1", 0)])
    ts = targets.route(v, 2, np.zeros(2), nu=0.5).align(["b", "a"])
    assert ts.routes.tolist() == ["clear", "clear"]
