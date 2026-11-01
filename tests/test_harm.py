import numpy as np
import pandas as pd
import pytest

from labeltriage.harm import COUNTERFACTUALS, harm_targets


def _votes():
    # Item a: genuine 3/2 split between 0 and 1, plus one error vote for 2.
    # Item b: every vote is an error.
    return pd.DataFrame(
        {
            "item_id": ["a"] * 6 + ["b", "b"],
            "label": [0, 0, 0, 1, 1, 2, 1, 1],
            "is_error": [False] * 5 + [True, True, True],
        }
    )


@pytest.mark.parametrize(
    ("which", "expected_a"),
    [
        ("neither", [3 / 6, 2 / 6, 1 / 6]),
        ("errors_removed", [3 / 5, 2 / 5, 0.0]),
        ("ambiguity_collapsed", [5 / 6, 0.0, 1 / 6]),
        ("both", [1.0, 0.0, 0.0]),
    ],
)
def test_counterfactual_targets(which, expected_a):
    ts = harm_targets(_votes(), 3, which)
    assert ts.item_ids.tolist() == ["a", "b"]
    np.testing.assert_allclose(ts.targets[0], expected_a)


def test_all_error_items_fall_back_to_observed_votes():
    for which in COUNTERFACTUALS:
        np.testing.assert_allclose(harm_targets(_votes(), 3, which).targets[1], [0, 1, 0])


def test_unknown_counterfactual():
    with pytest.raises(KeyError):
        harm_targets(_votes(), 3, "nope")
