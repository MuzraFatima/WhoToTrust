import numpy as np
import pandas as pd
import pytest

from labeltriage.data import store


def _votes():
    return pd.DataFrame(
        {
            "dataset": ["toy"] * 6,
            "item_id": ["a", "a", "a", "b", "b", "b"],
            "annotator_id": ["u1", "u2", "u3", "u1", "u2", "u3"],
            "label": [0, 0, 1, 2, 2, 2],
        }
    )


def test_validate_accepts_minimal_table():
    out = store.validate(_votes(), num_classes=3)
    assert str(out["label"].dtype) == "int64"
    assert len(out) == 6


def test_validate_rejects_missing_column():
    with pytest.raises(store.SchemaError, match="missing"):
        store.validate(_votes().drop(columns="annotator_id"))


def test_validate_rejects_duplicate_vote():
    df = pd.concat([_votes(), _votes().iloc[[0]]])
    with pytest.raises(store.SchemaError, match="duplicate"):
        store.validate(df)


def test_one_annotator_may_pick_several_labels():
    extra = _votes().iloc[[0]].assign(label=2)
    assert len(store.validate(pd.concat([_votes(), extra]))) == 7


def test_validate_rejects_out_of_range_label():
    with pytest.raises(store.SchemaError, match="num_classes"):
        store.validate(_votes(), num_classes=2)


def test_validate_rejects_unknown_split():
    df = _votes().assign(split="holdout")
    with pytest.raises(store.SchemaError, match="split"):
        store.validate(df)


def test_vote_matrix_counts_and_majority():
    vm = store.vote_matrix(_votes(), num_classes=3, item_order=["b", "a"])
    np.testing.assert_array_equal(vm.counts, [[0, 0, 3], [2, 1, 0]])
    np.testing.assert_allclose(vm.distribution[1], [2 / 3, 1 / 3, 0])
    np.testing.assert_array_equal(vm.majority, [2, 0])


def test_round_trip(tmp_path):
    df = _votes().assign(split="train", is_error=False, error_type=None)
    path = store.save(df, tmp_path / "votes.parquet")
    back = store.load(path, num_classes=3)
    pd.testing.assert_frame_equal(back, store.validate(df))
