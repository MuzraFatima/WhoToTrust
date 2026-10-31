import numpy as np
import pandas as pd
import pytest

from labeltriage.data import splits


def _items(n=1000, k=10, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([f"i{j}" for j in range(n)]), rng.integers(0, k, size=n)


def test_absolute_sizes_are_exact():
    ids, strata = _items(10_000)
    s = splits.stratified_split(ids, strata, {"train": 7000, "val": 1000, "test": 2000})
    assert s.value_counts().to_dict() == {"train": 7000, "test": 2000, "val": 1000}


def test_fractions_cover_every_item():
    ids, strata = _items(997)
    s = splits.stratified_split(ids, strata, {"train": 0.7, "val": 0.1, "test": 0.2})
    assert s.notna().all()
    assert len(s) == 997


def test_strata_proportions_are_kept():
    ids, strata = _items(10_000, k=10)
    s = splits.stratified_split(ids, strata, {"train": 7000, "val": 1000, "test": 2000})
    overall = np.bincount(strata, minlength=10) / len(strata)
    test = np.bincount(strata[(s == "test").to_numpy()], minlength=10) / 2000
    assert np.abs(test - overall).max() < 0.01


def test_split_is_deterministic_and_seed_dependent():
    ids, strata = _items()
    sizes = {"train": 0.7, "val": 0.1, "test": 0.2}
    a = splits.stratified_split(ids, strata, sizes, seed=1)
    b = splits.stratified_split(ids, strata, sizes, seed=1)
    c = splits.stratified_split(ids, strata, sizes, seed=2)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(c)


def test_bad_sizes_rejected():
    ids, strata = _items(100)
    with pytest.raises(ValueError):
        splits.stratified_split(ids, strata, {"train": 50, "test": 20})


def test_assign_and_round_trip(tmp_path):
    votes = pd.DataFrame(
        {
            "dataset": "d",
            "item_id": ["a", "a", "b", "c"],
            "annotator_id": ["u", "v", "u", "u"],
            "label": [0, 1, 1, 0],
        }
    )
    split = pd.Series({"a": "train", "b": "val", "c": "test"}, name="split")
    out = splits.assign(votes, split)
    assert out["split"].tolist() == ["train", "train", "val", "test"]

    splits.save_split(split.rename_axis("item_id"), "toy", tmp_path)
    back = splits.load_split("toy", tmp_path)
    assert back.to_dict() == split.to_dict()

    with pytest.raises(ValueError, match="no split"):
        splits.assign(votes, split.drop("c"))
