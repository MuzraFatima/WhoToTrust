import pandas as pd
import pytest

from labeltriage.data import prepare, sources


def _src():
    votes = pd.DataFrame(
        {
            "dataset": "toy",
            "item_id": ["a", "a", "b", "c", "c"],
            "annotator_id": ["u", "v", "u", "u", "w"],
            "label": [0, 1, 1, 0, 0],
        }
    )
    items = pd.DataFrame({"gold": [0, 1, 0]}, index=pd.Index(["a", "b", "c"], name="item_id"))
    return sources.Source("toy", votes, items, ("x", "y"))


def test_finalise_attaches_split_and_meta():
    split = pd.Series({"a": "train", "b": "val", "c": "test"}, name="split")
    votes, items, meta = prepare.finalise(_src(), split, {"note": 1})
    assert votes["split"].tolist() == ["train", "train", "val", "test", "test"]
    assert items.loc["b", "split"] == "val"
    assert meta["num_items"] == 3
    assert meta["num_annotators"] == 3
    assert meta["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert meta["note"] == 1


def test_finalise_without_split_leaves_votes_unsplit():
    votes, items, meta = prepare.finalise(_src(), None)
    assert "split" not in votes
    assert "split_counts" not in meta


def test_resolve_split_refuses_stale_split(tmp_path, monkeypatch):
    stale = pd.Series({"a": "train", "b": "val"}, name="split")
    monkeypatch.setattr(prepare.splits, "load_split", lambda name: stale)
    with pytest.raises(ValueError, match="missing"):
        prepare.resolve_split("toy", _src(), tmp_path)


def test_committed_cifar10h_split_has_expected_sizes():
    split = prepare.splits.load_split("cifar10h")
    assert split.value_counts().to_dict() == {"train": 7000, "test": 2000, "val": 1000}
