import io
import json
import pickle
import zipfile

import numpy as np
import pandas as pd
import pytest

from labeltriage.data import download, sources


def _write_pt(path, payload):
    """Mimic torch.save's zip layout for a dict of NumPy arrays."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(payload, protocol=2))
        z.writestr("archive/version", "3\n")


def test_cifar10h_drops_attention_checks(tmp_path):
    csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "annotator_id": [0, 1, 2, 0, 1],
            "is_attn_check": [0, 0, 0, 0, 1],
            "true_label": [3, 3, 3, 5, 5],
            "chosen_label": [3, 5, 3, 5, 0],
            "cifar10_test_test_idx": [10, 10, 10, 11, 11],
        }
    ).to_csv(csv, index=False)
    zpath = tmp_path / "raw.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(csv, "cifar10h-raw.csv")

    src = sources.cifar10h(zpath)
    assert len(src.votes) == 4
    assert src.items.loc["10", "gold"] == 3
    assert src.items.loc["11", "image_index"] == 11
    assert src.num_classes == 10


def test_cifar10n_maps_workers_to_label_slots(tmp_path):
    n = 20
    labels = {
        "clean_label": np.arange(n) % 10,
        "aggre_label": np.arange(n) % 10,
        "worse_label": np.zeros(n, dtype=np.int64),
        "random_label1": np.full(n, 1),
        "random_label2": np.full(n, 2),
        "random_label3": np.full(n, 3),
    }
    _write_pt(tmp_path / "h.pt", labels)
    (tmp_path / "side.csv").write_text(
        "Image-batch,Worker1-id,Work-time-in-seconds-1,Worker2-id,Work-time-in-seconds-2,"
        "Worker3-id,Work-time-in-seconds-3\n0--9,7,1,8,1,9,1\n10--19,7,1,4,1,5,1\n"
    )
    src = sources.cifar10n(tmp_path / "h.pt", tmp_path / "side.csv")
    assert len(src.votes) == 3 * n
    v = src.votes.set_index(["item_id", "annotator_id"])["label"]
    assert v[("0", "8")] == 2
    assert v[("15", "5")] == 3
    assert src.votes["annotator_id"].nunique() == 5


def test_numpy_only_unpickler_refuses_other_objects(tmp_path):
    _write_pt(tmp_path / "bad.pt", {"x": io.StringIO("nope")})
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        sources.load_numpy_dict_pt(tmp_path / "bad.pt")


def _chaos_record(uid, counts, premise="p", hypothesis="h", nested=True):
    e, n, c = counts
    body = {"premise": premise, "hypothesis": hypothesis}
    rec = {
        "uid": uid,
        "label_counter": {"e": e, "n": n, "c": c},
        "majority_label": "enc"[int(np.argmax(counts))],
        "old_label": "e",
    }
    return {**rec, "example": body} if nested else {**rec, **body}


def test_chaosnli_reads_official_zip_and_flat_mirror(tmp_path):
    zpath = tmp_path / "chaos.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("v1/chaosNLI_snli.jsonl", json.dumps(_chaos_record("s1", (60, 30, 10))))
        z.writestr("v1/chaosNLI_mnli_m.jsonl", json.dumps(_chaos_record("m1", (5, 5, 90))))
        z.writestr("v1/chaosNLI_alphanli.jsonl", json.dumps({"uid": "a1"}))
    src = sources.chaosnli(zpath)
    assert set(src.items.index) == {"s1", "m1"}
    assert src.items.loc["s1", "source"] == "snli"
    assert src.items.loc["m1", "gold"] == 2
    assert src.votes.groupby("item_id").size().tolist() == [100, 100]

    flat = tmp_path / "chaos_mnli.jsonl"
    flat.write_text(json.dumps(_chaos_record("m2", (1, 2, 3), nested=False)) + "\n")
    assert sources.chaosnli(flat).items.loc["m2", "source"] == "mnli"


def _expl(annotator):
    return {"annotator": annotator, "judgments": [], "reason": "r"}


def _varierr_record():
    return {
        "id": "x1",
        "context": "A  Premise",
        "statement": "a hypothesis",
        "entailment": [_expl(0), _expl(1)],
        "neutral": [],
        "contradiction": [_expl(1)],
        "error_labels": ["contradiction"],
        "has_ambiguity": False,
        "chaosnli_labels": {"e": 90, "n": 8, "c": 2},
    }


def test_varierr_marks_official_error_labels(tmp_path):
    path = tmp_path / "varierr.json"
    path.write_text(json.dumps(_varierr_record()) + "\n")
    src = sources.varierr(path)
    assert len(src.votes) == 3
    errs = src.votes[src.votes["is_error"]]
    assert errs["label"].tolist() == [2]
    assert src.items.loc["x1", "chaos_e"] == 90


def test_overlap_by_normalised_text_and_id(tmp_path):
    a = pd.DataFrame(
        {"premise": ["a premise", "other"], "hypothesis": ["a hypothesis", "x"]},
        index=["m1", "m2"],
    )
    b = pd.DataFrame({"premise": [" A   premise "], "hypothesis": ["A hypothesis"]}, index=["zz"])
    assert sources.nli_overlap(a, b) == {"m1"}
    b2 = pd.DataFrame({"premise": ["q"], "hypothesis": ["r"]}, index=["m2"])
    assert sources.nli_overlap(a, b2) == {"m2"}


def test_drop_items_removes_votes_and_items():
    votes = pd.DataFrame(
        {"dataset": "d", "item_id": ["a", "b"], "annotator_id": ["u", "u"], "label": [0, 1]}
    )
    items = pd.DataFrame({"gold": [0, 1]}, index=["a", "b"])
    out = sources.drop_items(sources.Source("d", votes, items, ("x", "y")), {"a"})
    assert out.votes["item_id"].tolist() == ["b"]
    assert list(out.items.index) == ["b"]


def test_verify_and_fetch_use_checksums(tmp_path):
    f = tmp_path / "raw" / "cifar10h" / "cifar10h-raw.zip"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"not the real file")
    assert download.verify(f, "")
    assert not download.verify(f, download.FILES["cifar10h"].checksum)
    with pytest.raises(download.ChecksumError):
        download.fetch("cifar10h", tmp_path)


def test_manual_file_explains_where_to_put_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="by hand"):
        download.fetch("chaosnli", tmp_path)
