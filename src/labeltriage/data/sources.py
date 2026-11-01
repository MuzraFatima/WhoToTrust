"""Parsers that turn each raw dataset into the annotation store format.

Every parser returns a :class:`Source`: the long vote table, plus an ``items`` table with one row
per item holding whatever the item needs downstream (gold label, image index, premise and
hypothesis, and so on). Parsers take file paths so they can be tested on small synthetic files.
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from labeltriage.data import store

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
NLI_CLASSES = ("entailment", "neutral", "contradiction")
NLI_SHORT = {"e": 0, "n": 1, "c": 2}


@dataclass(frozen=True)
class Source:
    name: str
    votes: pd.DataFrame
    items: pd.DataFrame  # indexed by item_id
    classes: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def _votes_frame(dataset: str, item_ids, annotator_ids, labels, **extra) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "dataset": dataset,
            "item_id": pd.Series(item_ids).astype(str).to_numpy(),
            "annotator_id": pd.Series(annotator_ids).astype(str).to_numpy(),
            "label": np.asarray(labels, dtype=np.int64),
            **extra,
        }
    )
    return store.validate(df)


# --- CIFAR-10H -----------------------------------------------------------------------------


def cifar10h(raw_zip_or_csv: str | Path) -> Source:
    """CIFAR-10H human votes on the CIFAR-10 test set, attention checks removed.

    ``item_id`` is the CIFAR-10 test index, and ``items.gold`` is the original CIFAR-10 label.
    """
    path = Path(raw_zip_or_csv)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            raw = pd.read_csv(z.open(name))
    else:
        raw = pd.read_csv(path)

    raw = raw[raw["is_attn_check"] == 0]
    votes = _votes_frame(
        "cifar10h", raw["cifar10_test_test_idx"], raw["annotator_id"], raw["chosen_label"]
    )
    items = (
        raw.groupby("cifar10_test_test_idx")["true_label"]
        .first()
        .rename("gold")
        .rename_axis("item_id")
        .reset_index()
    )
    items["image_index"] = items["item_id"].astype(np.int64)
    items["item_id"] = items["item_id"].astype(str)
    return Source("cifar10h", votes, items.set_index("item_id"), CIFAR10_CLASSES)


# --- CIFAR-10N -----------------------------------------------------------------------------

_SAFE_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that can rebuild plain NumPy arrays and refuses everything else."""

    def find_class(self, module, name):
        if (module, name) in _SAFE_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("tensor storage is not expected in this file")


def load_numpy_dict_pt(path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``torch.save``-d dict of NumPy arrays without torch and without arbitrary code."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        return _NumpyOnlyUnpickler(io.BytesIO(z.read(name))).load()


def cifar10n(labels_pt: str | Path, side_info_csv: str | Path) -> Source:
    """CIFAR-10N: three human labels per CIFAR-10 training image, with real worker IDs.

    Worker IDs come from the side-info file, which assigns three workers to each batch of ten
    consecutive images; worker *j* produced ``random_label{j}``.
    """
    labels = load_numpy_dict_pt(labels_pt)
    n = len(labels["clean_label"])

    workers = np.empty((n, 3), dtype=object)
    with open(side_info_csv, newline="") as f:
        for row in csv.DictReader(f):
            lo, hi = (int(x) for x in row["Image-batch"].split("--"))
            for j in range(3):
                workers[lo : hi + 1, j] = row[f"Worker{j + 1}-id"]
    if any(w is None for w in workers.ravel()):
        raise ValueError("side info does not cover every image")

    idx = np.repeat(np.arange(n), 3)
    votes = _votes_frame(
        "cifar10n",
        idx,
        workers.ravel(),
        np.stack([labels[f"random_label{j + 1}"] for j in range(3)], axis=1).ravel(),
    )
    items = pd.DataFrame(
        {
            "item_id": np.arange(n).astype(str),
            "gold": labels["clean_label"].astype(np.int64),
            "image_index": np.arange(n),
            "aggre_label": labels["aggre_label"].astype(np.int64),
            "worse_label": labels["worse_label"].astype(np.int64),
        }
    )
    return Source("cifar10n", votes, items.set_index("item_id"), CIFAR10_CLASSES)


# --- ChaosNLI ------------------------------------------------------------------------------


def _chaos_part(filename: str) -> str:
    # Check "mnli" first: "chaosnli_mnli_m" also contains the letters "snli" (chao-snli).
    name = filename.lower()
    return "mnli" if "mnli" in name else "snli" if "snli" in name else "unknown"


def _chaos_rows(path: Path):
    """Yield ``(part, record)`` from the official zip, a ChaosNLI jsonl, or the HF mirror."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                base = Path(name).name
                if base.startswith("chaosNLI_") and base.endswith(".jsonl") and "alpha" not in base:
                    for line in z.open(name):
                        if line.strip():
                            yield _chaos_part(base), json.loads(line)
    else:
        with open(path, encoding="utf8") as f:
            for line in f:
                if line.strip():
                    yield _chaos_part(path.name), json.loads(line)


def chaosnli(paths: str | Path | list[str | Path]) -> Source:
    """ChaosNLI (SNLI and MNLI parts): 100 votes per item, with no annotator identities.

    ``paths`` is the official zip, one jsonl file, or several (e.g. the official
    ``chaosNLI_snli.jsonl`` and ``chaosNLI_mnli_m.jsonl``). The same item appearing in two inputs
    is an error, so the MNLI part can never be counted twice by also passing the HF mirror.

    Votes get synthetic annotator IDs ``v000``..``v099`` in a fixed order. Those IDs carry no
    meaning; AmbiNoise assigns its own simulated annotators when it samples K votes.
    """
    if isinstance(paths, str | Path):
        paths = [paths]
    item_rows, vote_items, vote_labels, vote_ann = [], [], [], []
    seen: dict[str, str] = {}
    for path in map(Path, paths):
        for part, rec in _chaos_rows(path):
            uid = rec["uid"]
            if uid in seen:
                raise ValueError(f"ChaosNLI item {uid} appears in both {seen[uid]} and {path.name}")
            seen[uid] = path.name
            _chaos_record(part, rec, item_rows, vote_items, vote_labels, vote_ann)
    votes = _votes_frame("chaosnli", vote_items, vote_ann, vote_labels)
    items = pd.DataFrame(item_rows).set_index("item_id")
    return Source("chaosnli", votes, items, NLI_CLASSES)


def _chaos_record(part, rec, item_rows, vote_items, vote_labels, vote_ann) -> None:
    """Append one ChaosNLI record's item row and its 100 votes to the running lists."""
    ex = rec.get("example", rec)  # official files nest the text; the HF mirror is flat
    uid = rec["uid"]
    counts = [int(rec["label_counter"].get(k, 0)) for k in ("e", "n", "c")]
    labels = np.repeat(np.arange(3), counts)
    vote_items.extend([uid] * len(labels))
    vote_labels.extend(labels.tolist())
    vote_ann.extend(f"v{k:03d}" for k in range(len(labels)))
    item_rows.append(
        {
            "item_id": uid,
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "source": part,
            "gold": NLI_SHORT[rec["majority_label"]],
            "old_label": NLI_SHORT.get(rec.get("old_label", ""), -1),
        }
    )


# --- VariErr -------------------------------------------------------------------------------


def varierr(path: str | Path) -> Source:
    """VariErr NLI: per-annotator labels with explanations, and official error labels.

    A vote is one (item, annotator, label) triple. ``is_error`` is True when the label is in the
    item's official ``error_labels`` (amendment A2).
    """
    item_rows, rows = [], []
    with open(path, encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            errors = set(rec["error_labels"])
            for c, name in enumerate(NLI_CLASSES):
                annotators = {e["annotator"] for e in rec[name]}
                rows.extend((rec["id"], a, c, name in errors) for a in sorted(annotators))
            chaos = rec.get("chaosnli_labels") or {}
            item_rows.append(
                {
                    "item_id": rec["id"],
                    "premise": rec["context"],
                    "hypothesis": rec["statement"],
                    "error_labels": ",".join(sorted(errors)),
                    "has_ambiguity": bool(rec.get("has_ambiguity", False)),
                    **{f"chaos_{k}": int(chaos.get(k, 0)) for k in ("e", "n", "c")},
                }
            )
    item_ids, annotators, labels, is_error = zip(*rows, strict=True)
    votes = _votes_frame(
        "varierr", item_ids, annotators, labels, is_error=pd.array(is_error, dtype="boolean")
    )
    items = pd.DataFrame(item_rows).set_index("item_id")
    return Source("varierr", votes, items, NLI_CLASSES)


# --- Overlap -------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def nli_overlap(a: pd.DataFrame, b: pd.DataFrame) -> set[str]:
    """Item IDs of ``a`` that also appear in ``b``, by ID or by normalised premise/hypothesis."""
    keys_b = {(_norm(p), _norm(h)) for p, h in zip(b["premise"], b["hypothesis"], strict=True)}
    by_text = {
        i
        for i, p, h in zip(a.index, a["premise"], a["hypothesis"], strict=True)
        if (_norm(p), _norm(h)) in keys_b
    }
    return by_text | (set(a.index) & set(b.index))


def drop_items(src: Source, item_ids: set[str]) -> Source:
    votes = src.votes[~src.votes["item_id"].isin(item_ids)].reset_index(drop=True)
    items = src.items.drop(index=list(set(item_ids) & set(src.items.index)))
    return Source(src.name, votes, items, src.classes)
