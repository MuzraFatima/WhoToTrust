import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labeltriage.data import prepare


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "release_ambinoise.py"
    spec = importlib.util.spec_from_file_location("release", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pool(num_classes, n_items=30, votes=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = [
        (f"i{i}", f"h{j}", int(rng.integers(0, num_classes)), "train" if i < 20 else "val")
        for i in range(n_items)
        for j in range(votes)
    ]
    df = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label", "split"])
    df.insert(0, "dataset", "toy")
    return df


def test_release_writes_draws_card_and_manifest(tmp_path, monkeypatch):
    mod = _load_script()
    pools = {"cifar10h": _pool(10), "chaosnli": _pool(3)}
    monkeypatch.setattr(prepare, "load_prepared", lambda base, d: (pools[base], None, None))
    m = mod.build(tmp_path / "rel", "data", str(tmp_path / "nocache"), [0], ks=(5,), rates=(0.2,))
    # 2 bases x 3 runnable error types; plausible_wrong is skipped without rankings.
    assert len(m["draws"]) == 6 and len(m["skipped"]) == 2
    draw = pd.read_parquet(next((tmp_path / "rel" / "10h").glob("*spammer*.parquet")))
    assert list(draw.columns) == mod.PUBLIC_COLUMNS
    card = (tmp_path / "rel" / "README.md").read_text(encoding="utf8")
    assert "\n# AmbiNoise dataset card" in card
    assert json.loads((tmp_path / "rel" / "manifest.json").read_text())["draws"]


def _upload_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "upload_ambinoise.py"
    spec = importlib.util.spec_from_file_location("upload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def _release(tmp_path, licence):
    rel = tmp_path / "rel"
    (rel / "10h").mkdir(parents=True)
    (rel / "10h" / "a.parquet").write_bytes(b"x")
    (rel / "README.md").write_text(
        f"---\npretty_name: AmbiNoise\nlicense: {licence}\n---\n# AmbiNoise\n"
    )
    return rel


def test_built_card_has_hub_front_matter(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(
        prepare,
        "load_prepared",
        lambda base, d: (_pool(10 if base == "cifar10h" else 3), None, None),
    )
    mod.build(tmp_path / "rel", "data", str(tmp_path / "nocache"), [0], ks=(5,), rates=(0.2,))
    text = (tmp_path / "rel" / "README.md").read_text(encoding="utf8")
    assert text.startswith("---\npretty_name: AmbiNoise\nlicense: other")
    assert "## Licensing" in text


def test_upload_refuses_while_licence_is_other(tmp_path):
    api = _FakeApi()
    rc = _upload_script().main(
        [str(_release(tmp_path, "other")), "--repo-id", "u/a", "--yes"], api=api
    )
    assert rc == 2 and api.calls == []


def test_upload_is_a_dry_run_without_yes(tmp_path):
    api = _FakeApi()
    rc = _upload_script().main(
        [str(_release(tmp_path, "cc-by-nc-4.0")), "--repo-id", "u/a"], api=api
    )
    assert rc == 0 and api.calls == []


def test_upload_creates_a_private_repo_with_yes(tmp_path):
    api = _FakeApi()
    rc = _upload_script().main(
        [str(_release(tmp_path, "cc-by-nc-4.0")), "--repo-id", "u/a", "--yes"], api=api
    )
    assert rc == 0
    assert api.calls[0] == (
        "create_repo",
        "u/a",
        {"repo_type": "dataset", "private": True, "exist_ok": True},
    )
    assert api.calls[1][0] == "upload_folder"
