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
    assert (tmp_path / "rel" / "README.md").read_text(encoding="utf8").startswith("# AmbiNoise")
    assert json.loads((tmp_path / "rel" / "manifest.json").read_text())["draws"]
