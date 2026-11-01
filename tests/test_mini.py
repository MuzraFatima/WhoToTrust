import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_module

from labeltriage.data import prepare, splits, store


def _compose(*overrides):
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def test_full_profile_is_the_default_and_changes_nothing():
    cfg = _compose("data=cifar10h")
    assert cfg.paths.cache_dir == "cache" and cfg.paths.output_dir == "outputs"
    assert "subset" not in cfg.data and cfg.data.train.epochs == 60


@pytest.mark.parametrize("data", ["cifar10h", "chaosnli", "ambinoise_10h", "ambinoise_nli"])
def test_mini_profile_uses_the_subset_frozen_trainer_and_own_folders(data):
    cfg = _compose(f"data={data}", "profile=mini")
    assert cfg.data.subset == "mini"
    assert cfg.data.train.backbone == "frozen_head"
    assert cfg.paths.cache_dir == "cache_mini" and cfg.paths.output_dir == "outputs_mini"
    assert cfg.stage == "dev"


@pytest.mark.parametrize(
    ("name", "sizes"),
    [
        ("cifar10h", {"train": 1000, "val": 200, "test": 300}),
        ("chaosnli", {"train": 800, "val": 150, "test": 250}),
        ("cifar10_train_clean", {"val": 500, "test": 1000}),
        ("varierr", {"val": 250, "test": 250}),
    ],
)
def test_committed_mini_subsets_keep_each_items_original_split(name, sizes):
    mini = splits.load_split(name, splits.splits_dir() / "mini")
    full = splits.load_split(name)
    assert mini.value_counts().to_dict() == sizes
    assert mini.index.is_unique and mini.index.isin(full.index).all()
    assert (full.loc[mini.index] == mini).all()


def _processed(tmp_path):
    votes = pd.DataFrame(
        {
            "dataset": "toy",
            "item_id": ["a", "a", "b", "b", "c", "c"],
            "annotator_id": ["x", "y"] * 3,
            "label": [0, 0, 1, 1, 0, 1],
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    )
    d = tmp_path / "data" / "processed" / "toy"
    d.mkdir(parents=True)
    store.save(votes, d / "votes.parquet")
    items = pd.DataFrame({"split": ["train", "val", "test"]}, index=pd.Index(list("abc")))
    items.index.name = "item_id"
    items.to_parquet(d / "items.parquet")
    (d / "meta.json").write_text('{"classes": ["n", "p"]}')
    return tmp_path / "data"


def test_load_prepared_with_a_subset_keeps_only_its_items(tmp_path, monkeypatch):
    data = _processed(tmp_path)
    monkeypatch.setattr(splits, "splits_dir", lambda: tmp_path / "splits")
    keep = pd.Series({"a": "train", "c": "test"}, name="split")
    keep.index.name = "item_id"
    splits.save_split(keep, "toy", tmp_path / "splits" / "mini")

    votes, items, _ = prepare.load_prepared("toy", data, "mini")
    assert items.index.tolist() == ["a", "c"]
    assert set(votes["item_id"]) == {"a", "c"} and len(votes) == 4
    assert len(prepare.load_prepared("toy", data)[1]) == 3


def test_load_prepared_refuses_a_subset_with_unknown_items(tmp_path, monkeypatch):
    data = _processed(tmp_path)
    monkeypatch.setattr(splits, "splits_dir", lambda: tmp_path / "splits")
    bad = pd.Series({"a": "train", "zz": "val"}, name="split")
    bad.index.name = "item_id"
    splits.save_split(bad, "toy", tmp_path / "splits" / "mini")
    with pytest.raises(ValueError, match="not in processed"):
        prepare.load_prepared("toy", data, "mini")


def test_take_is_stratified_and_deterministic():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_mini.py"
    spec = importlib.util.spec_from_file_location("make_mini", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ids = np.array([f"i{k}" for k in range(100)])
    strata = np.repeat([0, 1], [80, 20])
    kept = mod.take(ids, strata, 10)
    assert len(kept) == 10
    assert np.isin(kept, ids[strata == 1]).sum() == 2
    assert (kept == mod.take(ids, strata, 10)).all()


# --- the frozen-embedding trainer (needs PyTorch) -------------------------------------------


@pytest.mark.torch
def test_frozen_head_learns_separable_features_and_is_reproducible():
    torch = pytest.importorskip("torch")
    from labeltriage.targets import TargetSet
    from labeltriage.train import frozen, loop

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 90)
    feats = (np.eye(3)[y] * 3 + rng.normal(0, 0.3, (90, 3))).astype(np.float32)
    ts = TargetSet(np.arange(90).astype(str), np.eye(3)[y], np.ones(90))
    cfg = loop.LoopConfig(epochs=15, batch_size=16, optimizer="adamw", lr=1e-2, amp=False)
    cpu = torch.device("cpu")
    model, history = frozen.fit(feats, ts, 3, cfg, seed=0, device=cpu, hidden=16)
    assert (frozen.predict(model, feats, device=cpu).argmax(1) == y).mean() >= 0.95
    _, again = frozen.fit(feats, ts, 3, cfg, seed=0, device=cpu, hidden=16)
    assert history == again


@pytest.mark.torch
def test_frozen_coteaching_runs():
    torch = pytest.importorskip("torch")
    from labeltriage.targets import TargetSet
    from labeltriage.train import frozen, loop

    y = np.arange(40) % 2
    feats = np.eye(2, dtype=np.float32)[y]
    ts = TargetSet(np.arange(40).astype(str), np.eye(2)[y], np.ones(40))
    cfg = loop.LoopConfig(epochs=3, batch_size=8, optimizer="adamw", lr=1e-2, amp=False)
    _, history = frozen.fit(feats, ts, 2, cfg, 0, torch.device("cpu"), 8, coteach=(0.2, 2))
    assert len(history) == 3


@pytest.mark.torch
def test_encode_computes_once_then_reads_the_cache(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from labeltriage.train import frozen

    calls = []

    def fake_encoder(domain, model_name):
        def run(inputs):
            calls.append(1)
            return np.ones((len(inputs[0]), 4))

        return run

    monkeypatch.setattr(frozen, "_encoder", fake_encoder)
    pairs = (["p1", "p2"], ["h1", "h2"])
    a = frozen.encode("text", pairs, "org/model", tmp_path)
    b = frozen.encode("text", pairs, "org/model", tmp_path)
    assert a.shape == (2, 4) and (a == b).all() and len(calls) == 1
    frozen.encode("text", (["p1", "p2"], ["h1", "other"]), "org/model", tmp_path)
    assert len(calls) == 2


@pytest.mark.torch
def test_content_key_tracks_image_bytes():
    pytest.importorskip("torch")
    from labeltriage.train import frozen

    x = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    y = x.copy()
    y[0, 0, 0, 0] = 1
    assert frozen.content_key("image", x) == frozen.content_key("image", x.copy())
    assert frozen.content_key("image", x) != frozen.content_key("image", y)


@pytest.mark.torch
def test_train_eval_under_the_mini_profile_uses_the_frozen_head(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from labeltriage import run
    from labeltriage.data import bundles
    from labeltriage.train import frozen

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 24)
    x = np.zeros((24, 32, 32, 3), dtype=np.uint8)
    x[y == 1] = 255
    ids = np.array([f"i{k}" for k in range(24)])
    votes = pd.DataFrame(
        {
            "dataset": "fake",
            "item_id": np.repeat(ids, 3),
            "annotator_id": np.tile(["a", "b", "c"], 24),
            "label": np.repeat(y, 3),
        }
    )
    evals = {"val": bundles.EvalSpec("val", lambda: bundles.EvalSet(x, y, np.eye(2)[y]))}
    seen = {}

    def fake_load(name, data_dir, subset=None):
        seen["subset"] = subset
        return bundles.Bundle("image", 2, x, ids, votes, evals)

    def fake_encode(domain, inputs, name, cache):
        bright = inputs.reshape(len(inputs), -1)[:, :1] / 255.0
        return np.hstack([bright, 1 - bright, np.full_like(bright, 0.5)]).astype(np.float32)

    monkeypatch.setattr(bundles, "load", fake_load)
    monkeypatch.setattr(frozen, "encode", fake_encode)
    cfg = _compose(
        "experiment=reproduce",
        "data=cifar10h",
        "profile=mini",
        "data.num_classes=2",
        "data.train.epochs=30",
        "data.train.lr=0.01",
        "data.evals.dev=[val]",
    )
    result = run.execute(cfg, tmp_path)
    assert seen["subset"] == "mini"
    assert result["metrics"]["val/acc"] >= 0.9
    assert len(result["history"]) == 30


def test_every_mini_sweep_point_composes_and_stays_off_the_test_split():
    from omegaconf import OmegaConf

    from labeltriage.grid import expand

    sweeps = sorted((splits.splits_dir().parent / "sweep").glob("mini_*.yaml"))
    assert len(sweeps) == 6
    for path in sweeps:
        for point in expand(OmegaConf.load(path)):
            cfg = _compose(*point)
            assert cfg.data.subset == "mini", (path.name, point)
            assert cfg.stage == "dev", (path.name, point)
