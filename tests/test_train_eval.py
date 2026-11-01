import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_module

from labeltriage import run
from labeltriage.data import bundles
from labeltriage.utils.runs import StageError

pytest.importorskip("torch")
pytestmark = pytest.mark.torch


def _fake_bundle(n=24, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    x[y == 1] = 255
    ids = np.array([f"i{k}" for k in range(n)])
    votes = pd.DataFrame(
        {
            "dataset": "fake",
            "item_id": np.repeat(ids, 3),
            "annotator_id": np.tile(["a", "b", "c"], n),
            "label": np.repeat(y, 3),
        }
    )
    human = np.eye(2)[y]
    evals = {
        "val": bundles.EvalSpec("val", lambda: bundles.EvalSet(x, y, human)),
        "secret": bundles.EvalSpec("test", lambda: bundles.EvalSet(x, y)),
    }
    return bundles.Bundle("image", 2, x, ids, votes, evals)


def _cfg(*overrides):
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "experiment=reproduce",
                "data=cifar10h",
                "data.num_classes=2",
                "data.train.epochs=2",
                "data.train.batch_size=8",
                "data.train.width=4",
                "data.train.lr=0.05",
                "data.evals.dev=[val]",
                "data.evals.final=[secret]",
                *overrides,
            ],
        )


@pytest.mark.parametrize("method", ["majority_vote", "soft_labels"])
def test_train_eval_end_to_end(tmp_path, monkeypatch, method):
    monkeypatch.setattr(bundles, "load", lambda name, data_dir: _fake_bundle())
    result = run.execute(_cfg(f"method={method}"), tmp_path)
    m = result["metrics"]
    assert set(m) == {
        f"val/{k}" for k in ("acc", "ece", "nll", "kl", "js", "manhattan", "wasserstein")
    }
    assert 0.0 <= m["val/acc"] <= 1.0
    assert result["n_train"] == 24
    assert len(result["history"]) == 2


def test_test_sets_only_evaluated_in_final_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(bundles, "load", lambda name, data_dir: _fake_bundle())
    result = run.execute(_cfg("stage=final"), tmp_path)
    assert "secret/acc" in result["metrics"]

    with pytest.raises(StageError):
        _fake_bundle().eval_set("secret", "dev")


def _fake_pool_bundle(n=24, votes_per_item=8, seed=0):
    b = _fake_bundle(n, seed)
    rng = np.random.default_rng(seed)
    y = b.train_votes.groupby("item_id")["label"].first().reindex(b.train_item_ids).to_numpy()
    rows = [
        (i, f"h{j}", int(c if rng.random() < 0.8 else 1 - c), "train" if k < 20 else "val")
        for k, (i, c) in enumerate(zip(b.train_item_ids, y, strict=True))
        for j in range(votes_per_item)
    ]
    pool = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label", "split"])
    pool.insert(0, "dataset", "fake")
    train_ids = b.train_item_ids[:20]
    return bundles.Bundle(
        "image",
        2,
        b.train_inputs[:20],
        train_ids,
        pool[pool["split"] == "train"],
        b.evals,
        votes=pool,
    )


@pytest.mark.parametrize("method", ["oracle", "clean_votes"])
def test_ambinoise_runs_train_on_injected_votes(tmp_path, monkeypatch, method):
    monkeypatch.setattr(bundles, "load", lambda name, data_dir: _fake_pool_bundle())
    cfg = _cfg(
        "data=ambinoise_10h",
        "data.noise.k=5",
        "data.noise.rate=0.4",
        "data.noise.num_annotators=20",
        f"method={method}",
    )
    result = run.execute(cfg, tmp_path)
    assert result["n_train"] == 20
    assert "val/acc" in result["metrics"]
    if method == "oracle":
        assert any(k.startswith("route/") for k in result["metrics"])


def test_injection_model_writes_a_cross_fitted_ranking(tmp_path, monkeypatch):
    b = _fake_pool_bundle()
    pool = b.votes
    b.inputs_for = lambda ids: np.where(
        np.isin(ids, pool.loc[pool["label"] == 1, "item_id"])[:, None, None, None], 255, 0
    ).astype(np.uint8) * np.ones((1, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(bundles, "load", lambda name, data_dir: b)
    from labeltriage.data import prepare

    monkeypatch.setattr(
        prepare, "load_prepared", lambda name, d: (pool, None, {"classes": ["a", "b"]})
    )
    cfg = _cfg(
        "experiment=injection",
        "experiment.injection.folds=2",
        f"paths.cache_dir={tmp_path.as_posix()}/cache",
    )
    result = run.execute(cfg, tmp_path / "run")
    ranking = pd.read_parquet(result["ranking_path"])
    assert ranking.shape == (24, 2)
    np.testing.assert_allclose(ranking.sum(axis=1), 1.0, rtol=1e-5)


def test_labeltriage_method_routes_with_out_of_fold_scores(tmp_path, monkeypatch):
    from labeltriage.experiments import features as fx
    from labeltriage.features import g1_votes, g2_reliability

    monkeypatch.setattr(bundles, "load", lambda name, data_dir: _fake_pool_bundle(n=24))

    def fake_features(cfg, votes):
        f = pd.concat([g1_votes.compute(votes, 2), g2_reliability.compute(votes, 2)], axis=1)
        return pd.concat([votes.reset_index(drop=True), f], axis=1)

    monkeypatch.setattr(fx, "build_or_load", fake_features)
    cfg = _cfg(
        "data=ambinoise_10h",
        "data.noise.k=5",
        "data.noise.rate=0.4",
        "data.noise.num_annotators=20",
        "method=labeltriage",
        "method.groups=[g1,g2]",
    )
    result = run.execute(cfg, tmp_path)
    m = result["metrics"]
    assert "triage/val_auroc" in m and "route/ambiguity_recall" in m
    assert result["n_train"] == 20


@pytest.mark.parametrize(
    "method", ["dawid_skene", "entropy_weight", "confident_learning", "crowdlab"]
)
def test_baseline_methods_train_end_to_end(tmp_path, monkeypatch, method):
    pytest.importorskip("cleanlab")
    from labeltriage.experiments import features as fx
    from labeltriage.features import build as fb

    b = _fake_pool_bundle()
    monkeypatch.setattr(bundles, "load", lambda name, data_dir: b)
    ids = np.sort(b.votes["item_id"].unique())
    emb = pd.DataFrame(np.random.default_rng(0).standard_normal((len(ids), 4)), index=ids)
    monkeypatch.setattr(fx, "feature_inputs", lambda d, base: fb.FeatureInputs(None, emb))
    cfg = _cfg(
        "data=ambinoise_10h",
        "data.noise.k=5",
        "data.noise.num_annotators=20",
        f"method={method}",
    )
    result = run.execute(cfg, tmp_path)
    assert result["n_train"] == 20


def test_coteaching_method_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bundles, "load", lambda name, data_dir: _fake_pool_bundle())
    cfg = _cfg("data=ambinoise_10h", "data.noise.num_annotators=20", "method=coteaching")
    result = run.execute(cfg, tmp_path)
    assert result["n_train"] == 20


def test_labeltriage_on_real_noise_trains_triage_on_ambinoise(tmp_path, monkeypatch):
    from labeltriage.experiments import features as fx
    from labeltriage.features import g1_votes, g2_reliability

    monkeypatch.setattr(bundles, "load", lambda name, data_dir: _fake_pool_bundle(n=24))

    def fake_features(cfg, votes):
        f = pd.concat([g1_votes.compute(votes, 2), g2_reliability.compute(votes, 2)], axis=1)
        return pd.concat([votes.reset_index(drop=True), f], axis=1)

    monkeypatch.setattr(fx, "build_or_load", fake_features)
    cfg = _cfg(
        "method=labeltriage",
        "method.transfer_groups=[g1,g2]",
        "method.transfer_source.error_types=[spammer,swap]",
    )
    result = run.execute(cfg, tmp_path)
    m = result["metrics"]
    assert 0.0 <= m["triage/mean_p_error"] <= 1.0
    assert "triage/val_auroc" not in m


@pytest.mark.parametrize("method", ["neither", "errors_removed", "ambiguity_collapsed", "both"])
def test_harm_counterfactuals_train(tmp_path, monkeypatch, method):
    monkeypatch.setattr(bundles, "load", lambda name, data_dir: _fake_pool_bundle())
    cfg = _cfg("data=ambinoise_10h", "data.noise.num_annotators=20", f"method={method}")
    assert run.execute(cfg, tmp_path)["n_train"] == 20
