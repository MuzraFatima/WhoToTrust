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
