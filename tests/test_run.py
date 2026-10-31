import pytest
from hydra import compose, initialize_config_module

from labeltriage import run
from labeltriage.utils import runs


def _cfg(*overrides):
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def test_smoke_run_writes_result_and_marker(tmp_path):
    result = run.execute(_cfg("seed=3"), tmp_path)
    assert result["items"] == 200
    assert runs.is_done(tmp_path)
    assert runs.read_result(tmp_path)["config"]["seed"] == 3


def test_finished_run_is_skipped(tmp_path):
    run.execute(_cfg(), tmp_path)
    assert run.execute(_cfg(), tmp_path) is None


def test_same_seed_same_result(tmp_path):
    a = run.execute(_cfg("seed=7"), tmp_path / "a")
    b = run.execute(_cfg("seed=7"), tmp_path / "b")
    assert a["full_agreement"] == b["full_agreement"]


def test_test_split_locked_outside_final_stage():
    with pytest.raises(runs.StageError):
        runs.require_stage("dev", "test")
    runs.require_stage("dev", "val")
    runs.require_stage("final", "test")
