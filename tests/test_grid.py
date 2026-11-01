import json

from labeltriage.grid import expand, run_grid

SPEC = {
    "name": "t",
    "fixed": ["experiment=smoke"],
    "axes": {"method": ["majority_vote", "soft_labels"], "seed": [0, 1]},
}


def test_expand_is_cartesian_product():
    points = list(expand(SPEC))
    assert len(points) == 4
    assert points[0] == ["experiment=smoke", "method=majority_vote", "seed=0"]


def test_grid_resumes_without_rerunning(tmp_path):
    assert run_grid(SPEC, tmp_path) == {"ran": 4, "skipped": 0, "failed": 0}
    assert run_grid(SPEC, tmp_path) == {"ran": 0, "skipped": 4, "failed": 0}
    assert len(list(tmp_path.rglob("DONE"))) == 4


def test_extra_overrides_apply_to_every_run(tmp_path):
    run_grid(SPEC, tmp_path, ["data.num_items=10"])
    items = {json.loads(p.read_text())["items"] for p in tmp_path.rglob("result.json")}
    assert items == {10}


def test_tuning_variants_get_their_own_run_directories(tmp_path):
    from hydra import compose, initialize_config_module

    from labeltriage.grid import run_dir_for

    dirs = set()
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        for nu in (1.0, 2.0):
            cfg = compose(
                config_name="config", overrides=["method=labeltriage_tune", f"method.nu={nu}"]
            )
            dirs.add(run_dir_for(cfg, tmp_path))
    assert len(dirs) == 2
    assert all("labeltriage-nu" in str(d) for d in dirs)
