from labeltriage.eval.summary import aggregate, collect
from labeltriage.utils.runs import write_result


def _result(method, seed, acc):
    return {
        "config": {
            "experiment": {"name": "e"},
            "data": {"name": "d"},
            "method": {"name": method},
            "seed": seed,
            "stage": "dev",
        },
        "metrics": {"val/acc": acc},
        "train_seconds": 10.0,
    }


def test_collect_and_aggregate(tmp_path):
    write_result(tmp_path / "a0", _result("a", 0, 0.8))
    write_result(tmp_path / "a1", _result("a", 1, 0.6))
    write_result(tmp_path / "b0", _result("b", 0, 0.5))
    (tmp_path / "unfinished").mkdir()

    runs = collect(tmp_path)
    assert len(runs) == 3

    table = aggregate(runs).set_index("method")
    assert table.loc["a", "val/acc_mean"] == 0.7
    assert table.loc["a", "n_seeds"] == 2
    assert table.loc["b", "n_seeds"] == 1
