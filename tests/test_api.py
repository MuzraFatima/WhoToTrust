import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from labeltriage import api  # noqa: E402


def _scored(n_items=12, votes=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        route = ["clear", "ambiguous", "unreliable"][i % 3]
        for v in range(votes):
            err = v == 0 and i % 4 == 0
            rows.append(
                {
                    "item_id": f"it{i:02d}",
                    "annotator_id": f"a{v}",
                    "label": int(rng.integers(0, 3)),
                    "is_error": err,
                    "error_type": "lazy" if err else "none",
                    "g1_entropy": 0.1 * i,
                    "g1_share": 0.2 if err else 0.8,
                    "g2_ds_accuracy": 0.9,
                    "g3_p_label": 0.5,
                    "g4_nb_include": 0.5,
                    "g5_aum": 0.5,
                    "p_error": 0.9 if err else 0.05 + 0.01 * i,
                    "route": route,
                }
            )
    return pd.DataFrame(rows)


def _write_run(root, rel, base, error_type, seed=0):
    d = root / rel
    d.mkdir(parents=True)
    _scored(seed=seed).to_parquet(d / "scored_votes.parquet")
    cfg = {
        "data": {
            "base": base,
            "name": f"ambinoise_{'10h' if base == 'cifar10h' else 'nli'}_k5_r0.2_{error_type}",
            "noise": {"error_type": error_type, "k": 5, "rate": 0.2},
        },
        "method": {"name": "labeltriage"},
        "experiment": {"name": "pilot"},
        "seed": seed,
        "stage": "dev",
    }
    metrics = {"val/acc": 0.9, "val/kl": 0.2, "clean_dev/acc": 0.88}
    (d / "result.json").write_text(
        json.dumps({"config": cfg, "metrics": metrics, "train_seconds": 3.0})
    )
    (d / "DONE").write_text("")
    return d


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "grid"
    _write_run(root, "dev/pilot/ambinoise_nli_k5_r0.2_lazy/labeltriage/seed0", "chaosnli", "lazy")
    _write_run(
        root, "dev/pilot/ambinoise_10h_k5_r0.2_spammer/labeltriage/seed0", "cifar10h", "spammer"
    )
    proc = tmp_path / "data" / "processed" / "chaosnli"
    proc.mkdir(parents=True)
    (proc / "meta.json").write_text(
        json.dumps({"classes": ["entailment", "neutral", "contradiction"]})
    )
    items = pd.DataFrame(
        {
            "premise": [f"premise {i}" for i in range(12)],
            "hypothesis": [f"hyp {i}" for i in range(12)],
        },
        index=pd.Index([f"it{i:02d}" for i in range(12)], name="item_id"),
    )
    items.to_parquet(proc / "items.parquet")
    api._base_info.cache_clear()
    api._results.cache_clear()
    return TestClient(api.create_app(root, tmp_path / "data", web_dir=None))


NLI = "dev/pilot/ambinoise_nli_k5_r0.2_lazy/labeltriage/seed0"
IMG = "dev/pilot/ambinoise_10h_k5_r0.2_spammer/labeltriage/seed0"


def test_runs_are_listed_with_their_setting(client):
    runs = {r["id"]: r for r in client.get("/api/runs").json()}
    assert set(runs) == {NLI, IMG}
    assert runs[NLI]["dataset"] == "chaosnli" and runs[NLI]["error_type"] == "lazy"
    assert runs[NLI]["n_items"] == 12 and runs[NLI]["n_votes"] == 60


def test_empty_folder_is_not_an_error(tmp_path):
    c = TestClient(api.create_app(tmp_path / "nothing", tmp_path, web_dir=None))
    assert c.get("/api/runs").json() == []
    assert c.get("/api/results").json() == {"n_runs": 0, "sections": {}}
    assert c.get("/api/health").json()["runs"] == 0


def test_points_are_compact_parallel_arrays(client):
    p = client.get("/api/points", params={"run": NLI}).json()
    n = len(p["item_id"])
    assert n == 12 and len(p["entropy"]) == len(p["max_p_error"]) == len(p["route"]) == n
    assert p["routes"] == ["clear", "ambiguous", "unreliable"]
    assert len(p["true_errors"]) == n


def test_items_are_sorted_filtered_and_paginated(client):
    r = client.get("/api/items", params={"run": NLI, "page_size": 5}).json()
    assert r["total"] == 12 and r["pages"] == 3 and len(r["items"]) == 5
    scores = [i["max_p_error"] for i in r["items"]]
    assert scores == sorted(scores, reverse=True)

    only = client.get("/api/items", params={"run": NLI, "only_errors": True}).json()
    assert only["total"] == 3 and all(i["true_errors"] > 0 for i in only["items"])

    routes = client.get("/api/items", params={"run": NLI, "route": ["clear", "unreliable"]}).json()
    assert {i["route"] for i in routes["items"]} == {"clear", "unreliable"}

    found = client.get("/api/items", params={"run": NLI, "q": "IT07"}).json()
    assert [i["item_id"] for i in found["items"]] == ["it07"]

    last = client.get("/api/items", params={"run": NLI, "page": 3, "page_size": 5}).json()
    assert len(last["items"]) == 2


def test_item_detail_has_votes_reasons_class_names_and_text(client):
    d = client.get("/api/item", params={"run": NLI, "item": "it00"}).json()
    assert d["text"] == {"premise": "premise 0", "hypothesis": "hyp 0"}
    assert len(d["votes"]) == 5 and d["votes"][0]["p_error"] == 0.9
    assert d["votes"][0]["label_name"] in d["classes"]
    assert "lone dissent among the votes" in d["votes"][0]["reasons"]
    assert d["has_image"] is False


def test_unknown_item_run_and_missing_picture_are_404(client):
    assert client.get("/api/item", params={"run": NLI, "item": "nope"}).status_code == 404
    assert client.get("/api/items", params={"run": "dev/missing"}).status_code == 404
    assert client.get("/api/image", params={"run": NLI, "item": "it00"}).status_code == 404
    assert "error" in client.get("/api/items", params={"run": "dev/missing"}).json()


def test_path_traversal_is_refused(client, tmp_path):
    (tmp_path / "scored_votes.parquet").write_bytes(b"x")
    for run in ("..", "../", "../../etc", str(tmp_path), "dev/../../"):
        assert client.get("/api/points", params={"run": run}).status_code == 404


def test_bad_query_values_are_rejected(client):
    assert client.get("/api/items", params={"run": NLI, "sort": "drop table"}).status_code == 422
    assert client.get("/api/items", params={"run": NLI, "page_size": 5000}).status_code == 422
    assert client.get("/api/items", params={"run": NLI, "page": 0}).status_code == 422


def test_responses_are_gzipped_and_revalidate_with_etag(tmp_path):
    d = tmp_path / "grid" / "dev" / "big"
    d.mkdir(parents=True)
    _scored(n_items=400).to_parquet(d / "scored_votes.parquet")
    c = TestClient(api.create_app(tmp_path / "grid", tmp_path, web_dir=None))
    r = c.get("/api/points", params={"run": "dev/big"}, headers={"Accept-Encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip"
    tag = r.headers["etag"]
    assert r.headers["cache-control"] == "no-cache"
    again = c.get("/api/points", params={"run": "dev/big"}, headers={"If-None-Match": tag})
    assert again.status_code == 304 and again.content == b""
    assert c.get("/api/points", params={"run": "dev/big"}).headers["etag"] == tag


def test_results_group_runs_by_pipeline_stage(client):
    res = client.get("/api/results").json()
    assert res["n_runs"] == 2
    compare = res["sections"]["compare"]
    assert {c["dataset"] for c in compare} == {"chaosnli", "cifar10h"}
    assert compare[0]["val_acc"] == pytest.approx(0.9)
    assert res["sections"]["reproduce"] == [] and res["sections"]["transfer"] == []


def test_clean_makes_nan_and_numpy_json_safe():
    out = api.clean(
        {"a": float("nan"), "b": np.int64(3), "c": [np.float32(0.5), pd.NA], "d": np.bool_(True)}
    )
    assert out == {"a": None, "b": 3, "c": [0.5, None], "d": True}
    json.dumps(out, allow_nan=False)


def test_a_relative_runs_folder_works_too(tmp_path, monkeypatch):
    _write_run(tmp_path / "grid", "dev/r/ambinoise_nli_k5_r0.2_lazy/m/seed0", "chaosnli", "lazy")
    monkeypatch.chdir(tmp_path)
    c = TestClient(api.create_app("grid", "data", web_dir=None))
    runs = c.get("/api/runs").json()
    assert [r["id"] for r in runs] == ["dev/r/ambinoise_nli_k5_r0.2_lazy/m/seed0"]
    assert c.get("/api/points", params={"run": runs[0]["id"]}).status_code == 200
