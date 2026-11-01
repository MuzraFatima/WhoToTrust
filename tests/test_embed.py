import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_module

from labeltriage import run
from labeltriage.experiments import embed


def _cfg(tmp_path, data):
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "experiment=embed",
                f"data={data}",
                f"paths.cache_dir={tmp_path.as_posix()}/cache",
            ],
        )


@pytest.mark.parametrize("domain", ["image", "text"])
def test_embed_writes_both_caches(tmp_path, monkeypatch, domain):
    ids = pd.Index(["a", "b", "c"], name="item_id")
    classes = ("x", "y", "z") if domain == "image" else ("entailment", "neutral", "contradiction")
    monkeypatch.setattr(
        embed, "item_inputs", lambda base, d: (ids, list(range(3)), domain, classes)
    )
    probs = lambda *a, **k: (lambda x: np.full((len(x), 3), 1 / 3))  # noqa: E731
    vecs = lambda *a, **k: (lambda x: np.arange(len(x) * 4.0).reshape(len(x), 4))  # noqa: E731
    for name in ("clip_zero_shot", "mnli_probs"):
        monkeypatch.setattr(embed, name, probs)
    for name in ("dinov2_embed", "sentence_embed"):
        monkeypatch.setattr(embed, name, vecs)

    result = run.execute(
        _cfg(tmp_path, "cifar10h" if domain == "image" else "varierr"), tmp_path / "r"
    )
    assert result["embedding_dim"] == 4
    emb = embed.load(result["paths"]["embeddings"])
    assert list(emb.index) == ["a", "b", "c"]
    assert list(emb.columns) == [0, 1, 2, 3]
    assert embed.load(result["paths"]["plausibility"]).shape == (3, 3)


def test_batched_concatenates_in_order():
    out = embed.batched(lambda b: np.asarray(b)[:, None] * 2, list(range(7)), 3)
    assert out.ravel().tolist() == [0, 2, 4, 6, 8, 10, 12]
