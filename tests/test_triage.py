import numpy as np
import pandas as pd
import pytest

from labeltriage.eval.metrics import auroc
from labeltriage.triage.model import TriageModel, calibration_error


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.2
    x = pd.DataFrame(
        {
            "g1_share": np.where(y, rng.uniform(0, 0.5, n), rng.uniform(0.3, 1, n)),
            "g2_acc": np.where(y, rng.normal(0.5, 0.15, n), rng.normal(0.8, 0.15, n)),
            "noise": rng.standard_normal(n),
        }
    )
    return x, y


@pytest.mark.parametrize("kind", ["lgbm", "mlp"])
def test_triage_detects_and_calibrates(kind):
    x, y = _data()
    tr, va, te = slice(0, 2000), slice(2000, 3000), slice(3000, None)
    m = TriageModel(list(x.columns), kind=kind).fit(x[tr], y[tr]).calibrate(x[va], y[va])
    p = m.predict(x[te])
    assert auroc(y[te], p) > 0.85
    assert ((p >= 0) & (p <= 1)).all()
    assert calibration_error(y[te].astype(float), p) < 0.06


def test_importance_points_at_informative_features():
    x, y = _data()
    imp = TriageModel(list(x.columns)).fit(x, y).importance()
    assert imp.index[-1] == "noise"


def test_calibration_error_zero_when_perfect():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0


def _table(n_items=600, seed=0):
    from labeltriage.ambinoise.injector import NoiseSpec, inject
    from labeltriage.features import g1_votes, g2_reliability

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        t = rng.integers(0, 3)
        share = rng.uniform(0.6, 1.0)
        rows += [
            (f"i{i:04d}", f"h{j}", int(t if rng.random() < share else (t + 1) % 3))
            for j in range(20)
        ]
    pool = pd.DataFrame(rows, columns=["item_id", "annotator_id", "label"])
    pool.insert(0, "dataset", "toy")
    pool["split"] = np.where(pool["item_id"] < "i0400", "train", "val")
    votes = inject(pool, 3, NoiseSpec(k=5, rate=0.2, num_annotators=60, concentration=1.0))
    feats = pd.concat([g1_votes.compute(votes, 3), g2_reliability.compute(votes, 3)], axis=1)
    return pd.concat([votes.reset_index(drop=True), feats], axis=1), list(feats.columns)


def test_pipeline_scores_every_vote_out_of_fold():
    from labeltriage.triage.pipeline import score

    table, cols = _table()
    res = score(table, cols)
    assert not np.isnan(res.p_error).any()
    m = res.metrics
    assert m["triage/val_auroc"] > 0.8
    assert m["triage/val_auroc"] > m["triage/val_entropy_only"]
    # Out-of-fold train AUROC should look like held-out val, not like a memorised fit.
    assert abs(m["triage/train_oof_auroc"] - m["triage/val_auroc"]) < 0.08


def test_ambiguity_recall_counts_soft_routes_on_split_items():
    from labeltriage.triage.pipeline import ambiguity_recall

    votes = pd.DataFrame(
        {
            "item_id": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 1, 0, 1, 0, 0],
            "is_error": [False, False, False, True, False, False],
        }
    )
    routes = pd.Series({"a": "ambiguous", "b": "clear", "c": "clear"})
    assert ambiguity_recall(votes, routes, 2) == 1.0


def test_noise_floor_shrinks_with_sample_size():
    from labeltriage.triage.model import calibration_noise_floor

    rng = np.random.default_rng(0)
    small = calibration_noise_floor(rng.uniform(0, 0.6, 200))
    large = calibration_noise_floor(rng.uniform(0, 0.6, 5000))
    assert large < small
    assert large < 0.03


def test_pair_scores_average_votes_per_item_label():
    from labeltriage.experiments.triage import pair_scores

    votes = pd.DataFrame(
        {
            "item_id": ["a", "a", "a", "b"],
            "label": [0, 0, 2, 1],
            "is_error": [False, False, True, False],
        }
    )
    pairs = pair_scores(votes, np.array([0.1, 0.3, 0.9, 0.2]))
    assert pairs[["item_id", "label"]].values.tolist() == [["a", 0], ["a", 2], ["b", 1]]
    assert pairs["p"].tolist() == pytest.approx([0.2, 0.9, 0.2])
    assert pairs["is_error"].tolist() == [False, True, False]
