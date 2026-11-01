import numpy as np
import pandas as pd
import pytest

from labeltriage.features import FORBIDDEN, check_inputs, g1_votes


def _votes(rows, **extra):
    item, ann, lab = zip(*rows, strict=True)
    df = pd.DataFrame({"dataset": "t", "item_id": item, "annotator_id": ann, "label": lab})
    return df.assign(**extra)


def test_check_inputs_strips_answer_columns():
    v = _votes([("a", "u", 0)], is_error=True, error_type="x", original_label=1)
    assert not set(FORBIDDEN) & set(check_inputs(v).columns)


def test_g1_lone_dissent():
    v = _votes([("a", "u1", 0), ("a", "u2", 0), ("a", "u3", 0), ("a", "u4", 1)])
    f = g1_votes.compute(v, 3)
    assert len(f) == 4
    assert f["g1_share"].tolist() == [0.75, 0.75, 0.75, 0.25]
    assert f["g1_rank"].tolist() == [0, 0, 0, 1]
    assert f["g1_alone"].tolist() == [0, 0, 0, 1]
    assert f["g1_share_others"].iloc[3] == 0.0
    assert f["g1_k"].eq(4).all()
    assert f["g1_entropy"].iloc[0] == pytest.approx(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))


def test_g1_even_split_has_tied_rank():
    v = _votes([("a", "u1", 0), ("a", "u2", 1)])
    f = g1_votes.compute(v, 2)
    assert f["g1_rank"].tolist() == [0, 0]
    assert f["g1_gap_to_top"].tolist() == [0.0, 0.0]


def _noisy_crowd(n_items=400, seed=0):
    """Ten annotators; two of them answer at random. True labels are uniform over 3 classes."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n_items)
    rows, bad = [], {"u8", "u9"}
    for i, t in enumerate(truth):
        for a in rng.choice(10, 5, replace=False):
            name = f"u{a}"
            lab = (
                rng.integers(0, 3)
                if name in bad
                else (t if rng.random() < 0.9 else rng.integers(0, 3))
            )
            rows.append((f"i{i}", name, int(lab)))
    return _votes(rows), truth


def test_dawid_skene_recovers_truth_and_spots_bad_annotators():
    from labeltriage.crowd import dawid_skene

    v, truth = _noisy_crowd()
    ds = dawid_skene(v, 3)
    order = np.argsort([int(i[1:]) for i in ds.item_ids])
    assert (ds.posterior.argmax(1)[order] == truth).mean() > 0.95
    acc = ds.annotator_accuracy
    assert acc[["u8", "u9"]].max() < acc.drop(["u8", "u9"]).min()


def test_g2_flags_unreliable_annotators_out_of_fold():
    from labeltriage.features import g2_reliability

    v, _ = _noisy_crowd()
    f = g2_reliability.compute(v, 3)
    assert len(f) == len(v) and not f.isna().any().any()
    bad = v["annotator_id"].isin(["u8", "u9"]).to_numpy()
    assert f.loc[bad, "g2_ds_accuracy"].mean() < f.loc[~bad, "g2_ds_accuracy"].mean() - 0.3
    assert (
        f.loc[bad, "g2_consensus_agreement"].mean() < f.loc[~bad, "g2_consensus_agreement"].mean()
    )


def test_g3_plausibility_against_majority():
    from labeltriage.features import g3_plausibility

    v = _votes([("a", "u1", 0), ("a", "u2", 0), ("a", "u3", 1), ("a", "u4", 2)])
    probs = pd.DataFrame([[0.6, 0.35, 0.05]], index=["a"])
    f = g3_plausibility.compute(v, probs)
    assert f["g3_p_label"].tolist() == pytest.approx([0.6, 0.6, 0.35, 0.05])
    assert f["g3_rank"].tolist() == [0, 0, 1, 2]
    assert f["g3_log_ratio_to_majority"].iloc[0] == pytest.approx(0.0)
    assert f["g3_log_ratio_to_majority"].iloc[3] < f["g3_log_ratio_to_majority"].iloc[2] < 0
    with pytest.raises(KeyError):
        g3_plausibility.compute(_votes([("zz", "u", 0)]), probs)


def test_g4_neighbours_share_ambiguity_but_not_errors():
    from labeltriage.features import g4_neighbours

    rng = np.random.default_rng(0)
    # Cluster A (items a0..a9): genuinely split between classes 0 and 1.
    # Cluster B (items b0..b9): clear class 2, with one stray vote for 1 on b0.
    rows = []
    for i in range(10):
        rows += [(f"a{i}", "u1", 0), (f"a{i}", "u2", 1), (f"a{i}", "u3", 0)]
        rows += [(f"b{i}", "u1", 2), (f"b{i}", "u2", 2), (f"b{i}", "u3", 2)]
    rows.append(("b0", "u4", 1))
    v = _votes(rows)
    ids = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)]
    centres = np.r_[np.tile([1.0, 0, 0], (10, 1)), np.tile([0, 1.0, 0], (10, 1))]
    emb = pd.DataFrame(centres + 0.01 * rng.standard_normal((20, 3)), index=ids)

    f = g4_neighbours.compute(v, 3, emb, k=5)
    amb_vote = (v["item_id"] == "a3") & (v["label"] == 1)
    stray = (v["item_id"] == "b0") & (v["label"] == 1)
    assert f.loc[amb_vote, "g4_nb_include"].item() == 1.0
    assert f.loc[stray, "g4_nb_include"].item() == 0.0


def test_neighbour_index_excludes_self():
    from labeltriage.features.g4_neighbours import neighbour_index

    x = np.random.default_rng(1).standard_normal((30, 4))
    nn = neighbour_index(x, 5)
    assert nn.shape == (30, 5)
    assert not (nn == np.arange(30)[:, None]).any()
