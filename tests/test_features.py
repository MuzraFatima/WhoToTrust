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
