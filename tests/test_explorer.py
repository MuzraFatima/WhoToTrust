import numpy as np
import pandas as pd

from labeltriage import explorer


def _scored():
    return pd.DataFrame(
        {
            "item_id": ["a"] * 5 + ["b"] * 5,
            "annotator_id": list("uvwxy") * 2,
            "label": [3, 3, 3, 3, 9, 3, 3, 5, 5, 5],
            "is_error": [False] * 4 + [True] + [False] * 5,
            "g1_entropy": [0.5] * 5 + [0.67] * 5,
            "g1_share": [0.8] * 4 + [0.2] + [0.4, 0.4, 0.6, 0.6, 0.6],
            "g2_ds_accuracy": [0.9] * 4 + [0.3] + [0.9] * 5,
            "g3_p_label": [0.7] * 4 + [0.01] + [0.5] * 5,
            "g4_nb_include": [0.9] * 4 + [0.0] + [0.7] * 5,
            "g5_aum": [2.0] * 4 + [-1.0] + [0.5] * 5,
            "p_error": [0.05] * 4 + [0.95] + [0.1] * 5,
            "route": ["clear"] * 5 + ["ambiguous"] * 5,
        }
    )


def test_item_table_summarises_each_item():
    t = explorer.item_table(_scored()).set_index("item_id")
    assert t.loc["a", "route"] == "clear"
    assert t.loc["a", "max_p_error"] == 0.95
    assert t.loc["a", "true_errors"] == 1
    assert t.loc["b", "labels"] == "3 3 5 5 5"


def test_reasons_explain_an_obvious_error():
    row = _scored().iloc[4]
    why = " ".join(explorer.reasons(row))
    for phrase in (
        "lone dissent",
        "unreliable",
        "implausible",
        "never get this label",
        "never learned",
    ):
        assert phrase in why


def test_reasons_explain_genuine_variation():
    why = " ".join(explorer.reasons(_scored().iloc[5]))
    assert "plausible reading" in why and "similar items often" in why


def test_vote_table_sorts_by_suspicion():
    vt = explorer.vote_table(_scored(), "a")
    assert vt.loc[0, "label"] == 9
    assert np.all(np.diff(vt["p_error"]) <= 0)
    assert "why" in vt


def test_streamlit_app_renders_a_scored_run(tmp_path):
    import pytest

    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    _scored().to_parquet(run_dir / "scored_votes.parquet")

    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app" / "explorer.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(tmp_path / "outputs")).run()
    assert not at.exception
    assert at.metric[0].value == "2"
    assert len(at.dataframe) == 1
