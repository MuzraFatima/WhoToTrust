import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from labeltriage.data.splits import splits_dir
from labeltriage.eval import hypotheses as H


@pytest.fixture
def hyp():
    return OmegaConf.to_container(
        OmegaConf.load(splits_dir().parent / "prereg" / "hypotheses.yaml")
    )


def _runs(lt_gain=0.02, seeds=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for etype in ("spammer", "swap", "lazy", "plausible_wrong"):
        for rate in (0.1, 0.2, 0.4):
            base = 0.8 - rate / 4
            for s in range(seeds):
                noise = rng.normal(0, 0.001)
                accs = {
                    "majority_vote": base + noise,
                    "soft_labels": base - 0.005 + noise,
                    "labeltriage": base + lt_gain + noise + rng.normal(0, 0.002),
                }
                kls = {"majority_vote": 1.0, "soft_labels": 0.5, "labeltriage": 0.52}
                for m in accs:
                    rows.append(
                        {
                            "data": f"ambinoise_10h_k5_r{rate}_{etype}",
                            "method": m,
                            "seed": s,
                            "clean_test/acc": accs[m],
                            "human_test/kl": kls[m] + rng.normal(0, 0.01),
                            "triage/val_auroc": 0.9 if m == "labeltriage" else np.nan,
                            "triage/val_entropy_only": 0.7 if m == "labeltriage" else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def test_settings_parse_from_run_names():
    r = H.with_settings(pd.DataFrame({"data": ["ambinoise_10h_k5_r0.2_plausible_wrong"]}))
    assert r.loc[0, ["base", "k", "rate", "etype"]].tolist() == ["10h", 5, 0.2, "plausible_wrong"]


def test_h2_holds_with_a_clear_gain(hyp):
    res = H.h2(_runs(0.02), hyp)
    assert res["verdict"] == "holds"
    assert len(res["table"]) == 12
    assert (res["table"]["best_global"] == "majority_vote").all()


def test_h2_falsified_without_a_gain(hyp):
    assert H.h2(_runs(0.0), hyp)["verdict"] in {"falsified", "not supported"}


def test_h4_and_h1(hyp):
    runs = _runs()
    assert H.h4(runs, hyp)["verdict"] == "holds"
    transfer = {
        "transfer/test_auroc": 0.75,
        "transfer/test_gain_over_minority_lo": -0.1,
        "transfer/test_gain_over_minority_hi": 0.02,
    }
    res = H.h1(runs, transfer, hyp)
    assert res["gain_over_entropy"] == pytest.approx(0.2)
    assert res["verdict"] == "not supported"  # fails A9's minority-share clause


def test_h3_and_holm(hyp):
    # Like the real design: 4 settings x 4 seeds = 16 pairs. (With 5 pairs a two-sided
    # Wilcoxon cannot go below p = 0.0625 however consistent the effect is.)
    rows = []
    for d in range(4):
        for s in range(4):
            for m, acc, ece in [
                ("neither", 0.80, 0.050),
                ("errors_removed", 0.82, 0.048),
                ("ambiguity_collapsed", 0.80, 0.070 + 0.001 * (s + d)),
            ]:
                rows.append(
                    {
                        "data": f"d{d}",
                        "seed": s,
                        "method": m,
                        "clean_test/acc": acc,
                        "clean_test/ece": ece,
                    }
                )
    res = H.h3(pd.DataFrame(rows), hyp)
    assert res["verdict"] == "holds"
    fam = H.holm_family({"H3": res, "H2": H.h2(_runs(), hyp)})
    assert set(fam) == {"H2", "H3"}


def test_analyze_script_ignores_non_final_runs(tmp_path):
    import importlib.util

    from labeltriage.utils.runs import write_result

    spec = importlib.util.spec_from_file_location("analyze", "scripts/analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for i, row in _runs().iterrows():
        metrics = {k: v for k, v in row.items() if "/" in k and v == v}
        for stage in ("final", "dev"):
            cfg = {
                "experiment": {"name": "main"},
                "data": {"name": row["data"]},
                "method": {"name": row["method"]},
                "seed": int(row["seed"]),
                "stage": stage,
            }
            bump = 0.0 if stage == "final" else -0.5  # dev runs would wreck H2 if counted
            write_result(
                tmp_path / stage / str(i),
                {
                    "config": cfg,
                    "metrics": {**metrics, "clean_test/acc": row["clean_test/acc"] + bump},
                },
            )
    out = tmp_path / "report.md"
    assert mod.main([str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text()
    assert "| H2 | **holds**" in text
