import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from labeltriage.data import splits
from labeltriage.eval import prereg


def _hyp():
    path = splits.splits_dir().parent / "prereg" / "hypotheses.yaml"
    return OmegaConf.to_container(OmegaConf.load(path))


def test_hypotheses_file_has_all_four_and_a9():
    h = _hyp()
    assert {"H1", "H2", "H3", "H4"} <= set(h)
    assert h["H1"]["gain_over_minority_share_ci_above"] == 0.0


def test_pooled_sd_and_mde_report():
    rng = np.random.default_rng(0)
    rows = []
    for method in ["majority_vote", "soft_labels", "labeltriage"]:
        for seed in range(3):
            rows.append(
                {
                    "data": "d",
                    "method": method,
                    "seed": seed,
                    "clean_dev/acc": 0.8 + rng.normal(0, 0.002),
                    "clean_dev/ece": 0.05 + rng.normal(0, 0.001),
                    "val/kl": 0.5 + rng.normal(0, 0.005),
                }
            )
    runs = pd.DataFrame(rows)
    assert 0.0005 < prereg.pooled_seed_sd(runs, "clean_dev/acc") < 0.005
    rep = prereg.mde_report(runs, _hyp())
    assert rep["H2/detectable"] == 1.0
    assert rep["gate/all_detectable"] == 1.0


def test_preregister_script_writes_record_and_picks_router(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("prereg_script", "scripts/preregister.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(1)
    pilot = pd.DataFrame(
        [
            {
                "data": "d",
                "method": m,
                "seed": s,
                "clean_dev/acc": 0.8 + rng.normal(0, 0.001),
                "clean_dev/ece": 0.05 + rng.normal(0, 0.0005),
                "val/kl": 0.5 + rng.normal(0, 0.002),
            }
            for m in ("majority_vote", "labeltriage")
            for s in range(3)
        ]
    )
    tune = pd.DataFrame(
        {"method.nu": [1.0, 1.0, 2.0], "method.h0": [0.3, 0.5, 0.3], "val/nll": [0.9, 0.7, 0.8]}
    )
    pilot.to_csv(tmp_path / "pilot.csv", index=False)
    tune.to_csv(tmp_path / "tune.csv", index=False)
    out = tmp_path / "prereg.md"
    rc = mod.main(
        [str(tmp_path / "pilot.csv"), "--tune", str(tmp_path / "tune.csv"), "--out", str(out)]
    )
    text = out.read_text()
    assert rc == 0
    assert "h0 = 0.5" in text and "nu = 1.0" in text
