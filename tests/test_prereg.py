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
