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
