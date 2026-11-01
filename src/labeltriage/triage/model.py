"""The LabelTriage scorer: p(error) per annotation, calibrated (blueprint section 8.3).

A LightGBM classifier (or an MLP, as an ablation) is fitted on train-split votes of an AmbiNoise
draw, where the error labels are known. Isotonic regression on val-split votes then calibrates
its scores, so the router can read them as probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


def calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Binary ECE: confidence-weighted gap between predicted p(error) and the error rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


@dataclass
class TriageModel:
    columns: list[str]
    kind: str = "lgbm"  # "lgbm" or "mlp"
    seed: int = 0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self._model = None
        self._calibrator: IsotonicRegression | None = None

    def _new(self):
        if self.kind == "lgbm":
            return lgb.LGBMClassifier(**{**LGBM_PARAMS, **self.params}, random_state=self.seed)
        if self.kind == "mlp":
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    early_stopping=True,
                    random_state=self.seed,
                    **self.params,
                ),
            )
        raise KeyError(f"unknown triage model {self.kind!r}")

    def raw_score(self, features: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(features[self.columns].to_numpy(dtype=float))[:, 1]

    def fit(self, features: pd.DataFrame, is_error: np.ndarray) -> TriageModel:
        self._model = self._new()
        self._model.fit(features[self.columns].to_numpy(dtype=float), np.asarray(is_error, int))
        return self

    def calibrate(self, features: pd.DataFrame, is_error: np.ndarray) -> TriageModel:
        self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._calibrator.fit(self.raw_score(features), np.asarray(is_error, float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated p(error) per vote (raw scores if never calibrated)."""
        s = self.raw_score(features)
        return self._calibrator.predict(s) if self._calibrator is not None else s

    def importance(self) -> pd.Series | None:
        if self.kind != "lgbm":
            return None
        gain = self._model.booster_.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=self.columns).sort_values(ascending=False)
