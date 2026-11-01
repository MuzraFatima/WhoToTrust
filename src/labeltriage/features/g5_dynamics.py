"""G5, training dynamics: errors are learned late or never (amendment A4).

A small MLP probe on frozen embeddings is trained with every vote as its own example. Five probes
are cross-fitted by item fold. Each vote is therefore trained on by four probes, which supply the
data-map statistics (Swayamdipta et al., 2020) and the area under the margin (Pleiss et al.,
2020), and held out by one, which supplies an out-of-fold probability for its label.

The margin is computed from log-probabilities, which equals the logit margin because softmax is
shift-invariant.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from labeltriage import crossfit
from labeltriage.features import check_inputs

EPS = 1e-9


def _margin(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    own = logp[np.arange(len(y)), y]
    other = logp.copy()
    other[np.arange(len(y)), y] = -np.inf
    return own - other.max(axis=1)


def compute(
    votes: pd.DataFrame,
    num_classes: int,
    embeddings: pd.DataFrame,
    epochs: int = 15,
    hidden: int = 256,
    lr: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    v = check_inputs(votes)
    x_all = StandardScaler().fit_transform(embeddings.loc[v["item_id"]].to_numpy(dtype=np.float32))
    y = v["label"].to_numpy()
    classes = np.arange(num_classes)

    p_sum = np.zeros(len(v))
    p_sq = np.zeros(len(v))
    margin_sum = np.zeros(len(v))
    n_obs = np.zeros(len(v))
    oof = np.zeros(len(v))
    oof_margin = np.zeros(len(v))

    for f, train, held in crossfit.splits(v["item_id"].to_numpy(), n_folds, seed):
        probe = MLPClassifier(
            hidden_layer_sizes=(hidden,),
            learning_rate_init=lr,
            batch_size=256,
            random_state=seed * 100 + f,
        )
        rng = np.random.default_rng(seed * 100 + f)
        tr = np.flatnonzero(train)
        for _ in range(epochs):
            order = rng.permutation(tr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                probe.partial_fit(x_all[order], y[order], classes=classes)
            logp = np.log(probe.predict_proba(x_all[tr]) + EPS)
            p = np.exp(logp[np.arange(len(tr)), y[tr]])
            p_sum[tr] += p
            p_sq[tr] += p**2
            margin_sum[tr] += _margin(logp, y[tr])
            n_obs[tr] += 1

        logp_held = np.log(probe.predict_proba(x_all[held]) + EPS)
        oof[held] = np.exp(logp_held[np.arange(held.sum()), y[held]])
        oof_margin[held] = _margin(logp_held, y[held])

    confidence = p_sum / n_obs
    variability = np.sqrt(np.maximum(p_sq / n_obs - confidence**2, 0.0))
    return pd.DataFrame(
        {
            "g5_confidence": confidence,
            "g5_variability": variability,
            "g5_aum": margin_sum / n_obs,
            "g5_oof_prob": oof,
            "g5_oof_margin": oof_margin,
        }
    )
