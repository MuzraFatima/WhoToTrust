"""G2, annotator reliability: errors cluster in annotators who are unreliable elsewhere.

Every statistic about annotator *a* used for a vote on item *i* is computed from votes on items
in other folds, so the vote being scored never informs its own annotator's reliability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeltriage import crossfit
from labeltriage.crowd import dawid_skene
from labeltriage.features import check_inputs


def _loo_agreement(votes: pd.DataFrame, num_classes: int, min_share: float) -> pd.DataFrame:
    """Per vote: does it match the other votes' majority, and is that majority strong?"""
    codes, _ = pd.factorize(votes["item_id"])
    labels = votes["label"].to_numpy()
    counts = np.zeros((codes.max() + 1, num_classes))
    np.add.at(counts, (codes, labels), 1)
    others = counts[codes].copy()
    others[np.arange(len(votes)), labels] -= 1
    n_others = others.sum(axis=1)
    top = others.argmax(axis=1)
    strong = (n_others > 0) & (others.max(axis=1) >= min_share * np.maximum(n_others, 1))
    return pd.DataFrame({"agree": (top == labels).astype(float), "strong": strong})


def compute(
    votes: pd.DataFrame,
    num_classes: int,
    n_folds: int = 5,
    seed: int = 0,
    min_share: float = 0.8,
) -> pd.DataFrame:
    v = check_inputs(votes)
    loo = _loo_agreement(v, num_classes, min_share)
    folds = crossfit.fold_ids(v["item_id"].to_numpy(), n_folds, seed)
    ann = v["annotator_id"].to_numpy()

    out = pd.DataFrame(
        index=v.index,
        columns=["g2_ds_accuracy", "g2_ds_own_prob", "g2_consensus_agreement", "g2_n_votes"],
        dtype=float,
    )
    for f in range(n_folds):
        fit, score = folds != f, folds == f

        ds = dawid_skene(v[fit], num_classes)
        acc = ds.annotator_accuracy
        # Unseen annotators get the mean accuracy of seen ones.
        out.loc[score, "g2_ds_accuracy"] = (
            pd.Series(ann[score]).map(acc).fillna(acc.mean()).to_numpy()
        )

        # How often this annotator answers y when y is the true class.
        a_pos = pd.Index(ds.annotator_ids).get_indexer(ann[score])
        y = v.loc[score, "label"].to_numpy()
        own = np.where(a_pos >= 0, ds.confusion[np.maximum(a_pos, 0), y, y], np.nan)
        out.loc[score, "g2_ds_own_prob"] = np.where(np.isnan(own), np.nanmean(own), own)

        fit_strong = fit & loo["strong"].to_numpy()
        agree = pd.Series(loo["agree"].to_numpy()[fit_strong]).groupby(ann[fit_strong]).mean()
        out.loc[score, "g2_consensus_agreement"] = (
            pd.Series(ann[score]).map(agree).fillna(agree.mean()).to_numpy()
        )
        n = pd.Series(ann[fit]).value_counts()
        out.loc[score, "g2_n_votes"] = pd.Series(ann[score]).map(n).fillna(0).to_numpy()

    return out.astype(float)
