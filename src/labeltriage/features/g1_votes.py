"""G1, vote statistics: how the item's votes split, and where this vote sits in that split.

A lone dissent looks different from an even split, which looks different again from a vote
for a class nobody else chose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeltriage.features import check_inputs
from labeltriage.targets import entropy


def compute(votes: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    v = check_inputs(votes)
    codes, item_ids = pd.factorize(v["item_id"])
    labels = v["label"].to_numpy()
    counts = np.zeros((len(item_ids), num_classes))
    np.add.at(counts, (codes, labels), 1)
    k = counts.sum(axis=1)
    dist = counts / k[:, None]

    own = counts[codes, labels]
    # Rank 0 is the most-voted class; ties share the better rank.
    rank = (counts[codes] > own[:, None]).sum(axis=1)
    top = counts.max(axis=1)[codes]

    return pd.DataFrame(
        {
            "g1_entropy": entropy(dist)[codes],
            "g1_share": own / k[codes],
            "g1_share_others": (own - 1) / np.maximum(k[codes] - 1, 1),
            "g1_rank": rank,
            "g1_gap_to_top": (top - own) / k[codes],
            "g1_alone": (own == 1).astype(float),
            "g1_n_classes": (counts > 0).sum(axis=1)[codes],
            "g1_k": k[codes],
        }
    )
