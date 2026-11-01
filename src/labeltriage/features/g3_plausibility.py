"""G3, semantic plausibility: is the chosen class a believable reading of the item?

Dog versus wolf is plausible; dog versus truck is not. Plausibility comes from a frozen model that
never trains on these votes: CLIP zero-shot for images, a public MNLI model for NLI. The input is
a per-item probability table indexed by ``item_id``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeltriage.features import check_inputs

EPS = 1e-6


def compute(votes: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    v = check_inputs(votes)
    rows = probs.index.get_indexer(v["item_id"])
    if (rows < 0).any():
        raise KeyError(f"plausibility table is missing {int((rows < 0).sum())} votes' items")
    p = probs.to_numpy(dtype=float)[rows]
    y = v["label"].to_numpy()
    p_y = p[np.arange(len(v)), y]

    # Compare with the class the item's voters favoured most.
    codes, _ = pd.factorize(v["item_id"])
    counts = np.zeros((codes.max() + 1, p.shape[1]))
    np.add.at(counts, (codes, y), 1)
    p_major = p[np.arange(len(v)), counts.argmax(axis=1)[codes]]

    return pd.DataFrame(
        {
            "g3_p_label": p_y,
            "g3_log_ratio_to_majority": np.log(p_y + EPS) - np.log(p_major + EPS),
            "g3_rank": (p > p_y[:, None]).sum(axis=1),
            "g3_model_entropy": -(p * np.log(p + EPS)).sum(axis=1),
        }
    )
