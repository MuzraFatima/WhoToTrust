"""G4, neighbourhood consistency: real ambiguity recurs among similar items; errors do not.

For each vote (item *i*, class *y*), look at the 20 nearest items in a frozen embedding space
(DINOv2 for images, a sentence encoder for NLI) and ask how often their votes include *y* too.
The item itself is never its own neighbour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from labeltriage.features import check_inputs


def neighbour_index(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's ``k`` nearest other rows by cosine similarity."""
    x = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    try:
        import faiss

        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x.astype(np.float32))
        _, nn = index.search(x.astype(np.float32), k + 1)
    except ImportError:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(x).kneighbors(x)[1]
    # Drop self, wherever the search put it (duplicates can outrank it).
    own = np.arange(len(x))[:, None]
    keep = nn != own
    return np.stack([row[m][:k] for row, m in zip(nn, keep, strict=True)])


def compute(
    votes: pd.DataFrame, num_classes: int, embeddings: pd.DataFrame, k: int = 20
) -> pd.DataFrame:
    """``embeddings`` is indexed by ``item_id`` and may cover more items than ``votes``.

    Only items that have votes in ``votes`` act as neighbours, so features depend on the votes
    a method actually sees.
    """
    v = check_inputs(votes)
    voted = pd.Index(v["item_id"].unique())
    emb = embeddings.loc[voted].to_numpy(dtype=np.float32)
    nn = neighbour_index(emb, min(k, len(voted) - 1))

    codes = voted.get_indexer(v["item_id"])
    labels = v["label"].to_numpy()
    counts = np.zeros((len(voted), num_classes))
    np.add.at(counts, (codes, labels), 1)
    dist = counts / counts.sum(axis=1, keepdims=True)

    nb_has = (counts[nn] > 0)[np.arange(len(voted))[codes], :, labels]  # (votes, k)
    nb_share = dist[nn][np.arange(len(voted))[codes], :, labels]
    nb_split = ((counts[nn] > 0).sum(axis=2) > 1).mean(axis=1)[codes]

    return pd.DataFrame(
        {
            "g4_nb_include": nb_has.mean(axis=1),
            "g4_nb_share": nb_share.mean(axis=1),
            "g4_nb_split_rate": nb_split,
            "g4_own_vs_nb": dist[codes, labels] - nb_share.mean(axis=1),
        }
    )
