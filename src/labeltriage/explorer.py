"""Data behind the Disagreement Explorer: per-item summaries and plain-language reasons.

Kept free of any UI code so it is testable; ``app/explorer.py`` draws it with Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROUTE_COLOURS = {"clear": "#2a9d8f", "ambiguous": "#e9c46a", "unreliable": "#e76f51"}


def item_table(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per item: how its votes split, how suspicious its votes look, where it went."""
    g = scored.groupby("item_id")
    out = pd.DataFrame(
        {
            "entropy": g["g1_entropy"].first(),
            "max_p_error": g["p_error"].max(),
            "mean_p_error": g["p_error"].mean(),
            "route": g["route"].first(),
            "n_votes": g.size(),
            "labels": g["label"].agg(lambda s: " ".join(map(str, sorted(s)))),
        }
    )
    if "is_error" in scored:
        out["true_errors"] = g["is_error"].sum().astype(int)
    return out.reset_index()


def reasons(row: pd.Series) -> list[str]:
    """Short, human-readable reasons behind one vote's error score."""
    out = []
    if row.get("g1_share", 1.0) <= 0.2 + 1e-9:
        out.append("lone dissent among the votes")
    elif row.get("g1_share", 1.0) < 0.5:
        out.append(f"minority vote ({row['g1_share']:.0%} of the item)")
    acc = row.get("g2_ds_accuracy")
    if acc is not None and not np.isnan(acc) and acc < 0.6:
        out.append(f"annotator is unreliable elsewhere (estimated accuracy {acc:.0%})")
    p = row.get("g3_p_label")
    if p is not None and not np.isnan(p):
        if p < 0.05:
            out.append(f"label looks implausible to a frozen model (p = {p:.2f})")
        elif p > 0.3:
            out.append(f"label is a plausible reading of the item (p = {p:.2f})")
    nb = row.get("g4_nb_include")
    if nb is not None and not np.isnan(nb):
        if nb < 0.1:
            out.append("similar items almost never get this label")
        elif nb > 0.5:
            out.append(f"similar items often get this label too ({nb:.0%})")
    aum = row.get("g5_aum")
    if aum is not None and not np.isnan(aum) and aum < 0:
        out.append("a probe model never learned this label for this item")
    return out or ["nothing stands out"]


def vote_table(scored: pd.DataFrame, item_id: str) -> pd.DataFrame:
    votes = scored[scored["item_id"] == item_id].sort_values("p_error", ascending=False)
    cols = [c for c in ("annotator_id", "label", "p_error", "is_error", "error_type") if c in votes]
    out = votes[cols].copy()
    out["why"] = ["; ".join(reasons(r)) for _, r in votes.iterrows()]
    return out.reset_index(drop=True)
