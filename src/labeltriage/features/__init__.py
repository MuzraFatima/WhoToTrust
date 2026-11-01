"""Disagreement-pattern features, one row per annotation (blueprint section 8.2).

Every group returns a DataFrame aligned row-for-row with the vote table it was given, with
columns prefixed by its group name (``g1_``, ``g2_``, ...). No group ever reads ``is_error``,
``error_type``, ``original_label`` or ``annotator_propensity``; :data:`FORBIDDEN` lists them and
:func:`check_inputs` enforces it.
"""

from __future__ import annotations

import pandas as pd

FORBIDDEN = ("is_error", "error_type", "original_label", "annotator_propensity")


def check_inputs(votes: pd.DataFrame) -> pd.DataFrame:
    """Strip answer-bearing columns so a feature group cannot read them even by accident."""
    return votes.drop(columns=[c for c in FORBIDDEN if c in votes.columns]).reset_index(drop=True)
