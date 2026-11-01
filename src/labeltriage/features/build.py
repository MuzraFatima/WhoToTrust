"""Assemble the full disagreement-pattern feature table for a vote table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from labeltriage.features import (
    check_inputs,
    g1_votes,
    g2_reliability,
    g3_plausibility,
    g4_neighbours,
    g5_dynamics,
)

GROUPS = ("g1", "g2", "g3", "g4", "g5")


@dataclass
class FeatureInputs:
    """Cached per-item arrays from frozen models, each indexed by ``item_id``."""

    plausibility: pd.DataFrame | None = None  # CLIP zero-shot / MNLI class probabilities (G3)
    embeddings: pd.DataFrame | None = None  # DINOv2 / sentence embeddings (G4 and G5)


def build(
    votes: pd.DataFrame,
    num_classes: int,
    inputs: FeatureInputs,
    groups=GROUPS,
    seed: int = 0,
    probe_epochs: int = 15,
) -> pd.DataFrame:
    """One row per vote, aligned with ``votes`` after ``reset_index``."""
    v = check_inputs(votes)
    parts = []
    for g in groups:
        if g == "g1":
            parts.append(g1_votes.compute(v, num_classes))
        elif g == "g2":
            parts.append(g2_reliability.compute(v, num_classes, seed=seed))
        elif g == "g3":
            parts.append(g3_plausibility.compute(v, _need(inputs.plausibility, g)))
        elif g == "g4":
            parts.append(g4_neighbours.compute(v, num_classes, _need(inputs.embeddings, g)))
        elif g == "g5":
            parts.append(
                g5_dynamics.compute(
                    v, num_classes, _need(inputs.embeddings, g), epochs=probe_epochs, seed=seed
                )
            )
        else:
            raise KeyError(f"unknown feature group {g!r}")
    return pd.concat(parts, axis=1)


def _need(table: pd.DataFrame | None, group: str) -> pd.DataFrame:
    if table is None:
        raise ValueError(f"feature group {group} needs cached model outputs; run the embed task")
    return table


def columns_of(features: pd.DataFrame, group: str) -> list[str]:
    return [c for c in features.columns if c.startswith(f"{group}_")]
