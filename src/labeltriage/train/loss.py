from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_soft_ce(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean over the batch of ``w_i * CE(q_i, softmax(logits_i))``.

    With one-hot targets and unit weights this is ordinary cross-entropy. A weight of zero removes
    an item from the gradient; down-weighted items count proportionally less.
    """
    per_item = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    if weights is not None:
        per_item = per_item * weights
    return per_item.mean()
