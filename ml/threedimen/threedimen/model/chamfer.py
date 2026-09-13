"""Symmetric Chamfer distance between two batched point sets -- the standard
reconstruction loss for point-cloud output, since point clouds have no
canonical ordering (a plain per-point MSE would penalize correct shapes that
happen to enumerate their points differently)."""

from __future__ import annotations

import torch


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred: (B, N, 3), target: (B, M, 3). Returns a scalar (mean over batch)."""
    dist = torch.cdist(pred, target)  # (B, N, M)
    pred_to_target = dist.min(dim=2).values.mean(dim=1)  # (B,)
    target_to_pred = dist.min(dim=1).values.mean(dim=1)  # (B,)
    return (pred_to_target + target_to_pred).mean()
