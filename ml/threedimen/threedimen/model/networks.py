"""nn.Module definitions for the text-to-shape proof-of-concept model.

Deliberately minimal, matched to the amount of training data this project
actually has (tens-to-hundreds of samples), not a scaled-down version of a
production architecture:

  - TextEncoder: a bag-of-embeddings encoder (embed each token, mean-pool)
    over a vocabulary built from the training set itself. No pretrained
    language model -- would be overkill for this much data.
  - PointCloudEncoder: a mini-PointNet (shared per-point MLP, then a
    permutation-invariant max-pool over points) -- the standard minimal
    architecture for encoding an unordered point cloud into a fixed-size
    vector.
  - PointCloudDecoder: an MLP mapping a latent vector to a fixed-size point
    cloud in one shot. The simplest point-cloud decoder that exists; no
    folding/upsampling machinery, which would again be more than this data
    scale can support.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, latent_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.project = nn.Sequential(
            nn.Linear(embed_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids != 0).float().unsqueeze(-1)  # (B, L, 1)
        embedded = self.embedding(token_ids) * mask
        pooled = embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.project(pooled)


class PointCloudEncoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        per_point = self.point_mlp(points)  # (B, N, latent_dim)
        pooled, _ = per_point.max(dim=1)
        return pooled


class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_points: int, hidden_dim: int):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(latent).view(-1, self.num_points, 3)
