"""Prompt -> SurfaceRepresentation using a checkpoint from train.py.

Reuses Part 7's ConvexHullDecoder to turn the predicted point cloud into an
actual SurfaceRepresentation, rather than duplicating that conversion here.

device="auto" here is independent of whatever device the checkpoint was
*trained* on -- torch.load always reads onto CPU first (map_location="cpu"),
then the resolved device moves the model if requested/available. A
GPU-trained checkpoint therefore still generates correctly on a CPU-only
machine (just not accelerated), and vice versa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..geometry.representation import SurfaceRepresentation
from ..ml.decoder import ConvexHullDecoder
from ..ml.encoder import EncodedSurface
from .device import resolve_device
from .networks import PointCloudDecoder, TextEncoder
from .train import TrainConfig
from .vocab import Vocabulary


def generate(prompt: str, checkpoint_path: str | Path, device: str = "auto") -> SurfaceRepresentation:
    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    vocab = Vocabulary.from_dict(checkpoint["vocab"])

    text_encoder = TextEncoder(len(vocab), config.embed_dim, config.latent_dim, pad_idx=vocab.pad_id)
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    text_encoder = text_encoder.to(resolved_device).eval()

    shape_decoder = PointCloudDecoder(config.latent_dim, config.num_points, config.hidden_dim)
    shape_decoder.load_state_dict(checkpoint["shape_decoder"])
    shape_decoder = shape_decoder.to(resolved_device).eval()

    with torch.no_grad():
        tokens = torch.tensor([vocab.encode(prompt, config.max_len)], dtype=torch.long, device=resolved_device)
        latent = text_encoder(tokens)
        points = shape_decoder(latent)[0].cpu().numpy()

    # No real surface normals exist for a freshly-predicted point cloud;
    # radial-outward is a reasonable placeholder for ConvexHullDecoder's
    # purposes (it only uses the points, but EncodedSurface requires normals).
    radial = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-9)
    encoded = EncodedSurface(
        points=points, normals=radial, metadata={"model": "threedimen_pointcloud_poc", "prompt": prompt}
    )
    return ConvexHullDecoder().decode(encoded)
