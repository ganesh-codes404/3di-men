"""Training loop for the 3di-men text-to-shape model.

Was originally a CPU-only plumbing validator over ~40-50 samples (see git
history / docs/threedimen-architecture.md's Phase 2 note). With a larger
real-Hunyuan3D dataset (~300-450 samples) and GPU compute (Kaggle) now
available, the defaults below are scaled up accordingly -- more points
sampled per shape, wider/deeper networks, bigger batches, more epochs. This
is still a small point-cloud autoencoder, not a claim of matching
Hunyuan3D-scale generative quality; it's sized to the data actually
available (hundreds, not millions, of examples), not to what the hardware
could technically support.

Architecture: a shape autoencoder (PointCloudEncoder -> PointCloudDecoder,
trained with Chamfer reconstruction loss) plus a TextEncoder trained to
regress toward the shape encoder's latent for the matching prompt -- so at
generation time, a prompt alone can be pushed through TextEncoder ->
PointCloudDecoder without ever needing a real mesh as input.

Device handling: train() takes its own `device` argument (default "auto" ->
CUDA if available, else CPU) rather than storing it in TrainConfig, so a
GPU-trained checkpoint is still just as loadable for CPU inference later --
see device.py's docstring for why that separation matters.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .chamfer import chamfer_distance
from .data import load_training_pairs
from .device import resolve_device
from .networks import PointCloudDecoder, PointCloudEncoder, TextEncoder
from .vocab import Vocabulary


@dataclass
class TrainConfig:
    num_points: int = 2048
    embed_dim: int = 128
    hidden_dim: int = 256
    latent_dim: int = 128
    max_len: int = 32
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    align_weight: float = 1.0
    seed: int = 0


class _PairDataset(Dataset):
    def __init__(self, pairs, vocab: Vocabulary, max_len: int):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        tokens = torch.tensor(self.vocab.encode(pair.prompt, self.max_len), dtype=torch.long)
        points = torch.tensor(pair.points, dtype=torch.float32)
        return tokens, points


def train(
    root: str | Path,
    out_path: str | Path,
    config: TrainConfig | None = None,
    device: str = "auto",
) -> dict:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    resolved_device = resolve_device(device)

    pairs = load_training_pairs(root, num_points=config.num_points, seed=config.seed)
    if len(pairs) < 2:
        raise ValueError(f"need at least 2 dataset samples to train, found {len(pairs)} in {root}")

    vocab = Vocabulary.build([p.prompt for p in pairs])
    dataset = _PairDataset(pairs, vocab, config.max_len)
    # pin_memory speeds up host->GPU transfer; harmless (ignored) on CPU.
    loader = DataLoader(
        dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=True,
        pin_memory=(resolved_device.type == "cuda"),
    )

    text_encoder = TextEncoder(len(vocab), config.embed_dim, config.latent_dim, pad_idx=vocab.pad_id).to(resolved_device)
    shape_encoder = PointCloudEncoder(config.hidden_dim, config.latent_dim).to(resolved_device)
    shape_decoder = PointCloudDecoder(config.latent_dim, config.num_points, config.hidden_dim).to(resolved_device)

    params = list(text_encoder.parameters()) + list(shape_encoder.parameters()) + list(shape_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    t0 = time.perf_counter()
    history: list[dict] = []
    for epoch in range(config.epochs):
        epoch_recon = epoch_align = 0.0
        n_batches = 0
        for tokens, points in loader:
            tokens = tokens.to(resolved_device, non_blocking=True)
            points = points.to(resolved_device, non_blocking=True)
            optimizer.zero_grad()

            shape_latent = shape_encoder(points)
            recon = shape_decoder(shape_latent)
            recon_loss = chamfer_distance(recon, points)

            text_latent = text_encoder(tokens)
            align_loss = F.mse_loss(text_latent, shape_latent.detach())

            loss = recon_loss + config.align_weight * align_loss
            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_align += align_loss.item()
            n_batches += 1
        history.append({"epoch": epoch, "recon_loss": epoch_recon / n_batches, "align_loss": epoch_align / n_batches})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # .state_dict() tensors are saved on whatever device they live on
            # (e.g. CUDA); generate.py loads with map_location="cpu", so a
            # GPU-trained checkpoint stays loadable on a machine with no GPU.
            "text_encoder": text_encoder.state_dict(),
            "shape_encoder": shape_encoder.state_dict(),
            "shape_decoder": shape_decoder.state_dict(),
            "vocab": vocab.to_dict(),
            "config": asdict(config),
        },
        out_path,
    )

    return {
        "checkpoint_path": str(out_path),
        "num_samples": len(pairs),
        "vocab_size": len(vocab),
        "epochs": config.epochs,
        "device": str(resolved_device),
        "final_recon_loss": history[-1]["recon_loss"],
        "final_align_loss": history[-1]["align_loss"],
        "training_time_seconds": time.perf_counter() - t0,
    }
