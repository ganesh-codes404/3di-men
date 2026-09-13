"""Device resolution shared by train.py and generate.py.

"auto" picks CUDA if available, else CPU. This is what makes the same code
correctly use a GPU when run somewhere that has one (a Kaggle GPU notebook,
a rented GPU box) while still working unchanged on this project's GPU-less
development laptop.

Deliberately NOT part of TrainConfig / the saved checkpoint: which device to
run *this process* on is an execution-time choice, not a property of the
trained model. A model trained on GPU must still be loadable for CPU
inference later (e.g. back on a laptop for a quick check) without carrying
a stale "cuda" setting forward from training time -- so train() and
generate() each take their own independent `device` argument instead.
"""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} requested but no CUDA GPU is available on this machine")
    return resolved
