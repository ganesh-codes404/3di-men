"""Leakage-safe train/val/test split for the synthetic prompt dataset.

Splitting naively (random row shuffle) would let two different renderings of
the SAME underlying object -- e.g. two paraphrases we deliberately generate
on purpose, or two independently-sampled objects that happen to land on the
same discrete-grid point (see prompt_dataset.py's samplers) -- span multiple
splits. That leaks the exact answer for a test-set object into training.
Instead, every sample is bucketed by a hash of its (part_family, resolved
params) -- its `canonical_key` -- so every rendering of the same object
always lands in the same split, deterministically and reproducibly.
"""

from __future__ import annotations

import hashlib
from typing import Literal

Split = Literal["train", "val", "test"]


def assign_split(key: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0) -> Split:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    # deterministic pseudo-random bucket in [0, 1), independent of Python's
    # random module state so this is a pure function of (key, ratios, seed)
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    bucket = int(digest[:16], 16) / 0xFFFFFFFFFFFFFFFF

    train_end = ratios[0]
    val_end = train_end + ratios[1]
    if bucket < train_end:
        return "train"
    if bucket < val_end:
        return "val"
    return "test"
