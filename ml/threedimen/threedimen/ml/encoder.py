"""Part 7: SurfaceEncoder interface.

Deliberately framework-agnostic (numpy in/out, not torch.Tensor) so this
doesn't lock in a neural representation before one's been chosen --
point-based, mesh-token, graph, implicit, latent, and hybrid representations
are all still open. PointSampleEncoder below is a minimal, real, testable
implementation (not a stub) so the interface has at least one working
example rather than only an ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from ..geometry.representation import SurfaceRepresentation


@dataclass
class EncodedSurface:
    points: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3)
    metadata: dict[str, Any] = field(default_factory=dict)


class SurfaceEncoder(ABC):
    @abstractmethod
    def encode(self, surface: SurfaceRepresentation) -> EncodedSurface: ...


class PointSampleEncoder(SurfaceEncoder):
    """Samples num_points points+normals from the merged mesh surface -- the
    simplest possible framework-agnostic feature representation. A
    placeholder for experimentation, not a claim that this is the final
    architecture (see Part 7 of the roadmap doc for the alternatives this is
    meant to be swappable with)."""

    def __init__(self, num_points: int = 2048, seed: int | None = None):
        self.num_points = num_points
        self.seed = seed

    def encode(self, surface: SurfaceRepresentation) -> EncodedSurface:
        mesh = surface.to_trimesh()
        rng = np.random.default_rng(self.seed)
        points, face_indices = trimesh.sample.sample_surface(mesh, self.num_points, seed=rng)
        normals = mesh.face_normals[face_indices]
        return EncodedSurface(
            points=np.asarray(points),
            normals=np.asarray(normals),
            metadata={"encoder": "point_sample", "num_points": self.num_points},
        )
