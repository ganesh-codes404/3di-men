"""Part 7: SurfaceDecoder interface.

ConvexHullDecoder below reconstructs only the convex hull of a point cloud --
it loses all concavity and exists purely to prove the encode -> decode round
trip through this interface works end to end before any real model exists.
A trained decoder replaces it later without changing any caller, since
callers only ever depend on SurfaceDecoder.decode().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import trimesh

from ..geometry.representation import SurfaceRepresentation
from .encoder import EncodedSurface


class SurfaceDecoder(ABC):
    @abstractmethod
    def decode(self, encoded: EncodedSurface) -> SurfaceRepresentation: ...


class ConvexHullDecoder(SurfaceDecoder):
    def decode(self, encoded: EncodedSurface) -> SurfaceRepresentation:
        hull = trimesh.Trimesh(vertices=encoded.points, process=True).convex_hull
        return SurfaceRepresentation.from_trimesh(hull, decoder="convex_hull_placeholder")
