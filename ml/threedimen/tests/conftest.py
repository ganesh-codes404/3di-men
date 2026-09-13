"""Shared fixtures. Every fixture mesh is generated procedurally (trimesh
primitives) rather than checked-in binary files -- no test depends on
Hunyuan3D, the network, or a GPU."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh


@pytest.fixture
def box_mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[2.0, 3.0, 4.0])


@pytest.fixture
def sphere_mesh() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(radius=1.5, subdivisions=2)


@pytest.fixture
def empty_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))


@pytest.fixture
def nan_mesh() -> trimesh.Trimesh:
    mesh = trimesh.creation.box()
    mesh.vertices[0] = [np.nan, 0.0, 0.0]
    return mesh


@pytest.fixture
def degenerate_face_mesh() -> trimesh.Trimesh:
    """A box with one extra zero-area (degenerate) triangle appended --
    two of its three corner indices are the same vertex."""
    base = trimesh.creation.box()
    faces = np.vstack([base.faces, [0, 0, 1]])
    return trimesh.Trimesh(vertices=base.vertices, faces=faces, process=False)
