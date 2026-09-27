"""Builds a complete assembly from an LLM-decomposed list of actionable
points, each one independently composed via program_model.py's tool-
sequence prediction + sandbox.py's execution -- built IN PARALLEL, one OS
process per point, since each point is fully independent geometry work with
no shared state. This is 3di-men's own proprietary builder end to end: the
LLM (llm_decompose.py) only ever produces plain-language descriptions, never
geometry, tool calls, or CAD parameters. No part-family concept anywhere in
this module -- see part_families.py's module docstring for why that
registry still exists elsewhere (internal training-corpus scaffolding
only), never as a live classification step here.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from llm_decompose import ActionablePoint, decompose_prompt


@dataclass
class PlacementTransform:
    """Moved verbatim from the now-deleted assembly.py -- a plain 4x4-matrix
    wrapper with zero part-family logic, so it's reused as-is rather than
    rewritten."""

    matrix: np.ndarray

    @classmethod
    def identity(cls) -> "PlacementTransform":
        return cls(np.eye(4))

    @classmethod
    def translation(cls, xyz) -> "PlacementTransform":
        m = np.eye(4)
        m[:3, 3] = xyz
        return cls(m)

    def compose(self, other: "PlacementTransform") -> "PlacementTransform":
        return PlacementTransform(self.matrix @ other.matrix)

    def apply_to_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        result = mesh.copy()
        result.apply_transform(self.matrix)
        return result


@dataclass
class PointBuildResult:
    point: ActionablePoint
    status: str  # "ok" | "rejected" | "partial" | "suspect" | "out_of_distribution"
    executed_steps: int
    error: str | None
    nearest_neighbor_similarity: float
    mesh: "trimesh.Trimesh | None" = field(repr=False, default=None)
    topology: "dict | None" = None


@dataclass
class AssemblyBuildResult:
    mesh: "trimesh.Trimesh | None"
    points: list[PointBuildResult]
    fell_back_to_single_point: bool  # true if the LLM decomposition itself failed (see llm_decompose.DecomposeResult)


_worker_bundle: Any = None  # per-process cache -- loaded once per worker, not once per point


def _init_worker(bundle_path: str) -> None:
    global _worker_bundle
    from program_model import ProgramModelBundle

    _worker_bundle = ProgramModelBundle.load(Path(bundle_path))


def _build_one(description: str) -> dict:
    """Runs in a worker process. Returns a plain dict (not a dataclass with
    a trimesh payload attached) plus the mesh separately as exported bytes,
    since ProcessPoolExecutor results must round-trip through pickling --
    trimesh.Trimesh pickles fine, but keeping the boundary explicit here
    documents that this crosses a process boundary."""
    from program_model import predict_program_checked
    from sandbox import Sandbox

    prediction = predict_program_checked(_worker_bundle, description)
    result = Sandbox.execute(prediction.program)

    if not prediction.in_distribution:
        status = "out_of_distribution"
    else:
        status = result.status

    topology = None
    if result.mesh is not None:
        from mesh_quality import mesh_topology_stats

        topology = mesh_topology_stats(result.mesh)

    return {
        "status": status,
        "executed_steps": result.executed_steps,
        "error": result.error,
        "nearest_neighbor_similarity": prediction.nearest_neighbor_similarity,
        "mesh": result.mesh,
        "topology": topology,
    }


def build_assembly_from_points(points: list[ActionablePoint], bundle_path: Path, max_workers: int = 4) -> AssemblyBuildResult:
    """The shared build core: takes a list of actionable points from
    WHATEVER source produced them -- the LLM's decomposition, or a user
    hand-editing/hand-adding points in the frontend's playground -- and
    builds each one independently in parallel via program_model.py +
    sandbox.py, exactly the same way either way. Manual points are not a
    separate, lesser path: they go through the identical proprietary
    tool-composing builder, just skipping the LLM's guess at what the
    parts and their positions should be."""
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(str(bundle_path),)) as pool:
        raw_results = list(pool.map(_build_one, [p.description for p in points]))

    point_results: list[PointBuildResult] = []
    placed_meshes: list[trimesh.Trimesh] = []
    for point, raw in zip(points, raw_results):
        point_results.append(
            PointBuildResult(
                point=point, status=raw["status"], executed_steps=raw["executed_steps"], error=raw["error"],
                nearest_neighbor_similarity=raw["nearest_neighbor_similarity"], mesh=raw["mesh"],
                topology=raw["topology"],
            )
        )
        # "out_of_distribution" and "suspect" still carry a mesh (a guess,
        # or a geometrically dubious result) -- included in the combined
        # assembly but disclosed via status, never silently upgraded to
        # look like a clean "ok". Only a genuinely missing mesh (rejected/
        # partial-with-no-mesh) is left out, and even then the point is
        # still reported, never dropped from the response entirely.
        if raw["mesh"] is not None:
            transform = PlacementTransform.translation(point.position_mm)
            placed_meshes.append(transform.apply_to_mesh(raw["mesh"]))

    combined = trimesh.util.concatenate(placed_meshes) if placed_meshes else None
    return AssemblyBuildResult(mesh=combined, points=point_results, fell_back_to_single_point=False)


def build_assembly_from_prompt(prompt: str, bundle_path: Path, max_workers: int = 4) -> AssemblyBuildResult:
    decomposition = decompose_prompt(prompt)
    result = build_assembly_from_points(decomposition.points, bundle_path, max_workers)
    result.fell_back_to_single_point = decomposition.fell_back
    return result


@dataclass
class EntityBuildResult:
    name: str
    status: str  # "ok" | "rejected" | "partial" | "suspect"
    executed_steps: int
    error: str | None
    mesh: "trimesh.Trimesh | None" = field(repr=False, default=None)
    topology: "dict | None" = None


@dataclass
class RawAssemblyBuildResult:
    mesh: "trimesh.Trimesh | None"
    entities: list[EntityBuildResult]


def build_assembly_from_programs(
    entities: list[tuple[str, list, tuple[float, float, float]]],
) -> RawAssemblyBuildResult:
    """The manual tool console's counterpart to build_assembly_from_points --
    same combine-by-translation logic, but for a user who has hand-authored
    each entity's own raw Sandbox program directly (via the console's
    form/gizmo/click-drag tools), not a text description for program_model
    to predict from. Each entry is (name, program, position_mm); `program`
    is expected to already end in its own "finalize" step, exactly like a
    single-entity /api/execute_program call.

    Deliberately synchronous / single-process, unlike
    build_assembly_from_points' ProcessPoolExecutor: a manual console
    session builds a handful of entities at a time (not hundreds), each
    already fast (single-digit seconds), and this sidesteps re-introducing
    macOS spawn's "re-executes the whole __main__ script in every worker"
    surface (see server.py's main() docstring for the real
    BrokenProcessPool incident that caused) for a path that doesn't need
    the throughput."""
    from sandbox import Sandbox

    results: list[EntityBuildResult] = []
    placed_meshes: list[trimesh.Trimesh] = []
    for name, program, position_mm in entities:
        exec_result = Sandbox.execute(program)
        topology = None
        if exec_result.mesh is not None:
            from mesh_quality import mesh_topology_stats

            topology = mesh_topology_stats(exec_result.mesh)
        results.append(
            EntityBuildResult(
                name=name, status=exec_result.status, executed_steps=exec_result.executed_steps,
                error=exec_result.error, mesh=exec_result.mesh, topology=topology,
            )
        )
        if exec_result.mesh is not None:
            transform = PlacementTransform.translation(position_mm)
            placed_meshes.append(transform.apply_to_mesh(exec_result.mesh))

    combined = trimesh.util.concatenate(placed_meshes) if placed_meshes else None
    return RawAssemblyBuildResult(mesh=combined, entities=results)
