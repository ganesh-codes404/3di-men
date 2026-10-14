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
    program: list[dict]  # [{"op": ..., "args": {...}}, ...] -- the actual predicted tool
                          # sequence. Previously dropped entirely once execution finished,
                          # which meant a point that executes "ok" but predicted the WRONG
                          # tool sequence (confirmed, real failure mode -- see sandbox.py's
                          # own regression notes on this) was indistinguishable from a
                          # correct one anywhere downstream. Kept as plain dicts, not
                          # ToolCall objects, so this survives the ProcessPoolExecutor
                          # pickling boundary and serializes straight to JSON for logging.
    mesh: "trimesh.Trimesh | None" = field(repr=False, default=None)
    topology: "dict | None" = None


@dataclass
class AssemblyBuildResult:
    mesh: "trimesh.Trimesh | None"
    points: list[PointBuildResult]
    fell_back_to_single_point: bool  # true if the LLM decomposition itself failed (see llm_decompose.DecomposeResult)
    raw_llm_output: str | None = None  # None for the manual-points path, which never calls the LLM
    knowledge_context: str = ""  # the engineering-reference text actually retrieved and shown to the
                                  # model for this prompt, if any (see llm_decompose.DecomposeResult) --
                                  # "" for the manual-points path too, which never calls decompose_prompt


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
        "program": [{"op": c.op, "args": c.args} for c in prediction.program],
        "mesh": result.mesh,
        "topology": topology,
    }


def _unique_point_ids(points: list[ActionablePoint]) -> list[str]:
    """A blank or duplicated id (an older caller that never set one, or a
    hand-edited playground list) must still produce a usable, unique id
    per point -- relationship resolution and the final response both key
    off this list, positionally aligned with `points`."""
    seen: set[str] = set()
    ids: list[str] = []
    for i, p in enumerate(points):
        pid = p.id or f"_idx{i}"
        if pid in seen:
            pid = f"{pid}_{i}"
        seen.add(pid)
        ids.append(pid)
    return ids


def build_assembly_from_points(points: list[ActionablePoint], bundle_path: Path, max_workers: int = 4) -> AssemblyBuildResult:
    """The shared build core: takes a list of actionable points from
    WHATEVER source produced them -- the LLM's decomposition, or a user
    hand-editing/hand-adding points in the frontend's playground -- and
    builds each one via program_model.py + sandbox.py, exactly the same
    way either way. Manual points are not a separate, lesser path: they go
    through the identical proprietary tool-composing builder, just
    skipping the LLM's guess at what the parts and their positions
    should be.

    Points are built in DEPENDENCY WAVES, not all at once: a point whose
    position comes from a relationship (see assembly_plan.py) can only be
    placed once its target's own mesh has actually been built, so it waits
    for that wave to finish. Points with no relationship (or whose target
    is already resolved) build in parallel within their own wave, exactly
    as before -- most real assemblies are 1-2 waves deep, so this rarely
    costs more than the fully-parallel version did."""
    from assembly_plan import RelationshipError, order_by_dependency, resolve_position, validate_relationships

    ids = _unique_point_ids(points)
    id_to_point = dict(zip(ids, points))
    relationships = {pid: p.relationship for pid, p in zip(ids, points)}

    try:
        validate_relationships(ids, relationships)
        waves = order_by_dependency(ids, relationships)
    except RelationshipError as exc:
        # A structurally invalid plan (dangling reference, cycle,
        # self-reference) is rejected BEFORE anything is built -- the same
        # "typed failure, never silent wrongness" rule sandbox.py's own
        # ProgramExecutionResult follows for a single part's tool program.
        rejected = [
            PointBuildResult(point=p, status="rejected", executed_steps=0, error=f"RelationshipError: {exc}",
                              nearest_neighbor_similarity=0.0, program=[], mesh=None, topology=None)
            for p in points
        ]
        return AssemblyBuildResult(mesh=None, points=rejected, fell_back_to_single_point=False)

    point_results: dict[str, PointBuildResult] = {}
    placed_meshes_by_id: dict[str, trimesh.Trimesh] = {}  # final, WORLD-placed meshes -- relationship targets for later waves

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(str(bundle_path),)) as pool:
        for wave in waves:
            if not wave:
                continue
            wave_points = [id_to_point[pid] for pid in wave]
            raw_results = list(pool.map(_build_one, [p.description for p in wave_points]))

            for pid, point, raw in zip(wave, wave_points, raw_results):
                own_mesh, status, error = raw["mesh"], raw["status"], raw["error"]
                position_mm = point.position_mm

                if own_mesh is not None and point.relationship is not None:
                    target_mesh = placed_meshes_by_id.get(point.relationship.target_id)
                    if target_mesh is None:
                        status, error, own_mesh = "rejected", (
                            f"cannot resolve relationship: target {point.relationship.target_id!r} "
                            "failed to build its own geometry"
                        ), None
                    else:
                        try:
                            position_mm = resolve_position(point.relationship, target_mesh, own_mesh)
                        except RelationshipError as exc:
                            status, error, own_mesh = "rejected", f"RelationshipError: {exc}", None

                point_results[pid] = PointBuildResult(
                    point=ActionablePoint(description=point.description, position_mm=position_mm, id=pid, relationship=point.relationship),
                    status=status, executed_steps=raw["executed_steps"], error=error,
                    nearest_neighbor_similarity=raw["nearest_neighbor_similarity"], program=raw["program"],
                    mesh=own_mesh, topology=raw["topology"],
                )
                # "out_of_distribution" and "suspect" still carry a mesh (a
                # guess, or a geometrically dubious result) -- included in
                # the combined assembly but disclosed via status, never
                # silently upgraded to look like a clean "ok". Only a
                # genuinely missing mesh (rejected/partial-with-no-mesh,
                # including a relationship that couldn't be resolved) is
                # left out, and even then the point is still reported,
                # never dropped from the response entirely.
                if own_mesh is not None:
                    placed = PlacementTransform.translation(position_mm).apply_to_mesh(own_mesh)
                    placed_meshes_by_id[pid] = placed

    ordered_results = [point_results[pid] for pid in ids]
    placed_meshes = [placed_meshes_by_id[pid] for pid in ids if pid in placed_meshes_by_id]
    combined = trimesh.util.concatenate(placed_meshes) if placed_meshes else None
    return AssemblyBuildResult(mesh=combined, points=ordered_results, fell_back_to_single_point=False)


def build_assembly_from_prompt(prompt: str, bundle_path: Path, max_workers: int = 4) -> AssemblyBuildResult:
    decomposition = decompose_prompt(prompt)
    result = build_assembly_from_points(decomposition.points, bundle_path, max_workers)
    result.fell_back_to_single_point = decomposition.fell_back
    result.raw_llm_output = decomposition.raw_llm_output
    result.knowledge_context = decomposition.knowledge_context
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
