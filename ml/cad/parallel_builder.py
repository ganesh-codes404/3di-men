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
from mesh_quality import bounding_box_overlap_fraction
from spec_validation import validate_dimensions_against_description, validate_hole_count_against_description


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

    @classmethod
    def rotation(cls, euler_deg) -> "PlacementTransform":
        """Three fixed-order axis rotations -- X, then Y, then Z, each built
        via trimesh's own rotation_matrix helper (the same convention
        sandbox.py's single-axis rotate() tool already uses) rather than a
        general quaternion, so this project has one rotation convention
        throughout, not two. Pivots about the world origin, same as
        sandbox.py's rotate() -- compose with translation() (X then Y then Z
        rotation first, then the translation) to pivot about the part's own
        local origin before moving it into place."""
        rx_deg, ry_deg, rz_deg = euler_deg
        rx = trimesh.transformations.rotation_matrix(np.radians(rx_deg), (1.0, 0.0, 0.0))
        ry = trimesh.transformations.rotation_matrix(np.radians(ry_deg), (0.0, 1.0, 0.0))
        rz = trimesh.transformations.rotation_matrix(np.radians(rz_deg), (0.0, 0.0, 1.0))
        return cls(rz @ ry @ rx)

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
    dimension_match_rate: float | None = None  # spec_validation.py's correctness-to-spec check --
        # fraction of the description's own "NNmm" dimension claims found somewhere on the
        # built mesh (within tolerance). None when there was no mesh to check or no numeric
        # claims in the description at all (not the same as a confirmed 100% match -- see
        # unmatched_dimensions_mm). A structurally "ok" status has NEVER meant "matches the
        # request" anywhere in this pipeline before this field existed (see ACCURACY_AUDIT.md
        # section 3) -- this is the first place that distinction becomes visible.
    unmatched_dimensions_mm: list[float] = field(default_factory=list)
    repair_applied: bool = False  # repair.py's bounded repair loop (section 8) -- True only if a
                                   # repair attempt actually IMPROVED the result's status
    repair_attempts: list[str] = field(default_factory=list)  # every attempt tried, improved or not --
                                                                 # never hidden, even a failed one
    resolution_method: str = "statistical"  # "compositional" (compositional_decomposer.py's deterministic
        # clause-by-clause resolution), "statistical" (program_model.py's TF-IDF/classifier prediction), or
        # "user_edited" (ActionablePoint.program_override -- prediction skipped entirely, the user's own
        # hand-edited program ran instead) -- disclosed, not hidden, same discipline as
        # in_distribution/dimension_match_rate: a caller should be able to tell a literally-parsed program,
        # a guessed one, and a hand-edited one apart.
    claimed_hole_count: int | None = None  # spec_validation.py's hole-count check -- None when the
        # description made no hole claim at all (not the same as a confirmed match -- see detected_hole_count)
    detected_hole_count: int | None = None  # evaluation/geometry_eval.py's GeometryReport, computed by every
        # build already but previously discarded by every caller. None when no mesh was built to check, OR
        # (its own disclosed scope) the detector only recognizes vertical, axis-aligned, watertight features.


@dataclass
class AssemblyBuildResult:
    mesh: "trimesh.Trimesh | None"
    points: list[PointBuildResult]
    fell_back_to_single_point: bool  # true if the LLM decomposition itself failed (see llm_decompose.DecomposeResult)
    raw_llm_output: str | None = None  # None for the manual-points path, which never calls the LLM
    knowledge_context: str = ""  # the engineering-reference text actually retrieved and shown to the
                                  # model for this prompt, if any (see llm_decompose.DecomposeResult) --
                                  # "" for the manual-points path too, which never calls decompose_prompt
    collision_warnings: list[str] = field(default_factory=list)  # confirmed real gap this closes: no
        # assembly-level check anywhere previously caught two independently-placed parts ending up
        # stacked on/inside each other (the exact shape of the gear-mesh-spacing bug this project
        # already fixed once) -- see bounding_box_overlap_fraction's own docstring for why this is
        # scoped to parts with NO relationship connecting them (a shaft through a boss's bore is
        # SUPPOSED to have one bbox almost entirely inside the other).


_worker_bundle: Any = None  # per-process cache -- loaded once per worker, not once per point


def _init_worker(bundle_path: str) -> None:
    global _worker_bundle
    from program_model import ProgramModelBundle

    _worker_bundle = ProgramModelBundle.load(Path(bundle_path))


def _build_one(args: tuple[str, "list[dict] | None"]) -> dict:
    """Runs in a worker process. Returns a plain dict (not a dataclass with
    a trimesh payload attached) plus the mesh separately as exported bytes,
    since ProcessPoolExecutor results must round-trip through pickling --
    trimesh.Trimesh pickles fine, but keeping the boundary explicit here
    documents that this crosses a process boundary.

    `args` is (description, program_override) -- see
    ActionablePoint.program_override's own docstring. When an override is
    given, prediction is skipped entirely and THIS program is run instead,
    still through the exact same repair/validation pipeline every
    predicted program goes through."""
    description, program_override = args
    from compositional_decomposer import compose_program_from_description
    from program_model import ProgramPrediction, predict_program_checked
    from repair import execute_with_bounded_repair
    from sandbox import ToolCall

    if program_override is not None:
        prediction = ProgramPrediction(
            program=[ToolCall(op=step["op"], args=dict(step["args"])) for step in program_override],
            nearest_neighbor_similarity=1.0, in_distribution=True,
        )
        resolution_method = "user_edited"
    else:
        # Try the general, deterministic clause-by-clause composer FIRST --
        # see compositional_decomposer.py's module docstring for why this is
        # the general fix (coverage by combination, not by trained whole-
        # sentence example) rather than a patch for any one failure case. It
        # returns None for anything it doesn't recognize (never a forced
        # guess), in which case this falls back to the existing statistical
        # path completely unchanged -- every description that path already
        # handled continues to go through it exactly as before.
        compositional_program = compose_program_from_description(description)
        if compositional_program is not None:
            prediction = ProgramPrediction(program=compositional_program, nearest_neighbor_similarity=1.0, in_distribution=True)
            resolution_method = "compositional"
        else:
            prediction = predict_program_checked(_worker_bundle, description)
            resolution_method = "statistical"

    # Section 8's bounded repair loop, not a bare Sandbox.execute() call --
    # see repair.py's own module docstring for exactly what's (safely)
    # repairable and why most tempting "fixes" are deliberately excluded.
    repaired = execute_with_bounded_repair(prediction.program)
    result = repaired.result

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
        "repair_applied": repaired.was_repaired,
        "repair_attempts": [a.description for a in repaired.attempts],
        "resolution_method": resolution_method,
        # Already computed by Sandbox.execute() on every build -- previously
        # discarded here (confirmed by grepping the whole repo: nothing
        # outside sandbox.py itself ever read result.geometry_report before
        # this). detected_hole_count feeds the new hole-count check below.
        "geometry_report": result.geometry_report.as_dict() if result.geometry_report is not None else None,
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
    from assembly_plan import RelationshipError, order_by_dependency, resolve_position, resolve_rotation_deg, validate_relationships

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
            raw_results = list(pool.map(_build_one, [(p.description, p.program_override) for p in wave_points]))

            for pid, point, raw in zip(wave, wave_points, raw_results):
                own_mesh, status, error = raw["mesh"], raw["status"], raw["error"]
                position_mm = point.position_mm
                rotation_deg = (0.0, 0.0, 0.0)

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
                            # resolve_position only computes WHERE a part's center
                            # belongs -- a cylinder/gear/capsule/etc built along its
                            # own default local Z axis (sandbox.py's convention)
                            # otherwise stays pointing along Z even when
                            # concentric_with/attached_to_face says it should run
                            # along x or y, which is exactly the confirmed real bug
                            # behind "a shaft floats beside its housing instead of
                            # running through it" -- the center lands right, the
                            # part itself is still oriented wrong.
                            rotation_deg = resolve_rotation_deg(point.relationship)
                        except RelationshipError as exc:
                            status, error, own_mesh = "rejected", f"RelationshipError: {exc}", None

                # Confirmed real bug these two checks close together:
                # dimension_match_rate/hole-count data was already computed
                # correctly on every build but never gated anything -- a
                # part whose built geometry didn't match ANY of its own
                # stated dimensions, or had the wrong number of holes, could
                # still come back status="ok" (a purely structural check:
                # watertight + positive volume, with no concept of
                # "correct"). The frontend never displayed either field,
                # so the mismatch was completely invisible -- the exact
                # "silent approximation" failure mode this project
                # otherwise explicitly designs against (sandbox.py's own
                # typed statuses, this point's own out_of_distribution
                # flag, etc.). Reusing "suspect" (not a new status) keeps
                # this additive: every existing consumer of the status enum
                # already has to handle it. Collected into one list so a
                # part that fails BOTH checks discloses both, not just
                # whichever ran first.
                problems: list[str] = []

                dimension_match_rate = None
                unmatched_dimensions_mm: list[float] = []
                if own_mesh is not None:
                    dim_report = validate_dimensions_against_description(point.description, own_mesh)
                    if dim_report.claims:  # None (not 1.0) when the description had no numeric claims at all
                        dimension_match_rate = dim_report.match_rate
                        unmatched_dimensions_mm = dim_report.unmatched_claims_mm
                        if dimension_match_rate < 1.0:
                            mismatch_desc = ", ".join(f"{d:g}mm" for d in unmatched_dimensions_mm)
                            problems.append(f"built geometry doesn't match the description's own stated dimensions ({mismatch_desc})")

                # Feature validation (hole count) -- same wiring principle as
                # above: GeometryReport.detected_hole_count was already
                # computed by Sandbox.execute() on every build (see
                # _build_one's own "geometry_report" field), just never
                # checked against anything until now.
                claimed_hole_count = None
                detected_hole_count = None
                if own_mesh is not None:
                    hole_report = validate_hole_count_against_description(point.description, raw.get("geometry_report"))
                    claimed_hole_count = hole_report.claimed_count
                    detected_hole_count = hole_report.detected_count
                    if not hole_report.matched:
                        problems.append(
                            f"description claims {hole_report.claimed_count} hole(s), detector found "
                            f"{hole_report.detected_count} ({hole_report.detector_limitation})"
                        )

                if status == "ok" and problems:
                    status = "suspect"
                    error = "; ".join(problems)

                point_results[pid] = PointBuildResult(
                    point=ActionablePoint(description=point.description, position_mm=position_mm, id=pid, relationship=point.relationship),
                    status=status, executed_steps=raw["executed_steps"], error=error,
                    nearest_neighbor_similarity=raw["nearest_neighbor_similarity"], program=raw["program"],
                    mesh=own_mesh, topology=raw["topology"],
                    dimension_match_rate=dimension_match_rate, unmatched_dimensions_mm=unmatched_dimensions_mm,
                    repair_applied=raw.get("repair_applied", False), repair_attempts=raw.get("repair_attempts", []),
                    resolution_method=raw.get("resolution_method", "statistical"),
                    claimed_hole_count=claimed_hole_count, detected_hole_count=detected_hole_count,
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
                    # Rotate about the part's own local origin FIRST (where
                    # it was actually built), then translate into its final
                    # world position -- an all-zero rotation_deg (the
                    # no-relationship case, same as before) reduces exactly
                    # to the prior translation-only behavior.
                    transform = PlacementTransform.translation(position_mm).compose(PlacementTransform.rotation(rotation_deg))
                    placed = transform.apply_to_mesh(own_mesh)
                    placed_meshes_by_id[pid] = placed

    ordered_results = [point_results[pid] for pid in ids]
    placed_meshes = [placed_meshes_by_id[pid] for pid in ids if pid in placed_meshes_by_id]
    combined = trimesh.util.concatenate(placed_meshes) if placed_meshes else None

    collision_warnings = _find_collisions(placed_meshes_by_id, relationships)
    return AssemblyBuildResult(
        mesh=combined, points=ordered_results, fell_back_to_single_point=False,
        collision_warnings=collision_warnings,
    )


def _relationship_connected(pid_a: str, pid_b: str, relationships: dict) -> bool:
    """True if either part's relationship chain (direct or transitive)
    reaches the other -- these pairs are DESIGNED to have one bbox nearly
    or fully inside the other (a shaft through a boss's bore, a boss into
    a housing wall) and must be excluded from the collision check below,
    which cannot otherwise tell that apart from a real collision."""

    def chain_reaches(start: str, target: str) -> bool:
        seen: set[str] = set()
        cur = start
        while cur not in seen:
            if cur == target:
                return True
            seen.add(cur)
            rel = relationships.get(cur)
            if rel is None:
                return False
            cur = rel.target_id
        return False  # cycle guard -- validate_relationships already rejects real cycles upstream

    return chain_reaches(pid_a, pid_b) or chain_reaches(pid_b, pid_a)


def _find_collisions(placed_meshes_by_id: dict, relationships: dict, threshold: float = 0.3) -> list[str]:
    """Confirmed real gap this closes (previously: nowhere in this pipeline
    checked for unintended overlap between independently-placed parts) --
    see AssemblyBuildResult.collision_warnings and
    mesh_quality.bounding_box_overlap_fraction's own docstrings for the
    full reasoning. Deliberately scoped to pairs with NO relationship
    connecting them: a shaft concentric_with a boss's bore is SUPPOSED to
    sit almost entirely inside the boss's own bounding box, which this
    coarse AABB check cannot distinguish from a genuine collision, so
    those pairs are excluded rather than producing constant false alarms
    on exactly the placement pattern this project relies on most.

    threshold=0.3 chosen from real calibration, not guessed: two gears
    correctly meshing at their tooth tips overlap by ~5-10% of the smaller
    gear's own bounding-box volume; the actual confirmed bug this project
    already fixed once (two gears resolved to the identical position)
    overlaps by 100%. 0.3 sits comfortably between the two."""
    warnings: list[str] = []
    ids = list(placed_meshes_by_id.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            if _relationship_connected(id_a, id_b, relationships):
                continue
            overlap = bounding_box_overlap_fraction(placed_meshes_by_id[id_a], placed_meshes_by_id[id_b])
            if overlap > threshold:
                warnings.append(
                    f"{id_a!r} and {id_b!r} overlap by {overlap:.0%} of the smaller part's bounding "
                    "box, with no relationship between them explaining it -- check placement"
                )
    return warnings


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
    entities: list[tuple[str, list, tuple[float, float, float], tuple[float, float, float]]],
) -> RawAssemblyBuildResult:
    """The manual tool console's counterpart to build_assembly_from_points --
    same combine-by-translation(-and-now-rotation) logic, but for a user who
    has hand-authored each entity's own raw Sandbox program directly (via the
    console's form/gizmo/click-drag tools), not a text description for
    program_model to predict from. Each entry is
    (name, program, position_mm, rotation_deg) -- rotation_deg is applied
    (X, then Y, then Z, about the entity's own local origin, see
    PlacementTransform.rotation()) BEFORE translating into position_mm, same
    order sandbox.py's own rotate() tool implies. `program` is expected to
    already end in its own "finalize" step, exactly like a single-entity
    /api/execute_program call. build_assembly_from_points (the separate,
    LLM-prompt path) is untouched by this -- it has its own, translation-only
    placement resolution in assembly_plan.py and is out of scope here.

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
    for name, program, position_mm, rotation_deg in entities:
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
            # Rotation first (about the entity's own local origin), then
            # translation into its final position_mm -- an all-zero
            # rotation_deg reduces this exactly to the prior
            # translation-only behavior, so this is provably additive.
            transform = PlacementTransform.translation(position_mm).compose(PlacementTransform.rotation(rotation_deg))
            placed_meshes.append(transform.apply_to_mesh(exec_result.mesh))

    combined = trimesh.util.concatenate(placed_meshes) if placed_meshes else None
    return RawAssemblyBuildResult(mesh=combined, entities=results)
