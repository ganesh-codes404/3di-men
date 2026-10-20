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
from spec_validation import (
    detect_semantic_size_conflicts, validate_dimensions_against_description,
    validate_hole_count_against_description, validate_hole_diameters_against_description,
)


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
    repair_dropped_a_step: bool = False  # True only when the repair that actually fixed this point
        # was repair.TRUNCATE_TO_EARLIER_STEP (not the operand-swap strategy) -- meaning some
        # trailing part of the originally-predicted program was genuinely DROPPED to let the rest
        # finish, not just recomputed via an equivalent commutative path. Distinct from
        # repair_applied: an operand swap leaves repair_applied=True too, but changes neither the
        # requested dimensions nor the requested feature (see repair.py's own module docstring),
        # so it does NOT set this. AssemblyBuildResult.overall_status checks this specifically --
        # a status="ok" point that got there by dropping a step is still a real deviation from
        # what was asked for, even though it's now a geometrically valid, finished solid.
    excluded_from_assembly: bool = False  # True only when this point built a perfectly valid mesh
        # on its own but was then excluded from the FINAL COMBINED mesh by
        # _find_and_resolve_collisions -- an unrelated collision with another part, with no
        # relationship explaining the overlap ("it can't just give up" applied at the assembly
        # level: excluding the conflicting part is how the rest of the assembly still finishes
        # as a real, complete model, same trade-off repair.py's truncation strategy makes within
        # one part's own tool program). The point's own `mesh`/`status` above are NOT touched by
        # this -- it stays a disclosed, individually-valid result; only the combined assembly
        # mesh leaves it out. Never silent: see error and AssemblyBuildResult.collision_warnings.
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
    unmatched_hole_diameters_mm: list[float] = field(default_factory=list)  # spec_validation.py's
        # validate_hole_diameters_against_description -- any claimed hole/bore diameter (e.g. "a 12mm bore")
        # that no DETECTED hole's own fitted-circle diameter matches within tolerance. Empty (not None) when
        # every claim matched or there was nothing to check -- distinguishing "checked, all matched" from
        # "nothing to check" isn't load-bearing here the way claimed_hole_count's None is, since an empty
        # list already means "no unmatched claims" either way.
    semantic_size_conflicts: list[str] = field(default_factory=list)  # spec_validation.py's
        # detect_semantic_size_conflicts -- a description naming a known standard (e.g. "NEMA 17") AND
        # ALSO explicitly stating a disagreeing dimension for it. Disclosure only, never a status
        # downgrade: the explicit value is already what's built and validated, this just makes the
        # conflict itself visible rather than silently resolved (this project's own non-negotiable
        # principle: "explicit numerical dimensions override semantic defaults").


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
        # SUPPOSED to have one bbox almost entirely inside the other). Each entry here describes a
        # collision that has ALREADY been resolved (by excluding one of the two parts from the
        # combined mesh below -- see _find_and_resolve_collisions and
        # PointBuildResult.excluded_from_assembly), not one left unfixed: this is a disclosure log,
        # not a pending-failure list -- the build does not stop or FAIL just because this is non-empty.
    envelope_warnings: list[str] = field(default_factory=list)  # confirmed real gap this closes
        # (mission-brief regression: "final bounding box 251.2x124.0x108.6mm, versus an approximate
        # requested footprint of 150x100mm" -- reported with NO way to tell which component was
        # actually responsible). Populated only when the ORIGINAL PROMPT TEXT (not any one part's own
        # description) makes an explicit overall-footprint claim -- see
        # spec_validation.extract_claimed_envelope_mm -- and the ACTUAL combined assembly exceeds it
        # beyond a declared tolerance. Each entry traces the overrun to whichever placed component
        # reaches furthest from the assembly's own center along the overrunning axis -- a diagnostic,
        # same disclosure discipline as collision_warnings, not a silent pass/fail.

    @property
    def overall_status(self) -> str:
        """Confirmed real gap this closes: before this property existed,
        there was NO single aggregate success/fail field anywhere in this
        pipeline -- a caller had to inspect every point's own status
        string individually, and the combined mesh's own watertightness
        (computed from whatever DID get placed) could look perfectly
        clean even when a required part was empty, suspect, or outright
        rejected. "A slot returns an empty mesh and a gusset is marked
        suspect, but both are merged anyway... reported as green and
        watertight" is exactly this gap, not a misunderstanding of
        existing behavior.

        - "FAIL": at least one point completely failed (status="rejected",
          no usable mesh at all) and nothing -- not repair.py's bounded
          repair loop, not an unrelated collision's own resolution -- was
          able to rescue it. A mandatory requirement is definitively unmet.
        - "INCOMPLETE": every point produced SOME mesh, but at least one is
          not fully trustworthy (status in {"suspect", "partial",
          "out_of_distribution"} -- built, but geometrically dubious, built
          from an earlier failed step, a low-confidence statistical guess,
          OR excluded from the combined mesh to resolve a collision with
          another part -- see PointBuildResult.excluded_from_assembly), OR
          at least one point only reached status="ok" because
          repair.py's TRUNCATE_TO_EARLIER_STEP strategy actually dropped a
          step to get there (PointBuildResult.repair_dropped_a_step) -- a
          geometrically valid, finished solid that is still NOT what was
          originally requested, which this property must not hide behind a
          clean "PASS" just because the mesh itself is now watertight. OR
          the whole ASSEMBLY's own combined envelope exceeds an explicit
          footprint claim in the original prompt (self.envelope_warnings)
          -- every individual part can be perfectly "ok" while their
          combined placement still overruns the requested footprint
          (confirmed real regression: "final bounding box 251.2x124.0x108.6mm
          versus an approximate requested footprint of 150x100mm"). Never
          silently promoted to PASS.
        - "PASS": every point is status="ok", nothing was excluded from the
          combined mesh, no point needed a step dropped to get there
          (an operand-swap repair, which changes neither the requested
          dimensions nor the requested feature, does NOT disqualify PASS --
          see repair.py's own module docstring), and the combined assembly
          doesn't overrun any explicit footprint claim. The ONLY status
          that means "built and verified correct," not just "something was
          produced."

        A collision between two unrelated parts does NOT, by itself, mean
        FAIL any more: _find_and_resolve_collisions already excludes one of
        the two conflicting parts before this property is ever evaluated
        (the assembly-level version of "it can't just give up" -- see that
        function's own docstring), which downgrades the excluded point's
        own status to "suspect" and is caught by the INCOMPLETE check below
        like any other disclosed, non-"ok" result.

        Watertightness is already folded into each point's own status by
        Sandbox.execute() -- this property doesn't re-check topology, it
        aggregates what per-point status already disclosed, same
        discipline as dimension_match_rate/claimed_hole_count: never
        re-deriving the definition of correct from the generated parts
        themselves."""
        if not self.points:
            return "INCOMPLETE"
        if any(p.status == "rejected" for p in self.points):
            return "FAIL"
        if any(p.status in ("suspect", "partial", "out_of_distribution") for p in self.points):
            return "INCOMPLETE"
        if any(p.repair_dropped_a_step for p in self.points):
            return "INCOMPLETE"
        if self.envelope_warnings:
            return "INCOMPLETE"
        return "PASS"


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
    repair_dropped_a_step = any(
        a.strategy == "truncate_to_earlier_step" and a.improved for a in repaired.attempts
    )

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
        "repair_dropped_a_step": repair_dropped_a_step,
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


def build_assembly_from_points(
    points: list[ActionablePoint], bundle_path: Path, max_workers: int = 4,
    claimed_envelope_mm: "tuple[float, float] | None" = None,
) -> AssemblyBuildResult:
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
    costs more than the fully-parallel version did.

    Applies llm_decompose._correct_gear_mesh_spacing UNCONDITIONALLY, not
    just when points arrive via decompose_prompt (which already applies
    it too -- calling it twice is harmless/idempotent, see that
    function's own design: an already-correct distance produces a ~zero
    correction). Confirmed real gap this closes: hand-constructed points
    (the frontend's manual playground, or any other direct caller of
    this function) previously never got the exact gear-mesh-distance
    correction at all -- only the LLM-decomposition path did, even
    though the correction itself has nothing LLM-specific about it (pure
    text-regex + relationship-chain math)."""
    from assembly_plan import (
        Relationship, RelationshipError, order_by_dependency, resolve_position, resolve_rotation_deg,
        separate_coincident_face_attachments, validate_relationships,
    )
    from llm_decompose import _correct_gear_mesh_spacing

    points = _correct_gear_mesh_spacing(points)
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

            # Pre-pass: sibling points in THIS wave that share the identical
            # attached_to_face target+face+offset (a mirrored pair the LLM
            # emitted without actually differentiating -- see
            # assembly_plan.separate_coincident_face_attachments's own
            # docstring) get auto-separated using their own ALREADY-BUILT
            # mesh extents, before resolve_position ever sees the original
            # coincident offsets below. Grouped by (target_id, face) across
            # the whole wave, not just pairs we already suspect -- cheap,
            # and the function itself is a no-op for any group that isn't
            # exactly 2 members with identical offsets.
            adjusted_relationships: dict[str, Relationship] = {}
            face_attach_groups: dict[tuple[str, str], list[tuple[str, Relationship, trimesh.Trimesh]]] = {}
            for pid, point, raw in zip(wave, wave_points, raw_results):
                rel = point.relationship
                if rel is not None and rel.kind == "attached_to_face" and raw["mesh"] is not None:
                    key = (rel.target_id, rel.face or "+z")
                    face_attach_groups.setdefault(key, []).append((pid, rel, raw["mesh"]))
            for (target_id, _face), members in face_attach_groups.items():
                target_mesh = placed_meshes_by_id.get(target_id)
                if target_mesh is not None:
                    adjusted_relationships.update(separate_coincident_face_attachments(members, target_mesh))

            for pid, point, raw in zip(wave, wave_points, raw_results):
                own_mesh, status, error = raw["mesh"], raw["status"], raw["error"]
                position_mm = point.position_mm
                rotation_deg = (0.0, 0.0, 0.0)
                effective_relationship = adjusted_relationships.get(pid, point.relationship)

                if own_mesh is not None and effective_relationship is not None:
                    target_mesh = placed_meshes_by_id.get(effective_relationship.target_id)
                    if target_mesh is None:
                        status, error, own_mesh = "rejected", (
                            f"cannot resolve relationship: target {effective_relationship.target_id!r} "
                            "failed to build its own geometry"
                        ), None
                    else:
                        try:
                            position_mm = resolve_position(effective_relationship, target_mesh, own_mesh)
                            # resolve_position only computes WHERE a part's center
                            # belongs -- a cylinder/gear/capsule/etc built along its
                            # own default local Z axis (sandbox.py's convention)
                            # otherwise stays pointing along Z even when
                            # concentric_with/attached_to_face says it should run
                            # along x or y, which is exactly the confirmed real bug
                            # behind "a shaft floats beside its housing instead of
                            # running through it" -- the center lands right, the
                            # part itself is still oriented wrong.
                            rotation_deg = resolve_rotation_deg(effective_relationship)
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

                # Feature validation (hole/bore DIAMETER) -- same wiring
                # principle again: GeometryReport.hole_diameters_mm was
                # already measurable from each detected hole's own
                # fitted-circle radius, just never checked against a
                # claimed diameter until now. More precise than the
                # general dimension check above, which can only compare
                # against the mesh's overall extents, not a specific
                # feature.
                unmatched_hole_diameters_mm: list[float] = []
                if own_mesh is not None:
                    diameter_report = validate_hole_diameters_against_description(point.description, raw.get("geometry_report"))
                    if diameter_report.claims and not diameter_report.all_matched:
                        unmatched_hole_diameters_mm = diameter_report.unmatched_claims_mm
                        mismatch_desc = ", ".join(f"{d:g}mm" for d in unmatched_hole_diameters_mm)
                        problems.append(
                            f"description claims a hole/bore diameter of {mismatch_desc} that no detected hole "
                            f"matches within tolerance ({diameter_report.detector_limitation})"
                        )

                # Semantic-name-vs-explicit-dimension conflict (non-
                # negotiable principle: "explicit numerical dimensions
                # override semantic defaults associated with names such as
                # 'NEMA 17'"). Purely a DISCLOSURE, not a new "problem" --
                # the explicit value is already what dimension_match_rate
                # above validates the built mesh against, so correctly
                # honoring it is not itself a defect; this just makes the
                # fact that a conflict existed, and which value won,
                # visible rather than silent. Text-only (doesn't need
                # own_mesh), so computed unconditionally.
                semantic_size_conflicts = [
                    f"description names {c.semantic_name} (standard {c.standard_value_mm:g}mm face) but also "
                    f"explicitly states {c.explicit_value_mm:g}mm -- the explicit dimension takes precedence"
                    for c in detect_semantic_size_conflicts(point.description)
                ]

                if status == "ok" and problems:
                    status = "suspect"
                    error = "; ".join(problems)
                if semantic_size_conflicts:
                    note = "; ".join(semantic_size_conflicts)
                    error = f"{error}; {note}" if error else note

                if pid in adjusted_relationships:
                    # Disclosed, never silent -- same discipline as repair.py's
                    # RepairAttempt log and collision_warnings: this part's real
                    # placement differs from what the ORIGINAL relationship
                    # would have given, because one or more OTHER members
                    # sharing the same attached_to_face target+face ended up
                    # too close together in the REAL built geometry. Phrased
                    # generically (not "identical offset", not "a sibling")
                    # since separate_coincident_face_attachments now handles
                    # groups of any size with non-identical offsets too (e.g.
                    # a structural plate plus two smaller gussets sharing one
                    # face, none of which were given the same offset).
                    note = (
                        "auto-separated from other part(s) sharing the same attached_to_face target+face -- the "
                        f"originally given offsets put them too close together in the real built geometry; used "
                        f"offset_u={effective_relationship.offset_u:g}, offset_v={effective_relationship.offset_v:g} instead"
                    )
                    error = f"{error}; {note}" if error else note

                point_results[pid] = PointBuildResult(
                    point=ActionablePoint(description=point.description, position_mm=position_mm, id=pid, relationship=effective_relationship),
                    status=status, executed_steps=raw["executed_steps"], error=error,
                    nearest_neighbor_similarity=raw["nearest_neighbor_similarity"], program=raw["program"],
                    mesh=own_mesh, topology=raw["topology"],
                    dimension_match_rate=dimension_match_rate, unmatched_dimensions_mm=unmatched_dimensions_mm,
                    repair_applied=raw.get("repair_applied", False), repair_attempts=raw.get("repair_attempts", []),
                    repair_dropped_a_step=raw.get("repair_dropped_a_step", False),
                    resolution_method=raw.get("resolution_method", "statistical"),
                    claimed_hole_count=claimed_hole_count, detected_hole_count=detected_hole_count,
                    unmatched_hole_diameters_mm=unmatched_hole_diameters_mm,
                    semantic_size_conflicts=semantic_size_conflicts,
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

    # Collision resolution runs BEFORE the combined mesh is assembled, not
    # after: the whole point is to keep the conflicting part out of
    # `combined`, not to report a collision that's already baked into the
    # output. See _find_and_resolve_collisions's own docstring for why
    # this is the assembly-level counterpart to repair.py's "it can't just
    # give up" truncation strategy.
    collision_warnings, excluded_ids = _find_and_resolve_collisions(placed_meshes_by_id, relationships)
    for pid in excluded_ids:
        placed_meshes_by_id.pop(pid, None)
        pr = point_results[pid]
        pr.excluded_from_assembly = True
        note = (
            "excluded from the combined assembly: collided with another part with no "
            "relationship explaining the overlap -- the rest of the assembly still completes "
            "without it, rather than leaving colliding geometry in place or failing outright"
        )
        pr.error = f"{pr.error}; {note}" if pr.error else note
        if pr.status == "ok":
            pr.status = "suspect"  # reusing the existing status enum, same discipline as the
                                    # dimension/hole-count checks above -- never a new value

    ordered_results = [point_results[pid] for pid in ids]
    placed_meshes = [placed_meshes_by_id[pid] for pid in ids if pid in placed_meshes_by_id]
    combined = trimesh.util.concatenate(placed_meshes) if placed_meshes else None

    envelope_warnings = (
        _diagnose_envelope_overrun(claimed_envelope_mm, placed_meshes_by_id)
        if claimed_envelope_mm is not None else []
    )

    return AssemblyBuildResult(
        mesh=combined, points=ordered_results, fell_back_to_single_point=False,
        collision_warnings=collision_warnings, envelope_warnings=envelope_warnings,
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


def _find_and_resolve_collisions(
    placed_meshes_by_id: dict, relationships: dict, threshold: float = 0.3
) -> tuple[list[str], set[str]]:
    """Confirmed real gap this closes (previously: nowhere in this pipeline
    checked for unintended overlap between independently-placed parts, and
    even once a check was added it only ever REPORTED a collision -- the
    colliding geometry stayed in the combined mesh and the whole assembly
    was marked FAIL, which is exactly the "it can't just give up" gap:
    a collision between two parts is a build-time problem exactly like a
    rejected part is, and must get the same active-recovery treatment
    repair.py's truncation strategy already gives a single part's own
    failing tool program, not just a flagged-and-abandoned report).

    Resolution strategy: for each colliding pair, exclude the LATER of the
    two (by placement/id order) from the final combined mesh -- the same
    "drop the trailing thing, keep the earlier one" bias
    repair._repair_by_truncation already uses when a part's own program has
    to lose a step to finish. This is never silent: every exclusion is
    disclosed both in the returned warning strings AND, back at the call
    site, via the excluded point's own PointBuildResult.excluded_from_assembly
    and .error fields -- the point is never dropped from the response,
    only from the combined mesh.

    See AssemblyBuildResult.collision_warnings and
    mesh_quality.bounding_box_overlap_fraction's own docstrings for the
    full overlap-detection reasoning. Deliberately scoped to pairs with NO
    relationship connecting them: a shaft concentric_with a boss's bore is
    SUPPOSED to sit almost entirely inside the boss's own bounding box,
    which this coarse AABB check cannot distinguish from a genuine
    collision, so those pairs are excluded from consideration entirely
    rather than producing constant false alarms -- and false exclusions --
    on exactly the placement pattern this project relies on most.

    threshold=0.3 chosen from real calibration, not guessed: two gears
    correctly meshing at their tooth tips overlap by ~5-10% of the smaller
    gear's own bounding-box volume; the actual confirmed bug this project
    already fixed once (two gears resolved to the identical position)
    overlaps by 100%. 0.3 sits comfortably between the two."""
    warnings: list[str] = []
    excluded: set[str] = set()
    ids = list(placed_meshes_by_id.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            if _relationship_connected(id_a, id_b, relationships):
                continue
            overlap = bounding_box_overlap_fraction(placed_meshes_by_id[id_a], placed_meshes_by_id[id_b])
            if overlap > threshold:
                loser = id_b  # later in placement order -- consistent with repair.py's
                              # own bias toward dropping trailing work before earlier work
                warnings.append(
                    f"{id_a!r} and {id_b!r} overlapped by {overlap:.0%} of the smaller part's "
                    f"bounding box with no relationship explaining it -- {loser!r} excluded from "
                    "the combined assembly to resolve it, rather than leaving colliding geometry "
                    "in place or failing the whole build"
                )
                excluded.add(loser)
    return warnings, excluded


def _diagnose_envelope_overrun(
    claimed_envelope_mm: tuple[float, float], placed_meshes_by_id: dict[str, trimesh.Trimesh],
    tolerance_fraction: float = 0.2,
) -> list[str]:
    """Confirmed real gap this closes (mission-brief regression): "final
    bounding box 251.2x124.0x108.6mm, versus an approximate requested
    footprint of 150x100mm" was reported with NO way to tell WHICH
    component was actually responsible -- every individual part can be a
    perfectly valid "ok" solid while their combined PLACEMENT still
    overruns the requested footprint; this checks the assembly's own
    combined extent, not any one part's.

    Scoped to X/Y only (a footprint claim is a floor-plan size, not a
    height -- the brief's own point: "the prompt may specify only a
    footprint, not a height"). tolerance_fraction=0.2 is a deliberately
    loose default: "close to" / "approximately" in the claim's own
    wording already discloses this is a target, not a hard maximum (see
    extract_claimed_envelope_mm's own docstring) -- flagging every few-
    percent deviation on an explicitly approximate claim would be noise,
    not a diagnostic.

    Traces the overrun to whichever placed component reaches furthest
    from the assembly's own center along the overrunning axis -- a
    real, inspectable diagnostic (e.g. "'gear_b' reaches furthest..."),
    not just a bare pass/fail on the combined number."""
    if not placed_meshes_by_id:
        return []
    all_bounds = np.array([mesh.bounds for mesh in placed_meshes_by_id.values()])
    combined_min, combined_max = all_bounds[:, 0, :].min(axis=0), all_bounds[:, 1, :].max(axis=0)
    actual_extent = combined_max - combined_min
    combined_center = (combined_min + combined_max) / 2.0

    warnings: list[str] = []
    for axis_idx, axis_name in ((0, "X"), (1, "Y")):
        claimed = claimed_envelope_mm[axis_idx]
        if claimed <= 0:
            continue
        actual = float(actual_extent[axis_idx])
        overrun = actual - claimed
        if overrun <= claimed * tolerance_fraction:
            continue
        worst_id, worst_reach = None, 0.0
        for pid, mesh in placed_meshes_by_id.items():
            reach = max(
                abs(float(mesh.bounds[0][axis_idx]) - combined_center[axis_idx]),
                abs(float(mesh.bounds[1][axis_idx]) - combined_center[axis_idx]),
            )
            if reach > worst_reach:
                worst_reach, worst_id = reach, pid
        warnings.append(
            f"assembly's {axis_name} extent is {actual:.1f}mm, exceeding the requested ~{claimed:.0f}mm "
            f"footprint by {overrun:.1f}mm -- {worst_id!r} reaches furthest from the assembly's own "
            f"center along {axis_name} ({worst_reach:.1f}mm), most likely responsible for the overrun"
        )
    return warnings


def build_assembly_from_prompt(prompt: str, bundle_path: Path, max_workers: int = 4) -> AssemblyBuildResult:
    from spec_validation import extract_claimed_envelope_mm

    decomposition = decompose_prompt(prompt)
    claimed_envelope_mm = extract_claimed_envelope_mm(prompt)
    result = build_assembly_from_points(decomposition.points, bundle_path, max_workers, claimed_envelope_mm=claimed_envelope_mm)
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
