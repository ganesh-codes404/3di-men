"""Bounded, targeted repair for a hole-COUNT or hole-PLACEMENT mismatch
between what the text requires and what the built geometry actually
has -- Part 8's own ask, for the one failure category this task's own
regression suite confirmed live and dimension_assignment.py alone
cannot fix (it correctly sets `count`/diameter/depth on whichever op(s)
the chosen program has, but has no way to correct an op's own PLACEMENT
arguments -- margin, z, individual center coordinates -- which are
never stated in the text at all, so there is no requirement to assign
them from).

Two confirmed real failure patterns this closes:

1. A single hole-PATTERN op whose regressed margin/z placement
   misses the actual built material entirely (confirmed live: "A
   bracket with a 5mm thick base and two 6mm mounting holes" predicts
   drill_hole_pattern(pattern="linear_array", margin=17.2, ...) against
   an l_shape profile -- 0 real holes detected, euler_number unchanged
   from the holeless body, despite count=2 being correctly assigned).
   Repaired by retrying with a progressively SAFER, directly-computed
   margin (never a regressor guess) -- bounded to `max_tries`.

2. Several INDIVIDUAL hole ops (drill_hole, not a pattern) whose
   regressed center coordinates land nowhere near the actual body at
   all (confirmed live: "A 70mm by 40mm plate with ... three 8mm
   holes" predicts 4 separate drill_hole calls -- one more than
   requested -- with centers like y=-34mm on a 40mm-wide plate,
   detected_hole_count=0). Trimming the extra op would not fix
   anything here (every one of the 4 is equally mispositioned, not
   just the extra one) -- repaired instead by REPLACING all of them
   with one drill_hole_pattern call, using the already-correctly-
   assigned diameter/count and a directly-computed, safe margin.

A third, separate function (repair_missing_stepped_section) handles a
confirmed Problem-1/Problem-2 BOUNDARY case the brief's own Part 10
anticipates: "A shaft with a 20mm diameter main section, 12mm diameter
stepped section, and 100mm overall length" has both op_arbitration.
arbitrate() AND op_requirements.extract_requirements() already
correctly recognize this text needs a stepped_section feature no op in
the predicted (add_cylinder, finalize) program can satisfy --
arbitration's own outcome is honestly NO_VALID_PLAN -- but nothing
downstream previously ACTED on that disclosure, so a plain cylinder
was built anyway and main_diameter/reduced_diameter had nowhere to be
assigned at all. Per Part 10's own instruction ("make only the changes
necessary to preserve the requirements"), this is a narrowly-scoped,
disclosed op swap -- never a general "try other ops" fallback -- to
the one specific op (add_shouldered_cylinder) that can satisfy a
stepped_section requirement, triggered only when the text's own
extracted dimensions already confirm it needs exactly that.

Never invents a feature the text didn't ask for and never changes a
dimension dimension_assignment.py already correctly assigned -- only
the PLACEMENT strategy (or, for the stepped-section case, the base op
itself) for an already-REQUESTED feature."""

from __future__ import annotations

from dataclasses import dataclass, field

from dimension_requirements import DimensionRequirement, FeatureCountRequirement, RequirementStatus
from evaluation.geometry_eval import evaluate_mesh
from sandbox import Sandbox, ToolCall

_PATTERN_OPS_WITH_MARGIN = frozenset({"drill_hole_pattern", "counterbore_hole_pattern", "add_slot_pattern"})
_INDIVIDUAL_HOLE_OPS = frozenset({"drill_hole", "counterbore_hole", "add_boss_with_bore"})
_MARGIN_RETRY_FRACTIONS = (0.5, 0.3, 0.15)  # progressively tighter -- a smaller margin means a LARGER usable placement span, more likely to land on real material


@dataclass
class RepairAttempt:
    strategy: str
    description: str
    improved: bool


@dataclass
class RepairResult:
    program: list[ToolCall]
    detected_count: "int | None"
    attempts: list[RepairAttempt] = field(default_factory=list)

    @property
    def was_repaired(self) -> bool:
        return any(a.improved for a in self.attempts)


def _set_arg(program: list[ToolCall], index: int, name: str, value) -> list[ToolCall]:
    updated = list(program)
    call = updated[index]
    updated[index] = ToolCall(op=call.op, args={**call.args, name: value})
    return updated


def _base_extents_mm(program: list[ToolCall]) -> "tuple[float, float] | None":
    """The base shape's own intended footprint (length, width), read
    straight from whichever add_box/add_extrusion args dimension_
    assignment.py already assigned -- never from the regressor's own
    placement guess, so a repair's own margin is always computed from a
    TRUSTED number."""
    if not program:
        return None
    base = program[0]
    if base.op == "add_box":
        return float(base.args.get("length", 0.0)), float(base.args.get("width", 0.0))
    if base.op == "add_extrusion":
        return float(base.args.get("param_a", 0.0)), float(base.args.get("param_b", 0.0))
    return None


def _safe_margin_mm(program: list[ToolCall], fraction: float) -> float:
    extents = _base_extents_mm(program)
    if not extents or min(extents) <= 0:
        return 15.0 * fraction  # sandbox.py's own documented default, scaled down -- no trusted footprint to compute from
    return max(1.0, min(extents) * fraction / 2.0)


def _replace_individual_holes_with_pattern(
    program: list[ToolCall], hole_indices: list[int], radius_mm: float, depth_mm: float, count: int, pattern_kind: str, margin_mm: float,
) -> list[ToolCall]:
    kept = [call for i, call in enumerate(program) if i not in hole_indices and call.op != "finalize"]
    pattern_call = ToolCall(op="drill_hole_pattern", args={
        "target": len(kept) - 1, "pattern": pattern_kind, "count": count, "radius": radius_mm,
        "depth": depth_mm, "axis": "z", "margin": margin_mm, "circle_diameter": 0.0, "z": 0.0,
    })
    return kept + [pattern_call, ToolCall(op="finalize", args={"target": len(kept)})]


def repair_hole_count_mismatch(
    program: list[ToolCall], count_req: FeatureCountRequirement, max_tries: int = 3,
) -> RepairResult:
    """Never raises. Re-executes and re-measures after every attempt --
    an attempt is only kept if it strictly improves how close the
    detected count is to the requested one (never accepted just because
    it ran without error). Leaves count_req.status as FAILED (not
    silently reverted to its pre-repair value) when every attempt is
    exhausted without reaching the exact requested count -- an explicit,
    disclosed failure per Part 8's own "if repair is impossible, return
    an explicit failure" rule."""
    initial_result = Sandbox.execute(program)
    if initial_result.mesh is None:
        return RepairResult(program=program, detected_count=None)
    initial_detected = evaluate_mesh(initial_result.mesh).detected_hole_count
    if initial_detected == count_req.count:
        count_req.status = RequirementStatus.GEOMETRY_VERIFIED
        count_req.actual_count = initial_detected
        return RepairResult(program=program, detected_count=initial_detected)

    outcome = RepairResult(program=program, detected_count=initial_detected)
    best_program, best_detected = program, initial_detected

    # Applied FIRST, as a correction to the WORKING program every later
    # strategy builds on -- not as an independent, alternative attempt --
    # confirmed necessary: an l_shape/triangle profile's own missing
    # notch region breaks EVERY hole-placement strategy equally (a
    # pattern op's margin retry, and a from-scratch pattern built to
    # replace individually-placed holes, both still land positions in
    # the same missing material regardless), so a repair that only
    # tries the profile swap as a LAST, separate fallback never actually
    # combines it with whichever placement strategy the program needs.
    # Nothing in these prompts' own extracted requirements ever asked
    # for an L-shape or triangle (op_requirements.py's own base-geometry
    # extraction has no such signal) -- it was purely the classifier's
    # own categorical guess, not a stated requirement, so correcting it
    # here is fixing an unsupported default, never overriding something
    # the text actually asked for.
    if best_program[0].op == "add_extrusion" and best_program[0].args.get("profile_kind") in ("l_shape", "triangle"):
        original_profile = best_program[0].args.get("profile_kind")
        candidate = _set_arg(best_program, 0, "profile_kind", "rect")
        candidate = _set_arg(candidate, 0, "param_c", 0.0)
        result = Sandbox.execute(candidate)
        detected = evaluate_mesh(result.mesh).detected_hole_count if result.mesh is not None else None
        improved = detected is not None and abs(detected - count_req.count) < abs((best_detected if best_detected is not None else -1) - count_req.count)
        outcome.attempts.append(RepairAttempt(
            "swap_incompatible_profile_to_rect",
            f"base shape's profile_kind={original_profile!r} is not a stated requirement and its own notch/corner geometry is incompatible with rectangular hole placement -- retried as profile_kind='rect' -- detected {detected} vs requested {count_req.count}",
            improved,
        ))
        if improved or detected == initial_detected:
            # Even when not YET an improvement (the placement strategy
            # below still needs to run on top of this), a profile swap
            # that doesn't make things WORSE is kept as the new working
            # base -- later strategies need the rectangular profile to
            # have any chance of working at all.
            best_program, best_detected = candidate, detected

    pattern_index = next((i for i, c in enumerate(best_program) if c.op in _PATTERN_OPS_WITH_MARGIN), None)
    individual_indices = [i for i, c in enumerate(best_program) if c.op in _INDIVIDUAL_HOLE_OPS]

    if pattern_index is not None and best_detected != count_req.count:
        for fraction in _MARGIN_RETRY_FRACTIONS[:max_tries]:
            margin = _safe_margin_mm(best_program, fraction)
            candidate = _set_arg(best_program, pattern_index, "margin", margin)
            result = Sandbox.execute(candidate)
            detected = evaluate_mesh(result.mesh).detected_hole_count if result.mesh is not None else None
            improved = detected is not None and abs(detected - count_req.count) < abs((best_detected if best_detected is not None else -1) - count_req.count)
            outcome.attempts.append(RepairAttempt(
                "retry_pattern_with_safe_margin", f"retried program[{pattern_index}] with margin={margin:.1f}mm (computed from the base shape's own real footprint) -- detected {detected} vs requested {count_req.count}", improved,
            ))
            if improved:
                best_program, best_detected = candidate, detected
            if best_detected == count_req.count:
                break

    elif individual_indices and best_detected != count_req.count:
        radius = next((best_program[i].args.get("radius") or best_program[i].args.get("hole_radius") for i in individual_indices), None)
        depth = next((best_program[i].args.get("depth") for i in individual_indices), None)
        if radius is not None and depth is not None:
            pattern_kind = count_req.pattern_hint or "corner_array"
            margin = _safe_margin_mm(best_program, _MARGIN_RETRY_FRACTIONS[0])
            candidate = _replace_individual_holes_with_pattern(best_program, individual_indices, float(radius), float(depth), count_req.count, pattern_kind, margin)
            result = Sandbox.execute(candidate)
            detected = evaluate_mesh(result.mesh).detected_hole_count if result.mesh is not None else None
            improved = detected is not None and abs(detected - count_req.count) < abs((best_detected if best_detected is not None else -1) - count_req.count)
            outcome.attempts.append(RepairAttempt(
                "replace_individual_holes_with_pattern",
                f"replaced {len(individual_indices)} individually-placed hole op(s) (regressed centers landed off the built body) with one drill_hole_pattern(pattern={pattern_kind!r}, margin={margin:.1f}mm, computed from the base shape's own real footprint) -- detected {detected} vs requested {count_req.count}",
                improved,
            ))
            if improved:
                best_program, best_detected = candidate, detected

    outcome.program = best_program
    outcome.detected_count = best_detected
    count_req.actual_count = best_detected
    count_req.status = RequirementStatus.GEOMETRY_VERIFIED if best_detected == count_req.count else RequirementStatus.FAILED
    return outcome


@dataclass
class StepRepairResult:
    program: list[ToolCall]
    applied: bool
    description: str


def repair_missing_stepped_section(program: list[ToolCall], dimension_reqs: list[DimensionRequirement]) -> StepRepairResult:
    """Never raises. A no-op (applied=False) unless BOTH conditions hold:
    the chosen program's own base op is add_cylinder (the one wrong-op
    pattern this closes -- confirmed live, see this module's own
    docstring), AND the text's own extraction already produced a
    reduced_diameter requirement ALONGSIDE either a main_diameter or a
    plain outer_diameter one -- i.e. the text genuinely describes two
    different diameters for one shaft, not a guess this function
    invents on its own. The outer_diameter case matters because not
    every phrasing of the shaft's own wider section uses the word
    "main" (confirmed real: "a shaft 10mm diameter, 80mm long, with a
    stepped section 14mm diameter for 20mm" states the wider diameter
    as a bare, unqualified "10mm diameter," which extraction correctly
    tags outer_diameter -- the same tag an ordinary, non-stepped
    cylinder's diameter gets -- while the SEPARATE "stepped section
    14mm diameter" phrase is what actually signals a step exists here).
    Swaps the base op to add_shouldered_cylinder with a plausible
    starting split (the old radius/height carried over as a sensible
    seed, not a final value -- dimension_assignment.assign_requirements,
    run again by the caller after this, is what actually places the
    two stated diameters and the overall length correctly)."""
    if not program or program[0].op != "add_cylinder":
        return StepRepairResult(program=program, applied=False, description="base op is not add_cylinder -- nothing to repair")
    has_wide = any(r.parameter in ("main_diameter", "outer_diameter") and r.status is not RequirementStatus.UNRESOLVED for r in dimension_reqs)
    has_reduced = any(r.parameter == "reduced_diameter" and r.status is not RequirementStatus.UNRESOLVED for r in dimension_reqs)
    if not (has_wide and has_reduced):
        return StepRepairResult(program=program, applied=False, description="text does not state two distinct diameters for this shaft -- no stepped section to add")

    old = program[0]
    radius = float(old.args.get("radius", 10.0))
    height = float(old.args.get("height", 20.0))
    center = old.args.get("center", [0.0, 0.0, 0.0])
    new_base = ToolCall(op="add_shouldered_cylinder", args={
        "bottom_radius": radius, "bottom_height": height / 2.0,
        "top_radius": radius, "top_height": height / 2.0,
        "transition_height": max(height * 0.1, 1.0), "center": list(center),
    })
    return StepRepairResult(
        program=[new_base] + list(program[1:]), applied=True,
        description=(
            "the text states two distinct diameters for this shaft (a required stepped_section -- see "
            "op_requirements.py's own extraction) but the chosen program's base op (add_cylinder) has no "
            "second diameter to assign one to -- swapped to add_shouldered_cylinder, the one op that can "
            "represent a stepped shaft, so dimension_assignment can place both diameters and the overall length"
        ),
    )
