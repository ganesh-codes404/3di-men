"""The explicit assignment stage between extraction (dimension_
requirements.py) and operation planning -- determines WHICH (operation
index, argument name) each extracted DimensionRequirement/
FeatureCountRequirement actually targets, and overrides that argument
with the requirement's own literal value.

Replaces program_model._override_primary_shape_dimensions's and
_override_bore_hole_dimensions's own positional, all-or-nothing literal-
to-field zipping (see dimension_requirements.py's own module docstring
for the two confirmed failures that approach produces) with semantic
matching: a requirement's own `parameter` tag (set by its own nearby
descriptive keyword, not its position in the text) is looked up against
OPERATION_PARAMETER_SEMANTICS for the op actually chosen, so a correctly
-identified single dimension is assigned even when the text doesn't
supply enough numbers to cover every slot an op happens to have, and a
feature's own number can never leak into an unrelated op's argument.

Deliberately does NOT replace _override_gear_dimensions (gear-specific,
untouched, out of this task's own reproduced-failure scope) and does
NOT touch op_arbitration.py/op_requirements.py (Problem 1's own
classifier/RAG arbitration, explicitly out of scope -- see this
project's own standing instruction to keep the two fixes isolated)."""

from __future__ import annotations

from dataclasses import dataclass, field

from dimension_requirements import DimensionRequirement, FeatureCountRequirement, RequirementStatus
from sandbox import ToolCall, blind_hole_center_offset

# ---------------- operation parameter semantics ----------------

# op -> {requirement_type: (arg_name, is_diameter_field)}. `arg_name` is
# the EXISTING sandbox.py tool argument this requirement_type maps onto
# -- deliberately not a renamed/new argument (sandbox.py's own tool
# signatures are unchanged; this registry is the missing SEMANTIC label
# connecting an engineering concept to the argument that already exists,
# see this module's own docstring for why a full rename was rejected).
# `is_diameter_field=True` means the text states a DIAMETER but this
# argument is a RADIUS (this project's own established convention, see
# program_model._override_primary_shape_dimensions's own docstring) --
# the requirement's normalized_value is halved before assignment.
OPERATION_PARAMETER_SEMANTICS: dict[str, dict[str, tuple[str, bool]]] = {
    "add_box": {
        "length": ("length", False), "width": ("width", False),
        "height": ("height", False), "thickness": ("height", False),
    },
    "add_extrusion": {
        "length": ("param_a", False), "width": ("param_b", False),
        "height": ("height", False), "thickness": ("height", False),
    },
    "add_cylinder": {
        "outer_diameter": ("radius", True), "height": ("height", False),
        "length": ("height", False),  # a horizontal "shaft" is still built as add_cylinder -- "length" and "height" both mean its one axial dimension
    },
    "add_shouldered_cylinder": {
        "main_diameter": ("bottom_radius", True), "outer_diameter": ("bottom_radius", True),
        "reduced_diameter": ("top_radius", True),
        # "overall_length" has no single matching field (it's the SUM of
        # bottom_height+top_height+transition_height) -- handled by
        # _assign_shouldered_cylinder_length below, not this table.
    },
    "add_sphere": {"outer_diameter": ("radius", True)},
    "add_torus": {"outer_diameter": ("major_radius", True)},
    "drill_hole": {"hole_diameter": ("radius", True), "hole_depth": ("depth", False)},
    "drill_hole_pattern": {"hole_diameter": ("radius", True), "hole_depth": ("depth", False), "edge_offset": ("margin", False)},
    "counterbore_hole": {"hole_diameter": ("hole_radius", True)},
    "counterbore_hole_pattern": {"hole_diameter": ("hole_radius", True), "edge_offset": ("margin", False)},
    "add_boss_with_bore": {"hole_diameter": ("bore_radius", True)},
    "add_slot_pattern": {"edge_offset": ("margin", False)},
}

_BASE_SHAPE_PARAMETER_TYPES = frozenset({"outer_diameter", "main_diameter", "reduced_diameter", "overall_length", "thickness", "height", "length", "width"})
_REPEATED_FEATURE_PARAMETER_TYPES = frozenset({"hole_diameter", "hole_depth"})
_PATTERN_OPS = frozenset({"drill_hole_pattern", "counterbore_hole_pattern", "add_slot_pattern", "circular_pattern", "linear_pattern"})
# Of _PATTERN_OPS, only these take a "pattern" CATEGORICAL argument
# (corner_array/linear_array/bolt_circle/single) -- circular_pattern/
# linear_pattern's own layout is fixed by their op NAME itself, no
# separate argument to override.
_OPS_WITH_PATTERN_ARG = frozenset({"drill_hole_pattern", "counterbore_hole_pattern", "add_slot_pattern"})
_HOLE_FEATURE_OPS = frozenset({"drill_hole", "drill_hole_pattern", "counterbore_hole", "counterbore_hole_pattern", "add_boss_with_bore"})


@dataclass
class AssignmentLogEntry:
    requirement_id: str
    outcome: str  # "ASSIGNED" | "UNASSIGNED" | "UNSUPPORTED" | "UNRESOLVED"
    detail: str


@dataclass
class AssignmentResult:
    program: list[ToolCall]
    dimension_requirements: list[DimensionRequirement]
    count_requirements: list[FeatureCountRequirement]
    log: list[AssignmentLogEntry] = field(default_factory=list)


def _set_arg(program: list[ToolCall], index: int, name: str, value: float) -> list[ToolCall]:
    updated = list(program)
    call = updated[index]
    updated[index] = ToolCall(op=call.op, args={**call.args, name: value})
    return updated


def _axial_extent_mm(base_call: ToolCall) -> "float | None":
    """The base shape's own extent along its principal (drilling) axis,
    read from whichever already-assigned dimension argument means that
    for this op -- never from a built mesh, since this runs before
    execution. None for an op this registry has no opinion about (the
    blind-hole center-offset computation is simply skipped then, same
    "never guess" discipline as everywhere else in this module)."""
    if base_call.op in ("add_cylinder", "add_box", "add_extrusion"):
        return float(base_call.args.get("height", 0.0)) or None
    return None


def _assign_shouldered_cylinder_length(program: list[ToolCall], index: int, requested_total_mm: float) -> list[ToolCall]:
    """add_shouldered_cylinder's own overall length is the SUM of three
    separate fields (bottom_height + top_height + transition_height),
    not one single argument -- rescales whatever the regressor already
    predicted for the three of them PROPORTIONALLY so their sum exactly
    equals the requested total, preserving the regressor's own relative
    split between the three sections (never inventing an arbitrary
    one) rather than guessing a fixed proportion out of thin air. Falls
    back to a plain 3-way even split only if the regressor's own guess
    was degenerate (sums to ~0)."""
    call = program[index]
    bottom_h = float(call.args.get("bottom_height", 0.0))
    top_h = float(call.args.get("top_height", 0.0))
    trans_h = float(call.args.get("transition_height", 0.0))
    current_total = bottom_h + top_h + trans_h
    if current_total <= 1e-6:
        bottom_h, top_h, trans_h = requested_total_mm / 3.0, requested_total_mm / 3.0, requested_total_mm / 3.0
    else:
        scale = requested_total_mm / current_total
        bottom_h, top_h, trans_h = bottom_h * scale, top_h * scale, trans_h * scale
    updated = list(program)
    updated[index] = ToolCall(op=call.op, args={**call.args, "bottom_height": bottom_h, "top_height": top_h, "transition_height": trans_h})
    return updated


def assign_requirements(
    dimension_requirements: list[DimensionRequirement],
    count_requirements: list[FeatureCountRequirement],
    program: list[ToolCall],
) -> AssignmentResult:
    """Never raises. Every requirement ends this function with a status
    of ASSIGNED, UNASSIGNED, UNSUPPORTED, or UNRESOLVED (never left at
    the initial EXTRACTED) -- see dimension_requirements.RequirementStatus
    for what each means. An UNRESOLVED requirement (extraction itself
    couldn't determine a parameter) is passed through untouched here --
    nothing downstream invents a resolution extraction didn't have."""
    result = AssignmentResult(program=list(program), dimension_requirements=list(dimension_requirements), count_requirements=list(count_requirements))

    # Track which op indices have already received a requirement of a
    # given requirement_type, so a SECOND requirement of the same type
    # (e.g. two differently-sized holes, a case these 10 tests don't
    # exercise but a real future prompt could) fans out to the NEXT
    # matching op instead of overwriting the first one's assignment.
    consumed: set[tuple[int, str]] = set()

    for req in result.dimension_requirements:
        if req.status is RequirementStatus.UNRESOLVED:
            result.log.append(AssignmentLogEntry(req.id, "UNRESOLVED", f"{req.source_phrase!r} -- no parameter could be determined at extraction time"))
            continue

        if req.parameter in _BASE_SHAPE_PARAMETER_TYPES:
            base_index = 0
            base_op = result.program[base_index].op if result.program else None
            if (base_index, req.parameter) in consumed:
                # Two DIFFERENT extracted mentions landed on the SAME
                # single-slot base parameter -- confirmed real case:
                # "a shaft 10mm diameter, ... with a stepped section
                # 14mm diameter for 20mm" extracts "outer_diameter" TWICE
                # (10mm from its own clause, and a spurious second match
                # on "20mm" from the trailing "for 20mm" -- a genuine
                # extraction ambiguity, not a bug in this function).
                # Silently letting the SECOND one overwrite the first
                # (whichever happened to be right) is exactly the kind
                # of silent-decision this brief's own Part 4 "preserve
                # ambiguity rather than pretend to resolve it" rule
                # forbids -- first-stated wins (same "explicit beats
                # inferred" precedence used throughout this project),
                # and the conflict itself is disclosed, never hidden.
                req.status = RequirementStatus.UNASSIGNED
                result.log.append(AssignmentLogEntry(
                    req.id, "UNASSIGNED",
                    f"{req.normalized_value:g}mm {req.parameter} conflicts with an already-assigned {req.parameter} on the same operation -- "
                    f"the first-stated value was kept, this one was not applied (ambiguous/duplicate extraction, not a decision this stage can safely make)",
                ))
                continue
            if req.parameter in ("overall_length", "length", "height") and base_op == "add_shouldered_cylinder":
                # "length"/"height" reach here too, not just the
                # explicit "overall_length" tag -- "80mm long" and
                # "100mm overall length" are the same physical concept
                # (a shaft's one axial dimension) once the base op is
                # specifically a stepped cylinder, which has no plain
                # single "height"/"length" field of its own to target
                # (only the 3-way bottom/top/transition split this
                # helper rescales).
                result.program = _assign_shouldered_cylinder_length(result.program, base_index, req.normalized_value)
                consumed.add((base_index, req.parameter))
                req.status = RequirementStatus.ASSIGNED
                req.assigned_op_index, req.assigned_arg_name = base_index, "bottom_height+top_height+transition_height"
                result.log.append(AssignmentLogEntry(req.id, "ASSIGNED", f"{req.normalized_value:g}mm {req.parameter} -> {base_op}'s 3 height fields, rescaled proportionally"))
                continue
            mapping = OPERATION_PARAMETER_SEMANTICS.get(base_op, {}).get(req.parameter) if base_op else None
            if mapping is None:
                req.status = RequirementStatus.UNASSIGNED
                result.log.append(AssignmentLogEntry(req.id, "UNASSIGNED", f"{req.normalized_value:g}mm {req.parameter} -- the chosen base operation ({base_op}) has no matching parameter"))
                continue
            target_arg, is_diameter = mapping
            value = req.normalized_value / 2.0 if is_diameter else req.normalized_value
            result.program = _set_arg(result.program, base_index, target_arg, value)
            consumed.add((base_index, req.parameter))
            req.status, req.assigned_op_index, req.assigned_arg_name = RequirementStatus.ASSIGNED, base_index, target_arg
            result.log.append(AssignmentLogEntry(req.id, "ASSIGNED", f"{req.normalized_value:g}mm {req.parameter} -> program[{base_index}] ({base_op}).{target_arg}"))
            continue

        # Feature-level parameter: a SINGLE stated value (e.g. "holes of
        # 6mm diameter") describes EVERY individual instance of a
        # repeated feature when the chosen program built that feature as
        # several separate ops rather than one pattern op (confirmed
        # real gap: 3 individual drill_hole calls for "three 8mm holes"
        # -- only the first one was getting the override, leaving the
        # other two at whatever the regressor separately guessed for
        # each). Applied to every unconsumed matching op, not just the
        # first, for hole_diameter/hole_depth specifically; every other
        # feature parameter (e.g. edge_offset/margin, naturally a single
        # pattern-level value) keeps the original first-match-only
        # behavior.
        matching_indices = [
            i for i, call in enumerate(result.program)
            if i != 0 and (i, req.parameter) not in consumed and OPERATION_PARAMETER_SEMANTICS.get(call.op, {}).get(req.parameter) is not None
        ]
        if req.parameter not in _REPEATED_FEATURE_PARAMETER_TYPES:
            matching_indices = matching_indices[:1]
        if not matching_indices:
            req.status = RequirementStatus.UNASSIGNED
            result.log.append(AssignmentLogEntry(req.id, "UNASSIGNED", f"{req.normalized_value:g}mm {req.parameter} -- no operation in the chosen program accepts this parameter"))
            continue

        target_arg, is_diameter = OPERATION_PARAMETER_SEMANTICS[result.program[matching_indices[0]].op][req.parameter]
        value = req.normalized_value / 2.0 if is_diameter else req.normalized_value
        for idx in matching_indices:
            result.program = _set_arg(result.program, idx, target_arg, value)
            consumed.add((idx, req.parameter))
        req.status, req.assigned_op_index, req.assigned_arg_name = RequirementStatus.ASSIGNED, matching_indices[0], target_arg
        targets_desc = ", ".join(f"program[{i}] ({result.program[i].op})" for i in matching_indices)
        result.log.append(AssignmentLogEntry(req.id, "ASSIGNED", f"{req.normalized_value:g}mm {req.parameter} -> {targets_desc}.{target_arg}"))

        if req.parameter == "hole_depth" and "bore_type=blind" in req.notes:
            # Setting `depth` ALONE never makes a hole blind -- see
            # sandbox.blind_hole_center_offset's own docstring: the
            # cutter's own length always floors at the target's full
            # extent (CSG robustness), so without a matching center
            # offset the cutter reaches both faces regardless of depth.
            # Confirmed real gap this closes (live-reproduced): "a 10mm
            # diameter blind hole, 8mm deep" on a 20mm-tall cylinder
            # built as a genuine THROUGH hole, because depth was
            # assigned correctly but center never was.
            base_extent = _axial_extent_mm(result.program[0])
            if base_extent is not None:
                offset = blind_hole_center_offset(base_extent, req.normalized_value)
                if offset > 0:
                    for idx in matching_indices:
                        axis = result.program[idx].args.get("axis", "z")
                        axis_pos = {"x": 0, "y": 1, "z": 2}[axis]
                        center = list(result.program[idx].args.get("center", [0.0, 0.0, 0.0]))
                        center[axis_pos] = offset
                        result.program = _set_arg(result.program, idx, "center", center)
                    result.log.append(AssignmentLogEntry(req.id, "ASSIGNED", f"blind hole -- also set center offset={offset:g}mm so depth={value:g}mm actually stops short of the far face"))

    for count_req in result.count_requirements:
        pattern_index = next((i for i, c in enumerate(result.program) if c.op in _PATTERN_OPS), None)
        if pattern_index is not None:
            result.program = _set_arg(result.program, pattern_index, "count", int(count_req.count))
            pattern_note = ""
            # The text's own layout wording ("one near each corner" / "equally
            # spaced along its centreline" / "around a bolt circle") overrides
            # whatever pattern TYPE the regressor separately guessed --
            # confirmed real gap this closes: "five holes equally spaced along
            # its centreline" (a linear row) predicted pattern="corner_array"
            # (which only ever places at a rectangle's 4 corners, structurally
            # incapable of 5 evenly-spaced positions) with no mechanism to
            # correct it.
            if count_req.pattern_hint and result.program[pattern_index].op in _OPS_WITH_PATTERN_ARG:
                result.program = _set_arg(result.program, pattern_index, "pattern", count_req.pattern_hint)
                pattern_note = f", pattern type -> {count_req.pattern_hint!r} (from the text's own layout wording)"
            count_req.status = RequirementStatus.ASSIGNED
            count_req.assigned_op_index = pattern_index
            result.log.append(AssignmentLogEntry(count_req.id, "ASSIGNED", f"count={count_req.count} -> program[{pattern_index}] ({result.program[pattern_index].op}).count{pattern_note}"))
            continue
        individual_holes = [i for i, c in enumerate(result.program) if c.op in _HOLE_FEATURE_OPS and c.op not in _PATTERN_OPS]
        if individual_holes:
            count_req.status = RequirementStatus.UNASSIGNED if len(individual_holes) != count_req.count else RequirementStatus.ASSIGNED
            if count_req.status is RequirementStatus.ASSIGNED:
                count_req.assigned_op_index = individual_holes[0]
            result.log.append(AssignmentLogEntry(
                count_req.id, count_req.status.value,
                f"requested count={count_req.count}, program has {len(individual_holes)} individual hole op(s) (no pattern op)",
            ))
            continue
        count_req.status = RequirementStatus.UNASSIGNED
        result.log.append(AssignmentLogEntry(count_req.id, "UNASSIGNED", f"requested count={count_req.count} -- no hole-producing operation at all in the chosen program"))

    return result
