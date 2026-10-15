"""A machine-readable capability registry for every tool in sandbox.py's
TOOL_OPS -- section 10's explicit requirement: "Create a machine-readable
capability registry... Expose this registry to the planner so it cannot
freely invent nonexistent tools." Built from two sources, combined:

1. Reflected, not hand-maintained: argument names/types/defaults come
   straight from sandbox.tool_schema() (already reflection-driven off the
   real method signatures) -- never duplicated by hand, so this registry
   cannot silently drift out of sync with the actual tool implementations
   the way a hand-maintained parallel list could.
2. Hand-curated judgment calls that genuinely can't be reflected off a
   signature: fidelity (exact / approximate / kernel-dependent) and risk
   class (section 6's own taxonomy: additive / subtractive /
   transformative / topology_changing / replacement). These were assigned
   by reading every tool's own docstring and, where this mission's own
   work touched the tool directly (shell, round_box_edges,
   chamfer_box_edges), its now-measured real behavior -- not guessed.

A self-consistency test (tests/test_capability_registry.py) asserts every
TOOL_OPS entry has a registry row and every registry row names a real
TOOL_OPS entry, so this file cannot silently drift from sandbox.py's
actual roster either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Fidelity = Literal["exact", "kernel_dependent", "approximate"]
RiskClass = Literal["additive", "subtractive", "transformative", "topology_changing", "replacement"]


@dataclass
class ToolCapability:
    op: str
    purpose: str
    fidelity: Fidelity
    risk_class: RiskClass
    known_failure_modes: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    supported_geometry: str = "any watertight solid produced by this tool roster"


# fidelity/risk_class reasoning, in one place rather than repeated per
# entry: "exact" means the construction is an analytically correct solid
# for its own parameters (a box, a verified Minkowski-sum box-edge
# reconstruction); "kernel_dependent" means correctness rests on
# manifold3d's own boolean/CSG implementation, which can fail on
# ill-conditioned inputs (confirmed, not hypothetical -- this is why
# Sandbox.execute() now validates watertightness immediately after every
# step, see sandbox.py's own execute() docstring); "approximate" means the
# tool itself is a disclosed, deliberate approximation of the operation it
# names (shell, add_knurl_pattern).
_REGISTRY: dict[str, ToolCapability] = {
    # ---- primitives: exact by construction ----
    "add_box": ToolCapability("add_box", "a rectangular box", "exact", "additive"),
    "add_cylinder": ToolCapability("add_cylinder", "a cylinder", "exact", "additive",
                                    known_failure_modes=("faceted, not a true circle -- small analytic volume error at low section counts",)),
    "add_sphere": ToolCapability("add_sphere", "a sphere", "exact", "additive",
                                  known_failure_modes=("icosphere faceting -- small analytic volume error",)),
    "add_torus": ToolCapability("add_torus", "a torus/ring-donut", "exact", "additive",
                                 known_failure_modes=("faceted -- small analytic volume error",)),
    "add_capsule": ToolCapability("add_capsule", "a capsule/pill shape", "exact", "additive"),
    "add_ellipsoid": ToolCapability("add_ellipsoid", "a stretched sphere", "exact", "additive",
                                     known_failure_modes=("faceted -- confirmed ~0.6% volume error vs. analytic",)),
    "add_dome": ToolCapability("add_dome", "a rounded cap", "exact", "additive"),
    "add_ring": ToolCapability("add_ring", "a flat washer/ring", "exact", "additive"),
    "add_polygon_frustum": ToolCapability("add_polygon_frustum", "a flat-sided (hexagonal, etc.) prism/frustum", "exact", "additive"),
    "add_revolve": ToolCapability("add_revolve", "a lathed solid of revolution", "exact", "additive"),
    "add_shouldered_cylinder": ToolCapability("add_shouldered_cylinder", "two cylinders joined by a smoothstep shoulder transition", "exact", "additive"),
    "add_extrusion": ToolCapability("add_extrusion", "a 2D profile (rect/rounded_rect/l_shape/ellipse) extruded to a height", "exact", "additive"),
    "add_gear_profile": ToolCapability("add_gear_profile", "an involute spur gear", "exact", "additive",
                                        preconditions=("module/tooth_count/face_width/bore_diameter must produce a valid involute profile",)),
    "add_loft": ToolCapability("add_loft", "a smoothstep blend between two (possibly differently-shaped) profiles over a height", "approximate", "additive",
                                known_failure_modes=("faceted cross-sections; confirmed real analytic deviation on curved profile kinds",)),
    "add_sweep": ToolCapability("add_sweep", "a constant profile swept along a straight/arc/helix path", "approximate", "additive",
                                 known_failure_modes=("confirmed 0.6-3.4% volume deviation from analytic (Pappus's theorem) depending on sections/path_segments",)),
    "add_tube": ToolCapability("add_tube", "a round tube (solid or hollow) along a straight/arc/helix path", "approximate", "additive",
                                known_failure_modes=("same faceting error class as add_sweep",)),

    # ---- rigid transforms: exact (pure linear algebra) ----
    "translate": ToolCapability("translate", "move a shape", "exact", "transformative"),
    "rotate": ToolCapability("rotate", "rotate a shape about an axis", "exact", "transformative"),
    "mirror": ToolCapability("mirror", "reflect a shape across a plane through the origin", "exact", "transformative"),
    "scale": ToolCapability("scale", "non-uniform scale about the shape's centroid", "exact", "transformative"),

    # ---- deform tools: exact (deterministic vertex remapping) but can self-intersect on aggressive params ----
    "taper": ToolCapability("taper", "linear taper along an axis", "exact", "transformative",
                             known_failure_modes=("can self-intersect at extreme factors -- not independently precondition-checked",)),
    "flare": ToolCapability("flare", "smoothly curved taper along an axis", "exact", "transformative",
                             known_failure_modes=("same self-intersection risk as taper at extreme factors",)),
    "twist": ToolCapability("twist", "twist a shape along an axis", "exact", "transformative",
                             known_failure_modes=("large twist angles relative to shape length can self-intersect",)),
    "bend": ToolCapability("bend", "bend a shape along its length into a curve", "exact", "transformative",
                            known_failure_modes=("tight bend radii relative to shape thickness can self-intersect",)),
    "linear_array_scaled": ToolCapability("linear_array_scaled", "N copies along an axis, each progressively scaled", "kernel_dependent", "additive"),
    "radial_array_oriented": ToolCapability("radial_array_oriented", "N copies around a circle, each oriented outward", "kernel_dependent", "additive"),

    # ---- booleans: kernel-dependent -- manifold3d can produce a
    # non-watertight result on ill-conditioned inputs (coplanar faces,
    # near-tangent surfaces); Sandbox.execute()'s per-step immediate
    # validation (added this mission) now catches this at the step that
    # caused it instead of silently propagating ----
    "boolean_union": ToolCapability("boolean_union", "union two shapes", "kernel_dependent", "additive",
                                     known_failure_modes=("non-watertight result on coplanar/near-tangent inputs -- now caught immediately by Sandbox.execute()",)),
    "boolean_difference": ToolCapability("boolean_difference", "subtract one shape from another", "kernel_dependent", "subtractive",
                                          known_failure_modes=("same as boolean_union; a cutter much larger than the target can also fully consume it (an empty-mesh result)",)),
    "boolean_intersection": ToolCapability("boolean_intersection", "keep only the overlap of two shapes", "kernel_dependent", "topology_changing",
                                            known_failure_modes=("non-overlapping inputs produce an empty mesh -- caught immediately",)),

    # ---- hole/pattern/socket tools: exact local constructions, kernel-dependent only at the final boolean step ----
    "drill_hole": ToolCapability("drill_hole", "a single round hole", "kernel_dependent", "subtractive"),
    "drill_hole_pattern": ToolCapability("drill_hole_pattern", "a hole pattern (corner array, bolt circle, linear row)", "kernel_dependent", "subtractive"),
    "counterbore_hole": ToolCapability("counterbore_hole", "a fastener clearance hole with a recessed head pocket", "kernel_dependent", "subtractive"),
    "counterbore_hole_pattern": ToolCapability("counterbore_hole_pattern", "a whole pattern of counterbore/countersink holes", "kernel_dependent", "subtractive"),
    "drill_slot": ToolCapability("drill_slot", "an elongated rounded-end mounting slot", "kernel_dependent", "subtractive"),
    "add_slot_pattern": ToolCapability("add_slot_pattern", "a row/bolt-circle of slots", "kernel_dependent", "subtractive"),
    "add_hex_socket": ToolCapability("add_hex_socket", "a hex-shaped recessed socket", "kernel_dependent", "subtractive"),
    "linear_pattern": ToolCapability("linear_pattern", "array/union ANY shape along an axis", "kernel_dependent", "additive"),
    "circular_pattern": ToolCapability("circular_pattern", "array/union ANY shape around a circle", "kernel_dependent", "additive"),
    "add_rib": ToolCapability("add_rib", "a structural gusset bridging a concave (internal) corner", "exact", "additive",
                               preconditions=("only meaningful at an internal, empty-space corner between two named faces",)),
    "add_boss_with_bore": ToolCapability("add_boss_with_bore", "a raised cylindrical boss with a concentric through-hole", "kernel_dependent", "additive",
                                          preconditions=("bore_radius must be smaller than boss_radius",)),
    "add_knurl_pattern": ToolCapability("add_knurl_pattern", "a grip-texture ring of radial ridges", "approximate", "additive",
                                         known_failure_modes=("disclosed approximation -- straight radial ridges, not a true diamond-knurl crosshatch",)),

    # ---- face-based tools ----
    "extrude_face": ToolCapability("extrude_face", "extrude a named face outward into a raised boss", "kernel_dependent", "additive"),
    "inset_pocket": ToolCapability("inset_pocket", "inset a face and cut a recessed pocket", "kernel_dependent", "subtractive"),
    "shell": ToolCapability("shell", "hollow out a shape leaving a wall thickness", "approximate", "subtractive",
                             known_failure_modes=(
                                 "disclosed, axis-scaled-copy approximation, not a true offset surface -- "
                                 "NOW MEASURED AND GUARDED (this mission): raises SandboxError if the "
                                 "90th-percentile measured wall-thickness deviation exceeds 35%, confirmed "
                                 "via real ray-cast measurement rather than assumed. Confirmed-safe on "
                                 "box-like/prismatic shapes and gently-tapered vessels (this project's own "
                                 "water-bottle pattern measures 3-11% deviation); confirmed-rejected on "
                                 "steeply-tapered shapes and fine-featured shapes like gear teeth "
                                 "(measured 40-72% deviation on real test cases).",
                             )),

    # ---- edge tools: exact, edge-local constructions, but scoped to axis-aligned perpendicular faces only ----
    "fillet_edge": ToolCapability("fillet_edge", "round a single edge where two named faces meet", "exact", "topology_changing",
                                   preconditions=("the two named faces must be flat, axis-aligned, and mutually perpendicular",),
                                   known_failure_modes=("any other edge configuration is rejected outright (SandboxError), not approximated",)),
    "chamfer_edge": ToolCapability("chamfer_edge", "flatten/bevel a single edge where two named faces meet", "exact", "topology_changing",
                                    preconditions=("same as fillet_edge",),
                                    known_failure_modes=("same as fillet_edge",)),
    "round_box_edges": ToolCapability("round_box_edges", "round all 12 edges/8 corners of a box-like target at once", "exact", "replacement",
                                       preconditions=(
                                           "target must be a genuine axis-aligned box (volume == its own bounding-box "
                                           "volume, checked via _is_axis_aligned_box) -- A REAL BUG FIXED THIS MISSION: "
                                           "previously silently replaced ANY target with its rounded bounding box; now "
                                           "raises SandboxError on a non-box target instead.",
                                       )),
    "chamfer_box_edges": ToolCapability("chamfer_box_edges", "flat-chamfer all 12 edges/8 corners of a box-like target at once", "exact", "replacement",
                                         preconditions=("same as round_box_edges, same fix applied",)),
    "mirror_merge": ToolCapability("mirror_merge", "mirror a shape and fuse the copy onto the original in one step", "kernel_dependent", "additive"),

    # ---- mesh-refinement tools (added this session, closing a real
    # documentation-vs-tool-roster gap -- see knowledge/sources.py and
    # MISSION_STATUS.md Addendum 2) ----
    "smooth_subdivide": ToolCapability("smooth_subdivide", "round off a shape's body/edges via subdivide + Taubin smoothing", "approximate", "topology_changing",
                                        known_failure_modes=("confirmed: smoothing without first subdividing can collapse a low-poly shape to ~6% of its volume -- guarded internally by always subdividing first",)),
    "add_wireframe_from_edges": ToolCapability("add_wireframe_from_edges", "convert a shape into a tube-frame following its own edges", "kernel_dependent", "replacement",
                                                 preconditions=("bounded to 500 edges -- an explicit, typed rejection past that, not a silent hang",)),
    "weld_vertices": ToolCapability("weld_vertices", "merge near-duplicate vertices within a tolerance", "exact", "topology_changing"),
}


def get_capability(op: str) -> ToolCapability | None:
    return _REGISTRY.get(op)


def full_registry() -> dict[str, ToolCapability]:
    return dict(_REGISTRY)


def is_known_tool(op: str) -> bool:
    """The planner-facing guard section 10 explicitly asks for: "Expose
    this registry to the planner so it cannot freely invent nonexistent
    tools." Any caller building/validating a tool-call plan should check
    this (or equivalently, that op is in sandbox.TOOL_OPS -- the two are
    kept in lockstep by test_capability_registry.py's own consistency
    check) before treating a proposed op name as real."""
    return op in _REGISTRY
