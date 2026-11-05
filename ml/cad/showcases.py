"""The single source of truth for the landing page's hero showcases that are
too large to hand-maintain safely as inline JS (currently: the F1 car).

Confirmed real gap this closes: before this module existed, the F1 car's
34-entity definition lived ONLY as a hardcoded array inside
frontend/index.html, independently maintained from any other copy (an
earlier, much simpler prototype in scripts/build_f1_car_example.py had
already drifted into a different, inferior shape -- different part count,
no relationships, nothing in common with this one beyond being "an F1
car"). Any future fix to the car's own geometry (another joint-gap fix,
the kind the axial-contact-gap audit already did once for the Nose) would
have had to be hand-copied into that JS array to ever reach the landing
page, with no mechanism forcing that to happen -- a second, parallel place
for "the real F1 car definition" to silently go stale against.

Now there is exactly one place this definition lives: here. server.py's
GET /api/showcases/<key> serves it as JSON; frontend/index.html fetches it
at build time instead of embedding its own copy (see buildShowcase() in
index.html -- showcases with "remote: true" fetch from this endpoint
first, then POST the result to /api/execute_programs exactly as every
other showcase already does). The landing page is therefore always
whatever is currently in this file, with no separate frontend-side edit
ever required or possible to forget.

Shape: a dict value is the same {label, upAxis?, entities: [...]} shape
/api/execute_programs's own "entities" field already expects per-entity
(name, position_mm as {x,y,z}, program as a list of {op, args} tool
calls, optional rotation_deg, optional relationship) -- see
server.py's _parse_position/_parse_rotation/_parse_entity_relationship,
which this endpoint's consumer (the frontend) ultimately feeds right back
into, unchanged, as its own request body.
"""

from __future__ import annotations

from typing import Any

# Built at the real target dimensions of a late-1960s/early-70s F1 car
# (2450mm wheelbase, 1500/1450mm front/rear track, 600/650mm front/rear
# tire diameters, etc.) -- converted from that spec's own X-forward/
# Y-left/Z-up convention into this showcase's established X-width/
# Y-height/Z-length one. Every part was verified against the live
# /api/execute_programs backend (watertight, correctly positioned, no
# floating/disconnected pieces -- confirmed by checking each neighboring
# pair's bounding boxes for a real zero-gap touch, not just eyeballed)
# before being recorded here. Ported byte-for-byte (every op/arg/position)
# from frontend/index.html's own prior hardcoded copy -- not redesigned --
# see this module's own docstring for why that copy no longer exists.
_F1_CAR_ENTITIES: list[dict[str, Any]] = [
    {"name": "ChassisMid", "position_mm": {"x": 0, "y": 415, "z": 100}, "program": [
        {"op": "add_loft", "args": {"bottom_profile": "rounded_rect", "bottom_param_a": 850, "bottom_param_b": 470, "bottom_param_c": 60, "top_profile": "rounded_rect", "top_param_a": 850, "top_param_b": 470, "top_param_c": 60, "height": 850, "sections": 28}},
        {"op": "add_loft", "args": {"bottom_profile": "ellipse", "bottom_param_a": 260, "bottom_param_b": 150, "bottom_param_c": 0, "top_profile": "ellipse", "top_param_a": 260, "top_param_b": 150, "top_param_c": 0, "height": 850, "sections": 24, "center": [0, 235, 0]}},
        {"op": "boolean_difference", "args": {"a": 0, "b": 1}},
        {"op": "finalize", "args": {"target": 2}},
    ]},
    {"name": "ChassisFront", "position_mm": {"x": 0, "y": 415, "z": 987.5}, "program": [
        {"op": "add_loft", "args": {"bottom_profile": "rounded_rect", "bottom_param_a": 850, "bottom_param_b": 470, "bottom_param_c": 60, "top_profile": "rounded_rect", "top_param_a": 320, "top_param_b": 300, "top_param_c": 50, "height": 925, "sections": 28}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "ChassisRear", "position_mm": {"x": 0, "y": 415, "z": -737.5}, "program": [
        {"op": "add_loft", "args": {"bottom_profile": "rounded_rect", "bottom_param_a": 650, "bottom_param_b": 400, "bottom_param_c": 50, "top_profile": "rounded_rect", "top_param_a": 850, "top_param_b": 470, "top_param_c": 60, "height": 825, "sections": 28}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    # position_mm kept only as the pre-fix fallback value (used only if the
    # relationship below somehow fails to resolve); abutted_to computes the
    # real flush position (z=2025) directly from both lofts' own actual
    # built geometry instead, closing a confirmed 550mm/81%-of-volume
    # overlap against ChassisFront that the hand-placed z=1475 alone had.
    {"name": "Nose", "position_mm": {"x": 0, "y": 415, "z": 1475}, "program": [
        {"op": "add_loft", "args": {"bottom_profile": "rounded_rect", "bottom_param_a": 320, "bottom_param_b": 300, "bottom_param_c": 50, "top_profile": "ellipse", "top_param_a": 34, "top_param_b": 48, "top_param_c": 0, "height": 1150, "sections": 28}},
        {"op": "finalize", "args": {"target": 0}},
    ], "relationship": {"kind": "abutted_to", "target_id": "ChassisFront", "axis": "z", "abutment_direction": "+"}},
    {"name": "EngineCover", "position_mm": {"x": 0, "y": 550, "z": -770}, "program": [
        {"op": "add_loft", "args": {"bottom_profile": "rounded_rect", "bottom_param_a": 180, "bottom_param_b": 180, "bottom_param_c": 40, "top_profile": "rounded_rect", "top_param_a": 650, "top_param_b": 420, "top_param_c": 55, "height": 850, "sections": 24}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "AirIntake", "position_mm": {"x": 0, "y": 650, "z": -120}, "program": [
        {"op": "add_loft", "args": {"bottom_profile": "rounded_rect", "bottom_param_a": 220, "bottom_param_b": 180, "bottom_param_c": 40, "top_profile": "rounded_rect", "top_param_a": 160, "top_param_b": 130, "top_param_c": 30, "height": 160, "sections": 16}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "Seat", "position_mm": {"x": 0, "y": 420, "z": 50}, "program": [
        {"op": "add_box", "args": {"length": 650, "width": 420, "height": 60, "center": [0, 0, 0]}},
        {"op": "add_box", "args": {"length": 60, "width": 420, "height": 420, "center": [-295, 150, 0]}},
        {"op": "rotate", "args": {"target": 1, "axis": "x", "angle_deg": -25}},
        {"op": "boolean_union", "args": {"a": 0, "b": 2}},
        {"op": "finalize", "args": {"target": 3}},
    ]},
    {"name": "DriverTorso", "position_mm": {"x": 0, "y": 560, "z": 10}, "program": [
        {"op": "add_capsule", "args": {"radius": 130, "height": 420, "center": [0, 0, 0], "axis": "y"}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "DriverHelmet", "position_mm": {"x": 0, "y": 800, "z": 100}, "program": [
        {"op": "add_sphere", "args": {"radius": 125, "center": [0, 0, 0]}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "DriverArms", "position_mm": {"x": 0, "y": 640, "z": 220}, "program": [
        {"op": "add_capsule", "args": {"radius": 45, "height": 320, "center": [120, 0, 0], "axis": "z"}},
        {"op": "add_capsule", "args": {"radius": 45, "height": 320, "center": [-120, 0, 0], "axis": "z"}},
        {"op": "boolean_union", "args": {"a": 0, "b": 1}},
        {"op": "finalize", "args": {"target": 2}},
    ]},
    {"name": "SteeringWheel", "position_mm": {"x": 0, "y": 600, "z": 420}, "program": [
        {"op": "add_torus", "args": {"major_radius": 128, "minor_radius": 12, "center": [0, 0, 0], "axis": "z"}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "SteeringColumn", "position_mm": {"x": 0, "y": 540, "z": 540}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 14, "inner_radius": 9, "path_kind": "straight", "path_param_a": 260, "path_param_b": 0, "path_param_c": 0, "sections": 10, "path_segments": 6}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    # a straight add_tube rod has only ONE fixed (y, z) for its entire
    # length -- these 8 arms' outboard ends are re-aimed to land inside
    # their own hub cylinder (+-20mm of the hub's own center, so upper and
    # lower still read as two separate arms, not one fused rod) rather
    # than the wishbone-angle y/z a straight rod can't actually reach.
    {"name": "FrontUpperArmL", "position_mm": {"x": -585, "y": 320, "z": 1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 16, "inner_radius": 11, "path_kind": "straight", "path_param_a": 330, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "FrontUpperArmR", "position_mm": {"x": 585, "y": 320, "z": 1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 16, "inner_radius": 11, "path_kind": "straight", "path_param_a": 330, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "FrontLowerArmL", "position_mm": {"x": -585, "y": 280, "z": 1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 20, "inner_radius": 14, "path_kind": "straight", "path_param_a": 330, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "FrontLowerArmR", "position_mm": {"x": 585, "y": 280, "z": 1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 20, "inner_radius": 14, "path_kind": "straight", "path_param_a": 330, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "RearUpperArmL", "position_mm": {"x": -572.5, "y": 345, "z": -1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 17, "inner_radius": 11, "path_kind": "straight", "path_param_a": 305, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "RearUpperArmR", "position_mm": {"x": 572.5, "y": 345, "z": -1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 17, "inner_radius": 11, "path_kind": "straight", "path_param_a": 305, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "RearLowerArmL", "position_mm": {"x": -572.5, "y": 305, "z": -1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 22, "inner_radius": 15, "path_kind": "straight", "path_param_a": 305, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "RearLowerArmR", "position_mm": {"x": 572.5, "y": 305, "z": -1225}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 22, "inner_radius": 15, "path_kind": "straight", "path_param_a": 305, "path_param_b": 0, "path_param_c": 0, "sections": 12, "path_segments": 6}},
        {"op": "rotate", "args": {"target": 0, "axis": "y", "angle_deg": 90}},
        {"op": "finalize", "args": {"target": 1}},
    ]},
    {"name": "TireFL", "position_mm": {"x": -750, "y": 300, "z": 1225}, "program": [
        {"op": "add_torus", "args": {"major_radius": 210, "minor_radius": 90, "center": [0, 0, 0], "axis": "x"}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "TireFR", "position_mm": {"x": 750, "y": 300, "z": 1225}, "program": [
        {"op": "add_torus", "args": {"major_radius": 210, "minor_radius": 90, "center": [0, 0, 0], "axis": "x"}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "TireRL", "position_mm": {"x": -725, "y": 325, "z": -1225}, "program": [
        {"op": "add_torus", "args": {"major_radius": 185, "minor_radius": 140, "center": [0, 0, 0], "axis": "x"}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "TireRR", "position_mm": {"x": 725, "y": 325, "z": -1225}, "program": [
        {"op": "add_torus", "args": {"major_radius": 185, "minor_radius": 140, "center": [0, 0, 0], "axis": "x"}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
    {"name": "RimFL", "position_mm": {"x": -750, "y": 300, "z": 1225}, "program": [
        {"op": "add_box", "args": {"length": 100, "width": 16, "height": 78, "center": [95, 0, 0]}},
        {"op": "radial_array_oriented", "args": {"target": 0, "count": 8, "radius": 0}},
        {"op": "rotate", "args": {"target": 1, "axis": "y", "angle_deg": 90}},
        {"op": "add_cylinder", "args": {"radius": 45, "height": 130, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 2, "b": 3}},
        {"op": "add_ring", "args": {"outer_radius": 165, "inner_radius": 145, "height": 130, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 4, "b": 5}},
        {"op": "finalize", "args": {"target": 6}},
    ]},
    {"name": "RimFR", "position_mm": {"x": 750, "y": 300, "z": 1225}, "program": [
        {"op": "add_box", "args": {"length": 100, "width": 16, "height": 78, "center": [95, 0, 0]}},
        {"op": "radial_array_oriented", "args": {"target": 0, "count": 8, "radius": 0}},
        {"op": "rotate", "args": {"target": 1, "axis": "y", "angle_deg": 90}},
        {"op": "add_cylinder", "args": {"radius": 45, "height": 130, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 2, "b": 3}},
        {"op": "add_ring", "args": {"outer_radius": 165, "inner_radius": 145, "height": 130, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 4, "b": 5}},
        {"op": "finalize", "args": {"target": 6}},
    ]},
    {"name": "RimRL", "position_mm": {"x": -725, "y": 325, "z": -1225}, "program": [
        {"op": "add_box", "args": {"length": 120, "width": 16, "height": 120, "center": [110, 0, 0]}},
        {"op": "radial_array_oriented", "args": {"target": 0, "count": 8, "radius": 0}},
        {"op": "rotate", "args": {"target": 1, "axis": "y", "angle_deg": 90}},
        {"op": "add_cylinder", "args": {"radius": 50, "height": 200, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 2, "b": 3}},
        {"op": "add_ring", "args": {"outer_radius": 190, "inner_radius": 170, "height": 200, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 4, "b": 5}},
        {"op": "finalize", "args": {"target": 6}},
    ]},
    {"name": "RimRR", "position_mm": {"x": 725, "y": 325, "z": -1225}, "program": [
        {"op": "add_box", "args": {"length": 120, "width": 16, "height": 120, "center": [110, 0, 0]}},
        {"op": "radial_array_oriented", "args": {"target": 0, "count": 8, "radius": 0}},
        {"op": "rotate", "args": {"target": 1, "axis": "y", "angle_deg": 90}},
        {"op": "add_cylinder", "args": {"radius": 50, "height": 200, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 2, "b": 3}},
        {"op": "add_ring", "args": {"outer_radius": 190, "inner_radius": 170, "height": 200, "axis": "x"}},
        {"op": "boolean_union", "args": {"a": 4, "b": 5}},
        {"op": "finalize", "args": {"target": 6}},
    ]},
    {"name": "FrontWing", "position_mm": {"x": 0, "y": 170, "z": 1850}, "program": [
        {"op": "add_box", "args": {"length": 1700, "width": 25, "height": 100, "center": [0, 0, 0]}},
        {"op": "add_box", "args": {"length": 20, "width": 150, "height": 260, "center": [-850, 62.5, 0]}},
        {"op": "boolean_union", "args": {"a": 0, "b": 1}},
        {"op": "add_box", "args": {"length": 20, "width": 150, "height": 260, "center": [850, 62.5, 0]}},
        {"op": "boolean_union", "args": {"a": 2, "b": 3}},
        {"op": "finalize", "args": {"target": 4}},
    ]},
    {"name": "FrontWingSupports", "position_mm": {"x": 0, "y": 330, "z": 1700}, "program": [
        {"op": "add_box", "args": {"length": 20, "width": 180, "height": 110, "center": [160, 0, 0]}},
        {"op": "add_box", "args": {"length": 20, "width": 180, "height": 110, "center": [-160, 0, 0]}},
        {"op": "boolean_union", "args": {"a": 0, "b": 1}},
        {"op": "finalize", "args": {"target": 2}},
    ]},
    {"name": "RearWing", "position_mm": {"x": 0, "y": 800, "z": -1650}, "program": [
        {"op": "add_box", "args": {"length": 1250, "width": 25, "height": 300, "center": [0, 0, 0]}},
        {"op": "add_box", "args": {"length": 20, "width": 200, "height": 340, "center": [-625, 20, 0]}},
        {"op": "boolean_union", "args": {"a": 0, "b": 1}},
        {"op": "add_box", "args": {"length": 20, "width": 200, "height": 340, "center": [625, 20, 0]}},
        {"op": "boolean_union", "args": {"a": 2, "b": 3}},
        {"op": "finalize", "args": {"target": 4}},
    ]},
    {"name": "RearWingSupports", "position_mm": {"x": 0, "y": 705, "z": -1337.5}, "program": [
        {"op": "add_box", "args": {"length": 26, "width": 110, "height": 290, "center": [150, 0, 0]}},
        {"op": "add_box", "args": {"length": 26, "width": 110, "height": 290, "center": [-150, 0, 0]}},
        {"op": "boolean_union", "args": {"a": 0, "b": 1}},
        {"op": "finalize", "args": {"target": 2}},
    ]},
    {"name": "Exhausts", "position_mm": {"x": 0, "y": 350, "z": -950}, "program": [
        {"op": "add_tube", "args": {"outer_radius": 22, "inner_radius": 16, "path_kind": "arc", "path_param_a": 180, "path_param_b": 70, "path_param_c": 0, "sections": 14, "path_segments": 16}},
        {"op": "translate", "args": {"target": 0, "dx": 300, "dy": 0, "dz": 0}},
        {"op": "add_tube", "args": {"outer_radius": 22, "inner_radius": 16, "path_kind": "arc", "path_param_a": 180, "path_param_b": 70, "path_param_c": 0, "sections": 14, "path_segments": 16}},
        {"op": "translate", "args": {"target": 2, "dx": -300, "dy": 0, "dz": 0}},
        {"op": "boolean_union", "args": {"a": 1, "b": 3}},
        {"op": "finalize", "args": {"target": 4}},
    ]},
    {"name": "Diffuser", "position_mm": {"x": 0, "y": 275, "z": -1425}, "program": [
        {"op": "add_loft", "args": {"bottom_profile": "rect", "bottom_param_a": 700, "bottom_param_b": 15, "bottom_param_c": 0, "top_profile": "rect", "top_param_a": 600, "top_param_b": 130, "top_param_c": 0, "height": 500, "sections": 4}},
        {"op": "finalize", "args": {"target": 0}},
    ]},
]

# label/upAxis are served alongside entities so the frontend never needs
# its own hardcoded copy of either -- upAxis:'y' is a deliberate, disclosed
# exception to this project's usual Z-up convention: this car's own frame
# (X=width, Y=height, Z=length) is Y-up, the backend script this was
# originally ported from had no single enforced "up" axis, and re-building
# the whole car in a different convention is out of scope of "serve the
# existing definition from one place."
SHOWCASES: dict[str, dict[str, Any]] = {
    "f1car": {"label": "Retro F1 car", "upAxis": "y", "entities": _F1_CAR_ENTITIES},
}


def get_showcase(key: str) -> "dict[str, Any] | None":
    return SHOWCASES.get(key)
