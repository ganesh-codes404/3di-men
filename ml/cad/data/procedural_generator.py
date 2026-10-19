"""Procedural generator for simple mechanical parts, used as v1 ground-truth data.

Purpose (see docs/cad-finetuning-technical-report.md, Section 5): DeepCAD/Text2CAD
covers general parametric CAD but is weak on explicit real-world dimensions and
doesn't give us known-correct ground truth to validate the evaluation suite against.
This module fills that gap with a small set of parametrized mechanical primitives
where every geometric property (hole count, spacing, fillet radius, symmetry) is
known exactly at generation time.

Runs on CPU only (trimesh + numpy). No GPU, no network, no Hunyuan3D dependency.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from mesh_quality import remesh_flat_facets


@dataclass
class MountingPlateParams:
    """Parameters for a rectangular mounting plate with a rectilinear hole pattern.

    Matches the example in the task brief: a plate with `hole_count` holes placed
    symmetrically `hole_spacing` apart, `hole_edge_margin` in from each edge, with
    rounded corners of `fillet_radius`.
    """

    length_mm: float
    width_mm: float
    thickness_mm: float
    hole_diameter_mm: float
    hole_edge_margin_mm: float
    fillet_radius_mm: float
    hole_count: int = 4  # currently supports 4 (one per corner) or 2 (along the long axis)
    material: str = "aluminum"

    def hole_centers(self) -> list[tuple[float, float]]:
        hx = self.length_mm / 2 - self.hole_edge_margin_mm
        hy = self.width_mm / 2 - self.hole_edge_margin_mm
        if self.hole_count == 4:
            return [(sx * hx, sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
        if self.hole_count == 2:
            return [(-hx, 0.0), (hx, 0.0)]
        raise ValueError(f"hole_count={self.hole_count} not supported (use 2 or 4)")

    def to_description(self) -> str:
        holes = self.hole_centers()
        # hole spacing along each axis, for the human-readable description
        xs = sorted({round(x, 3) for x, _ in holes})
        x_spacing = xs[-1] - xs[0] if len(xs) > 1 else 0.0
        fillet_clause = (
            f", with {self.fillet_radius_mm:g} mm corner fillets" if self.fillet_radius_mm > 0 else ""
        )
        return (
            f"rectangular {self.material} mounting plate, "
            f"{self.length_mm:g} mm x {self.width_mm:g} mm x {self.thickness_mm:g} mm, "
            f"with {self.hole_count} {self.hole_diameter_mm:g} mm mounting holes, "
            f"{self.hole_edge_margin_mm:g} mm from each edge"
            + (f", {x_spacing:g} mm apart" if x_spacing else "")
            + fillet_clause
        )


def _rounded_rect_polygon(length: float, width: float, fillet_radius: float, segments_per_corner: int = 8):
    """Build a 2D rounded-rectangle outline as an (N, 2) point array, CCW."""
    hl, hw = length / 2, width / 2
    r = max(0.0, min(fillet_radius, hl - 1e-6, hw - 1e-6))
    if r <= 1e-9:
        return np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])

    corners = [
        (hl - r, hw - r, 0),  # top-right, arc center, start angle 0
        (-hl + r, hw - r, 90),  # top-left
        (-hl + r, -hw + r, 180),  # bottom-left
        (hl - r, -hw + r, 270),  # bottom-right
    ]
    pts = []
    for cx, cy, start_deg in corners:
        for i in range(segments_per_corner + 1):
            theta = np.deg2rad(start_deg + 90 * i / segments_per_corner)
            pts.append((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    return np.array(pts)


def build_mounting_plate(params: MountingPlateParams) -> trimesh.Trimesh:
    """Extrude a rounded-rectangle plate and subtract cylindrical mounting holes."""
    outline = _rounded_rect_polygon(params.length_mm, params.width_mm, params.fillet_radius_mm)
    from shapely.geometry import Polygon

    # engine="triangle" (Shewchuk's Triangle, via the `triangle` package) with a
    # quality+max-area constraint, not the default mapbox_earcut -- ear-clipping
    # a plain rectangle produces a handful of long, thin fan triangles spanning
    # the whole face (visually and topologically poor, even though watertight).
    # The area cap scales with the plate's own footprint so triangle DENSITY
    # stays comparable across the sampler's size range instead of a fixed
    # constant over- or under-tessellating small/large plates.
    target_triangle_area = max(10.0, (params.length_mm * params.width_mm) / 1000)
    plate = trimesh.creation.extrude_polygon(
        Polygon(outline), height=params.thickness_mm,
        engine="triangle", triangle_args=f"pq30a{target_triangle_area:.1f}",
    )
    plate.apply_translation([0, 0, -params.thickness_mm / 2])

    for cx, cy in params.hole_centers():
        hole = trimesh.creation.cylinder(
            radius=params.hole_diameter_mm / 2,
            height=params.thickness_mm * 3,  # guarantee it fully punches through
            sections=32,
        )
        hole.apply_translation([cx, cy, 0])
        plate = plate.difference(hole, engine="manifold")

    plate = remesh_flat_facets(plate)
    plate.metadata["params"] = asdict(params)
    return plate


def random_mounting_plate(rng: random.Random) -> MountingPlateParams:
    """Sample a mechanically-plausible plate (bounds keep holes from overlapping/edges)."""
    length = rng.choice([80, 100, 120, 150, 200])
    width = rng.choice([60, 80, 100, 120])
    thickness = rng.choice([4, 5, 6, 8, 10])
    hole_count = rng.choice([2, 4])
    margin = rng.choice([15, 20, 25])
    max_hole_d = 2 * min(margin, length / 2 - margin, width / 2 - margin) * 0.6
    hole_d = min(rng.choice([6, 8, 10, 12]), max(3, max_hole_d))
    fillet = rng.choice([0, 3, 5, 8])
    return MountingPlateParams(
        length_mm=length,
        width_mm=width,
        thickness_mm=thickness,
        hole_diameter_mm=hole_d,
        hole_edge_margin_mm=margin,
        fillet_radius_mm=fillet,
        hole_count=hole_count,
        # matches every material nlp/prompt_parser.py's regex recognizes, so a
        # model trained on this sampler's output can learn the full label space
        material=rng.choice(["aluminum", "steel", "polycarbonate", "titanium", "plastic", "brass"]),
    )


@dataclass
class FlangeParams:
    """Circular flange: a disk with a central through-bore and N bolt holes evenly
    spaced on a bolt circle. Covers the brief's flange example: central bore,
    planar mounting surface, bolt holes, rotational symmetry, consistent thickness.
    """

    outer_diameter_mm: float
    bore_diameter_mm: float
    thickness_mm: float
    bolt_hole_count: int
    bolt_hole_diameter_mm: float
    bolt_circle_diameter_mm: float
    material: str = "steel"

    def bolt_hole_centers(self) -> list[tuple[float, float]]:
        r = self.bolt_circle_diameter_mm / 2
        return [
            (r * np.cos(theta), r * np.sin(theta))
            for theta in np.linspace(0, 2 * np.pi, self.bolt_hole_count, endpoint=False)
        ]

    def to_description(self) -> str:
        return (
            f"circular {self.material} flange, {self.outer_diameter_mm:g} mm outer diameter, "
            f"{self.thickness_mm:g} mm thick, with a {self.bore_diameter_mm:g} mm central bore "
            f"and {self.bolt_hole_count} {self.bolt_hole_diameter_mm:g} mm bolt holes "
            f"on a {self.bolt_circle_diameter_mm:g} mm bolt circle"
        )


def build_flange(params: FlangeParams) -> trimesh.Trimesh:
    disk = trimesh.creation.cylinder(radius=params.outer_diameter_mm / 2, height=params.thickness_mm, sections=64)
    bore = trimesh.creation.cylinder(radius=params.bore_diameter_mm / 2, height=params.thickness_mm * 3, sections=64)
    disk = disk.difference(bore, engine="manifold")

    for cx, cy in params.bolt_hole_centers():
        hole = trimesh.creation.cylinder(
            radius=params.bolt_hole_diameter_mm / 2, height=params.thickness_mm * 3, sections=24
        )
        hole.apply_translation([cx, cy, 0])
        disk = disk.difference(hole, engine="manifold")

    disk = remesh_flat_facets(disk)
    disk.metadata["params"] = asdict(params)
    return disk


def random_flange(rng: random.Random) -> FlangeParams:
    outer = rng.choice([80, 100, 120, 150])
    thickness = rng.choice([6, 8, 10, 12])
    bore = rng.choice([20, 25, 30, 40])
    bcd = rng.choice([v for v in [50, 60, 70, 90, 110] if bore + 15 < v < outer - 15])
    bolt_d = rng.choice([6, 8, 10])
    bolt_count = rng.choice([4, 6, 8])
    return FlangeParams(
        outer_diameter_mm=outer,
        bore_diameter_mm=bore,
        thickness_mm=thickness,
        bolt_hole_count=bolt_count,
        bolt_hole_diameter_mm=bolt_d,
        bolt_circle_diameter_mm=bcd,
        # matches every material nlp/prompt_parser.py's regex recognizes
        material=rng.choice(["steel", "aluminum", "cast iron", "titanium", "brass"]),
    )


@dataclass
class BracketParams:
    """L-shaped angle bracket: two perpendicular rectangular arms meeting at a
    shared edge, each arm with its own row of mounting holes.

    Formerly a known evaluation gap (now fixed, see evaluation/geometry_eval.py's
    _detect_holes_and_cylinders): this bracket's "length" runs along Z, so
    holes drilled through each arm's own thickness are horizontal-axis
    (arm 1: Y-axis, arm 2: X-axis) -- neither arm's holes are Z-axis. The
    detector used to only test the Z axis, so detected_hole_count read 0 for
    any bracket regardless of how correctly it was drilled (confirmed via
    volume-removal and position tests). It now tries all 3 world axes, so
    both arms' holes are detected correctly, each tagged with its own real
    axis in GeometryReport.hole_axes.
    """

    leg1_length_mm: float
    leg2_length_mm: float
    thickness_mm: float
    length_mm: float  # extrusion length (the bracket's third dimension)
    hole_diameter_mm: float
    hole_edge_margin_mm: float
    holes_per_arm: int = 2
    material: str = "steel"

    def arm1_hole_positions(self) -> list[float]:
        """Positions along the extrusion (length) axis for arm 1's holes,
        evenly spaced -- in [0, length_mm], recentered by the builder."""
        return _evenly_spaced_positions(self.length_mm, self.hole_edge_margin_mm, self.holes_per_arm)

    def arm2_hole_positions(self) -> list[float]:
        return _evenly_spaced_positions(self.length_mm, self.hole_edge_margin_mm, self.holes_per_arm)

    def to_description(self) -> str:
        return (
            f"L-shaped {self.material} bracket, {self.leg1_length_mm:g} mm x {self.leg2_length_mm:g} mm legs, "
            f"{self.thickness_mm:g} mm thick, {self.length_mm:g} mm long, "
            f"with {self.holes_per_arm} {self.hole_diameter_mm:g} mm holes per arm"
        )


def _evenly_spaced_positions(length: float, margin: float, count: int) -> list[float]:
    """count positions along [0, length], inset by margin from each end,
    centered on the segment midpoint (falls back to the midpoint alone if
    count <= 1 or the margin leaves no room)."""
    usable = max(length - 2 * margin, 0.0)
    if count <= 1 or usable <= 0:
        return [length / 2]
    step = usable / (count - 1)
    return [margin + i * step for i in range(count)]


def build_bracket(params: BracketParams) -> trimesh.Trimesh:
    """Extrude an L-shaped cross-section along Z (trimesh's natural extrusion
    axis -- "length" is Z here, no whole-solid rotation needed), then drill
    each arm's holes through its own thickness. Arm 1 (spans X, thin in Y)
    needs its hole cylinder rotated from Z-axis to Y-axis; arm 2 (spans Y,
    thin in X) needs it rotated to X-axis. Both rotations happen on the
    cylinder BEFORE translating -- rotating an already-translated cylinder
    would swing it around the wrong pivot."""
    from shapely.geometry import Polygon

    t = params.thickness_mm
    l1, l2 = params.leg1_length_mm, params.leg2_length_mm
    outline = np.array([[0, 0], [l1, 0], [l1, t], [t, t], [t, l2], [0, l2]])

    target_triangle_area = max(4.0, (l1 * l2) / 400)
    bracket = trimesh.creation.extrude_polygon(
        Polygon(outline), height=params.length_mm,
        engine="triangle", triangle_args=f"pq30a{target_triangle_area:.1f}",
    )
    bracket.apply_translation([0, 0, -params.length_mm / 2])  # center along the length axis, like the plate

    r = params.hole_diameter_mm / 2
    for z in params.arm1_hole_positions():
        z_centered = z - params.length_mm / 2
        hole = trimesh.creation.cylinder(radius=r, height=t * 3, sections=32)
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # Z-axis -> Y-axis
        hole.apply_translation([l1 / 2, t / 2, z_centered])
        bracket = bracket.difference(hole, engine="manifold")
    for z in params.arm2_hole_positions():
        z_centered = z - params.length_mm / 2
        hole = trimesh.creation.cylinder(radius=r, height=t * 3, sections=32)
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))  # Z-axis -> X-axis
        hole.apply_translation([t / 2, l2 / 2, z_centered])
        bracket = bracket.difference(hole, engine="manifold")

    bracket = remesh_flat_facets(bracket)
    bracket.metadata["params"] = asdict(params)
    return bracket


def random_bracket(rng: random.Random) -> BracketParams:
    leg1 = rng.choice([40, 50, 60, 80])
    leg2 = rng.choice([40, 50, 60, 80])
    thickness = rng.choice([3, 4, 5, 6])
    length = rng.choice([50, 80, 100, 150])
    holes_per_arm = rng.choice([1, 2])
    margin = rng.choice([10, 15])
    max_hole_d = 2 * min(margin, thickness * 2) * 0.6
    hole_d = min(rng.choice([4, 5, 6, 8]), max(2.0, max_hole_d))
    return BracketParams(
        leg1_length_mm=leg1, leg2_length_mm=leg2, thickness_mm=thickness, length_mm=length,
        hole_diameter_mm=hole_d, hole_edge_margin_mm=margin, holes_per_arm=holes_per_arm,
        material=rng.choice(["steel", "aluminum", "cast iron", "titanium", "brass"]),
    )


@dataclass
class ShaftParams:
    """Solid or hollow cylindrical shaft, with an optional keyway slot cut
    along part (or all) of its length."""

    outer_diameter_mm: float
    length_mm: float
    inner_diameter_mm: float = 0.0  # 0 = solid
    keyway_width_mm: float = 0.0  # 0 = no keyway
    keyway_depth_mm: float = 0.0
    keyway_length_mm: float = 0.0  # 0 = full length (consistent with the other keyway_* 0-means-absent fields)
    material: str = "steel"

    def to_description(self) -> str:
        hollow = f", {self.inner_diameter_mm:g} mm bore" if self.inner_diameter_mm > 0 else ""
        keyway = f", with a {self.keyway_width_mm:g} mm keyway" if self.keyway_width_mm > 0 else ""
        return f"{self.material} shaft, {self.outer_diameter_mm:g} mm diameter, {self.length_mm:g} mm long{hollow}{keyway}"


def build_shaft(params: ShaftParams) -> trimesh.Trimesh:
    shaft = trimesh.creation.cylinder(radius=params.outer_diameter_mm / 2, height=params.length_mm, sections=48)

    if params.inner_diameter_mm > 0:
        # additive margin, not the plate/flange's *3 convention -- shafts can be
        # long, so a multiplicative margin would be wastefully oversized
        bore = trimesh.creation.cylinder(radius=params.inner_diameter_mm / 2, height=params.length_mm + 10, sections=48)
        shaft = shaft.difference(bore, engine="manifold")

    if params.keyway_width_mm > 0:
        keyway_length = params.keyway_length_mm or params.length_mm
        od = params.outer_diameter_mm
        # a box face exactly tangent to the cylinder surface is a classic
        # boolean-robustness failure -- overstick past the OD by ~1mm
        keyway = trimesh.creation.box(extents=[params.keyway_depth_mm + 1.0, params.keyway_width_mm, keyway_length])
        keyway.apply_translation([od / 2 - params.keyway_depth_mm / 2 + 0.5, 0, 0])
        shaft = shaft.difference(keyway, engine="manifold")

    shaft = remesh_flat_facets(shaft)
    shaft.metadata["params"] = asdict(params)
    return shaft


def random_shaft(rng: random.Random) -> ShaftParams:
    od = rng.choice([10, 16, 20, 25, 32])
    length = rng.choice([50, 100, 150, 200, 300])
    hollow = rng.choice([True, False])
    inner = rng.choice([od * 0.3, od * 0.4]) if hollow else 0.0
    has_keyway = rng.choice([True, False])
    max_depth = (od - inner) / 2 * 0.7 if hollow else od * 0.15
    keyway_width = rng.choice([3, 4, 5]) if has_keyway else 0.0
    keyway_depth = min(rng.choice([1.5, 2, 3]), max_depth) if has_keyway else 0.0
    return ShaftParams(
        outer_diameter_mm=od, length_mm=length, inner_diameter_mm=round(inner, 1),
        keyway_width_mm=keyway_width, keyway_depth_mm=round(keyway_depth, 1),
        material=rng.choice(["steel", "aluminum", "titanium", "brass"]),
    )


@dataclass
class EnclosureParams:
    """Hollow rectangular box, optionally open on top, with optional
    ventilation holes drilled through the top wall.

    v1 scope: only open_face in {"none", "top"} -- other faces are a
    straightforward but unbuilt extension. No edge rounding -- true 3D edge
    fillets need real fillet CSG, materially more work than everything else
    in this module.
    """

    length_mm: float
    width_mm: float
    height_mm: float
    wall_thickness_mm: float
    open_face: str = "none"  # "none" | "top"
    vent_hole_count: int = 0
    vent_hole_diameter_mm: float = 0.0
    material: str = "aluminum"

    def to_description(self) -> str:
        open_clause = ", open top" if self.open_face == "top" else ""
        vent_clause = f", {self.vent_hole_count} {self.vent_hole_diameter_mm:g} mm vent holes" if self.vent_hole_count else ""
        return (
            f"{self.material} enclosure, {self.length_mm:g} mm x {self.width_mm:g} mm x {self.height_mm:g} mm, "
            f"{self.wall_thickness_mm:g} mm walls{open_clause}{vent_clause}"
        )


def build_enclosure(params: EnclosureParams) -> trimesh.Trimesh:
    L, W, H, wt = params.length_mm, params.width_mm, params.height_mm, params.wall_thickness_mm
    box = trimesh.creation.box(extents=[L, W, H])

    cavity_top = H / 2 + 1.0 if params.open_face == "top" else H / 2 - wt
    cavity_height = cavity_top - (-H / 2 + wt)
    cavity = trimesh.creation.box(extents=[L - 2 * wt, W - 2 * wt, cavity_height])
    cavity.apply_translation([0, 0, (cavity_top + (-H / 2 + wt)) / 2])
    box = box.difference(cavity, engine="manifold")

    # vent_hole_count and vent_hole_diameter_mm are independently omittable
    # fields (see data/prompt_dataset.py) -- "N holes of 0mm diameter" is a
    # real, reachable combination (a zero-radius drill cutter isn't a valid
    # volume), not hypothetical, so both must be checked, not just count
    if params.vent_hole_count > 0 and params.vent_hole_diameter_mm > 0 and params.open_face != "top":
        r = params.vent_hole_diameter_mm / 2
        positions = _evenly_spaced_positions(L - 2 * wt, r * 1.5, params.vent_hole_count)
        for x_offset in positions:
            x = x_offset - (L - 2 * wt) / 2
            # additive margin centered WITHIN the top wall's own thickness --
            # the plate/flange's *3 convention would punch through the
            # opposite wall on a shallow enclosure
            hole = trimesh.creation.cylinder(radius=r, height=wt + 2.0, sections=24)
            hole.apply_translation([x, 0, H / 2 - wt / 2])
            box = box.difference(hole, engine="manifold")

    box = remesh_flat_facets(box)
    box.metadata["params"] = asdict(params)
    return box


def random_enclosure(rng: random.Random) -> EnclosureParams:
    length = rng.choice([80, 100, 120, 150])
    width = rng.choice([60, 80, 100])
    height = rng.choice([30, 40, 50, 60])
    wall = rng.choice([2, 3, 4])
    open_face = rng.choice(["none", "top"])
    has_vents = rng.choice([True, False]) and open_face == "none"
    vent_count = rng.choice([2, 4, 6]) if has_vents else 0
    vent_d = rng.choice([3, 4, 5]) if has_vents else 0.0
    return EnclosureParams(
        length_mm=length, width_mm=width, height_mm=height, wall_thickness_mm=wall,
        open_face=open_face, vent_hole_count=vent_count, vent_hole_diameter_mm=vent_d,
        material=rng.choice(["aluminum", "steel", "plastic", "polycarbonate"]),
    )


@dataclass
class GearParams:
    """Spur gear with a real (if simplified) 20-degree involute tooth
    profile -- not a fake trapezoidal approximation.

    v1 scope: tooth_count is floored at 12 by the sampler. Below that,
    standard involute teeth undercut (the dedendum flank curves back on
    itself near the root), which can make the tooth polygon self-intersect.
    That's a well-known real property of involute gearing, not a bug to
    chase here.
    """

    module_mm: float
    tooth_count: int
    face_width_mm: float
    bore_diameter_mm: float
    pressure_angle_deg: float = 20.0
    material: str = "steel"

    def pitch_radius(self) -> float:
        return self.module_mm * self.tooth_count / 2

    def base_radius(self) -> float:
        return self.pitch_radius() * np.cos(np.radians(self.pressure_angle_deg))

    def addendum_radius(self) -> float:
        return self.pitch_radius() + self.module_mm

    def dedendum_radius(self) -> float:
        return self.pitch_radius() - 1.25 * self.module_mm

    def to_description(self) -> str:
        return (
            f"{self.material} spur gear, module {self.module_mm:g}, {self.tooth_count} teeth, "
            f"{self.face_width_mm:g} mm face width, {self.bore_diameter_mm:g} mm bore"
        )


def _involute_flank(r_b: float, r_a: float, n_points: int = 18) -> np.ndarray:
    """Points along the involute curve from the base circle to the addendum
    circle. Standard parametric form: x = r_b(cos t + t sin t), y = r_b(sin
    t - t cos t), t in [0, t_max] where t_max solves r(t_max) = r_a."""
    t_max = np.sqrt(max((r_a / r_b) ** 2 - 1, 0.0))
    t = np.linspace(0, t_max, n_points)
    x = r_b * (np.cos(t) + t * np.sin(t))
    y = r_b * (np.sin(t) - t * np.cos(t))
    return np.column_stack([x, y])


def _one_tooth_outline(params: GearParams) -> np.ndarray:
    r_b = params.base_radius()
    r_a = params.addendum_radius()
    r_d = params.dedendum_radius()
    pressure_angle = np.radians(params.pressure_angle_deg)

    flank = _involute_flank(r_b, r_a)
    # center the tooth: rotate the raw flank so its pitch-circle point lands
    # at angle -half_angle, using the involute-function identity
    # inv(a) = tan(a) - a
    inv_pa = np.tan(pressure_angle) - pressure_angle
    half_angle = np.pi / (2 * params.tooth_count) + inv_pa
    rot = -(half_angle)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    right_flank = flank @ R.T

    # if the root circle is inside the base circle (common for larger
    # modules), connect the flank's base-circle start down to the root
    # circle with a straight radial segment -- the standard simplification
    if r_d < r_b:
        start_angle = np.arctan2(right_flank[0, 1], right_flank[0, 0])
        root_point = np.array([[r_d * np.cos(start_angle), r_d * np.sin(start_angle)]])
        right_flank = np.vstack([root_point, right_flank])

    left_flank = right_flank.copy()
    left_flank[:, 1] *= -1
    left_flank = left_flank[::-1]

    return np.vstack([right_flank, left_flank])


def build_gear(params: GearParams) -> trimesh.Trimesh:
    from shapely.geometry import Polygon

    tooth = _one_tooth_outline(params)
    n = params.tooth_count
    loop = []
    for k in range(n):
        angle = k * 2 * np.pi / n
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        loop.append(tooth @ R.T)
    outline = np.vstack(loop)

    polygon = Polygon(outline)
    if not polygon.is_valid:
        # low tooth counts / large modules can self-intersect (undercutting);
        # buffer(0) repairs minor self-intersections without changing the
        # visible shape materially
        polygon = polygon.buffer(0)

    # feature size here is the tooth (~module_mm), not the pitch diameter --
    # reusing the plate's bbox-area heuristic would lose tooth shape on a
    # large-diameter, fine-module gear
    target_triangle_area = max(0.3, (params.module_mm ** 2) / 8)
    gear = trimesh.creation.extrude_polygon(
        polygon, height=params.face_width_mm,
        engine="triangle", triangle_args=f"pq30a{target_triangle_area:.2f}",
    )
    gear.apply_translation([0, 0, -params.face_width_mm / 2])

    if params.bore_diameter_mm > 0:
        bore = trimesh.creation.cylinder(radius=params.bore_diameter_mm / 2, height=params.face_width_mm * 3, sections=48)
        gear = gear.difference(bore, engine="manifold")

    gear = remesh_flat_facets(gear)
    gear.metadata["params"] = asdict(params)
    return gear


def random_gear(rng: random.Random) -> GearParams:
    module = rng.choice([1.5, 2, 2.5, 3])
    tooth_count = rng.choice([12, 16, 20, 24, 32])  # floored at 12 -- see class docstring
    face_width = rng.choice([8, 10, 15, 20])
    pitch_d = module * tooth_count
    bore = min(rng.choice([6, 8, 10, 12]), pitch_d * 0.3)
    return GearParams(
        module_mm=module, tooth_count=tooth_count, face_width_mm=face_width,
        bore_diameter_mm=round(bore, 1),
        material=rng.choice(["steel", "aluminum", "brass", "titanium"]),
    )


def generate_dataset(
    out_dir: Path,
    n: int,
    seed: int = 0,
    *,
    sampler=random_mounting_plate,
    builder=build_mounting_plate,
    id_prefix: str = "mounting_plate",
) -> list[dict]:
    """Generate n (mesh, description, params) triples under out_dir. Returns the manifest.

    Defaults to mounting plates only, for backward compatibility with existing
    callers -- pass sampler=random_flange, builder=build_flange, id_prefix="flange"
    (or use part_families.py's registry to loop over every registered family;
    see scripts/generate_dataset.py) to generate other part types. Kept generic
    here rather than importing part_families.py, which itself imports FROM this
    module -- that would be circular.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = []
    for i in range(n):
        params = sampler(rng)
        mesh = builder(params)
        stem = f"{id_prefix}_{i:04d}"
        mesh_path = out_dir / f"{stem}.stl"
        mesh.export(mesh_path)
        record = {
            "id": stem,
            "mesh_path": str(mesh_path.relative_to(out_dir)),
            "description": params.to_description(),
            "params": asdict(params),
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2))
        manifest.append(record)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
