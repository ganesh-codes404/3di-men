"""Stage D: slice-to-solid construction. Connects the PLANNED slices'
circular profiles into a single watertight solid by ring-lofting --
generalizing sandbox.py's add_loft technique (smoothstep-blend between
exactly two profiles) to N profiles, one per planned slice, each
profile already resampled to the SAME point count so consecutive rings
connect 1:1. This is a fresh implementation in Mitsugen's own module,
not a call into sandbox.py's Sandbox -- see this package's own
__init__.py for why that boundary matters.

Does NOT equate "slice-by-slice" with stacking unrelated flat slabs:
every pair of consecutive rings is connected by real triangulated side
faces, and the whole stack is one continuous watertight shell, not N
disconnected disks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from mitsugen.profile import Profile2D, make_circular_profile
from mitsugen.slice_planner import SlicePlan


@dataclass
class BuildResult:
    mesh: "trimesh.Trimesh | None"
    profiles: list[Profile2D]
    hollow_applied: bool
    error: str | None = None  # None on success -- never a partial/guessed mesh on failure


def _ring_loft(z_positions: list[float], radii: list[float], sections: int) -> trimesh.Trimesh:
    """The actual N-ring lofting primitive: one circular profile per
    (z, radius) pair, connected into a continuous tube and capped at
    both ends. Requires len(z_positions) >= 2."""
    n_rings = len(z_positions)
    angles = np.linspace(0.0, 2 * np.pi, sections, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    chunks = []
    for z, r in zip(z_positions, radii):
        r = max(r, 1e-3)  # a true-zero radius ring degenerates triangulation -- clamp to a tiny point-like ring
        chunks.append(np.column_stack([r * cos_a, r * sin_a, np.full(sections, z)]))
    vertices_side = np.vstack(chunks)

    faces = []
    for k in range(n_rings - 1):
        r0, r1 = k * sections, (k + 1) * sections
        for i in range(sections):
            j = (i + 1) % sections
            faces.append([r0 + i, r0 + j, r1 + j])
            faces.append([r0 + i, r1 + j, r1 + i])

    from shapely.geometry import Polygon
    from trimesh.creation import triangulate_polygon

    bottom_2d = vertices_side[0:sections, :2]
    top_2d = vertices_side[(n_rings - 1) * sections : n_rings * sections, :2]
    bottom_cap_v, bottom_cap_f = triangulate_polygon(Polygon(bottom_2d), engine="triangle", force_vertices=True)
    top_cap_v, top_cap_f = triangulate_polygon(Polygon(top_2d), engine="triangle", force_vertices=True)

    next_idx = n_rings * sections
    bottom_cap_idx = next_idx
    bottom_cap_3d = np.column_stack([bottom_cap_v[:, 0], bottom_cap_v[:, 1], np.full(len(bottom_cap_v), z_positions[0])])
    next_idx += len(bottom_cap_v)
    top_cap_idx = next_idx
    top_cap_3d = np.column_stack([top_cap_v[:, 0], top_cap_v[:, 1], np.full(len(top_cap_v), z_positions[-1])])

    vertices = np.vstack([vertices_side, bottom_cap_3d, top_cap_3d])
    for f in bottom_cap_f:
        faces.append([bottom_cap_idx + f[0], bottom_cap_idx + f[2], bottom_cap_idx + f[1]])  # reversed: outward normal is -z
    for f in top_cap_f:
        faces.append([top_cap_idx + f[0], top_cap_idx + f[1], top_cap_idx + f[2]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def build_solid_from_slices(plan: SlicePlan, sections: int = 32) -> BuildResult:
    """The main entry point: builds a real, watertight solid from
    `plan`'s own planned slices (NOT a stack of disconnected slabs).
    Never raises -- any construction failure comes back as
    BuildResult(mesh=None, error=...), the same typed-failure
    discipline 3di-men's own sandbox.py uses, not an exception a caller
    has to guard against separately."""
    if len(plan.slices) < 2:
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error="a solid needs at least 2 slices")

    profiles = [make_circular_profile(s.slice_id, s.radius_mm, sections) for s in plan.slices]
    z_positions = [s.position_mm for s in plan.slices]
    radii = [s.radius_mm for s in plan.slices]

    try:
        mesh = _ring_loft(z_positions, radii, sections)
    except Exception as exc:
        return BuildResult(mesh=None, profiles=profiles, hollow_applied=False, error=f"ring lofting failed: {exc}")

    if not mesh.is_watertight:
        return BuildResult(mesh=None, profiles=profiles, hollow_applied=False, error="constructed solid is not watertight -- rejected rather than exported as-is")

    return BuildResult(mesh=mesh, profiles=profiles, hollow_applied=False, error=None)


def apply_hollowing(build: BuildResult, plan: SlicePlan, wall_thickness_mm: float, sections: int = 32) -> BuildResult:
    """Hollows an already-built solid by subtracting an inner cavity
    shaped the same way (radius - wall_thickness at every slice,
    clamped to a minimum positive radius), left SHORTER than the outer
    shell at the bottom (preserving a floor) and open at the top --
    the actual shape every real open-topped vessel (a vase, a cup, a
    bowl) needs, not a fully sealed hollow shell. Returns the original
    `build` unchanged (not a silent no-op disguised as success) if the
    cavity can't be safely subtracted -- e.g. the wall thickness leaves
    no positive inner radius anywhere."""
    if build.mesh is None:
        return build
    inner_radii = [max(r - wall_thickness_mm, 0.0) for r in (s.radius_mm for s in plan.slices)]
    if all(r <= 1e-6 for r in inner_radii):
        return BuildResult(
            mesh=build.mesh, profiles=build.profiles, hollow_applied=False,
            error=f"wall_thickness_mm={wall_thickness_mm:g} leaves no positive inner cavity anywhere -- hollowing skipped",
        )

    z_positions = [s.position_mm for s in plan.slices]
    floor_mm = max(wall_thickness_mm, 1.0)
    inner_z = [z_positions[0] + floor_mm] + z_positions[1:-1] + [z_positions[-1] + floor_mm]
    # the cavity's own top ring extends past the outer shell's top on
    # purpose -- guarantees an open top after the boolean difference,
    # rather than leaving a thin sealed cap that watertightness would
    # otherwise quietly accept as "fine"
    inner_z[-1] = z_positions[-1] + floor_mm

    try:
        cavity = _ring_loft(inner_z, inner_radii, sections)
        hollowed = build.mesh.difference(cavity, engine="manifold")
    except Exception as exc:
        return BuildResult(mesh=build.mesh, profiles=build.profiles, hollow_applied=False, error=f"hollowing failed: {exc}")

    if not hollowed.is_watertight or hollowed.volume <= 0:
        return BuildResult(mesh=build.mesh, profiles=build.profiles, hollow_applied=False, error="hollowed result was invalid -- kept the solid version instead")

    return BuildResult(mesh=hollowed, profiles=build.profiles, hollow_applied=True, error=None)
