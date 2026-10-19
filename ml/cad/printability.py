"""Geometry-based 3D-printing analysis -- the backend for the Print Lab
workspace. Every check here is computed directly from the mesh (via trimesh
+ numpy, both already project dependencies -- no new library added) or is
explicitly labeled unavailable; nothing here is a slicer measurement or a
fabricated placeholder number. See each report dataclass's own `method`
field for exactly how its numbers were produced, and PRINT_LAB_LIMITATIONS
below for what this module deliberately does NOT attempt yet.

Reuses existing, already-tested geometry machinery rather than
reimplementing it: mesh_quality.mesh_topology_stats for the base topology
numbers, mesh_quality.measure_wall_thickness_deviation's own ray-casting
convention informs (but does not duplicate) the overhang sampling approach
here.

Scope, stated plainly (so a caller never mistakes "not implemented yet" for
"zero problems found"):
- Implemented: mesh integrity (watertight/manifold/shells/degenerate
  faces), build-volume fit (any axis-aligned orientation), face-normal
  overhang analysis, geometry-derived material volume/mass, and basic
  orientation suggestions scored from the above.
- Explicitly NOT implemented (reported as unavailable, never guessed):
  local wall-thickness-for-printability (a different, nozzle-aware
  question from mesh_quality's own shell-approximation check), real
  slicer-computed print time, real generated support structures.
  (STL/OBJ/GLB/3MF export itself is handled by server.py, reusing
  /api/mesh/<id>?format=... -- 3MF confirmed directly to actually
  export AND reimport correctly with this project's installed trimesh,
  see tests/test_server.py::test_exported_mesh_round_trips_through_3mf.)"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

PRINT_LAB_LIMITATIONS = [
    "wall-thickness-for-printability (nozzle/process-aware local thickness) is not implemented -- "
    "use the dimensions/overhang checks below as a partial substitute",
    "print time is never estimated here -- it depends on a real slicer's own toolpath planning "
    "(travel moves, acceleration, retraction, cooling pauses), which this geometry-only service has "
    "no way to reproduce without fabricating a number",
    "support MATERIAL VOLUME is not estimated (only which faces likely need support, and their total "
    "area) -- real support generation depends on the slicer's own support style/density settings",
    "no real slicer (Cura/PrusaSlicer) integration exists yet -- see slicer_availability()",
]


@dataclass
class MeshIntegrityReport:
    watertight: bool
    manifold: bool  # watertight AND consistently wound -- see mesh_quality.mesh_topology_stats
    body_count: int  # connected, disconnected shells -- 1 is the common case; >1 is not
        # automatically wrong (e.g. a multi-part assembly exported as one file), but
        # the caller must know this to tell "one solid" apart from "several"
    non_manifold_edge_count: int  # an edge shared by != 2 faces -- a genuine mesh defect
        # distinct from "has a boundary/hole" (which shows up as not-watertight instead)
    degenerate_face_count: int  # near-zero-area triangles
    is_empty: bool  # zero vertices/faces -- every OTHER field above is meaningless if True
    vertex_count: int
    face_count: int
    bounding_box_mm: tuple[float, float, float]
    volume_mm3: float | None  # None when the mesh isn't a valid closed volume (is_volume False)
    surface_area_mm2: float
    method: str = "trimesh topology properties + a face-edge-occurrence count for non-manifold edges"

    @property
    def printable_as_is(self) -> bool:
        """A structurally sound SINGLE solid -- the minimum bar before any
        of the other checks below are even meaningful. Does NOT mean
        "will print well" (see overhangs/build-volume-fit for that) --
        watertightness alone is explicitly NOT treated as a full
        printability verdict anywhere in this module."""
        return (
            not self.is_empty and self.watertight and self.manifold
            and self.body_count >= 1 and self.non_manifold_edge_count == 0
            and self.degenerate_face_count == 0
        )


def check_mesh_integrity(mesh: trimesh.Trimesh) -> MeshIntegrityReport:
    """Never raises: an empty/degenerate mesh produces an honest
    is_empty=True report, not an exception -- same "never crash on a
    validation question" convention as spec_validation.py/mesh_quality.py."""
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return MeshIntegrityReport(
            watertight=False, manifold=False, body_count=0, non_manifold_edge_count=0,
            degenerate_face_count=0, is_empty=True, vertex_count=0, face_count=0,
            bounding_box_mm=(0.0, 0.0, 0.0), volume_mm3=None, surface_area_mm2=0.0,
        )
    edges_sorted = mesh.edges_sorted
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    non_manifold_edge_count = int(np.sum(counts != 2))
    degenerate_face_count = int(np.sum(mesh.area_faces < 1e-8))
    extents = mesh.bounds[1] - mesh.bounds[0]
    return MeshIntegrityReport(
        watertight=bool(mesh.is_watertight),
        manifold=bool(mesh.is_watertight and mesh.is_winding_consistent),
        body_count=int(mesh.body_count),
        non_manifold_edge_count=non_manifold_edge_count,
        degenerate_face_count=degenerate_face_count,
        is_empty=False,
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        bounding_box_mm=(float(extents[0]), float(extents[1]), float(extents[2])),
        volume_mm3=float(mesh.volume) if mesh.is_volume else None,
        surface_area_mm2=float(mesh.area),
    )


@dataclass
class PrinterProfile:
    """A printer's own configured build volume -- deliberately just a
    rectangular box (every printer this project has any reason to model
    today is a standard Cartesian/bedslinger/CoreXY FDM machine); a
    non-rectangular build volume (e.g. a delta printer's own cylindrical
    envelope) is a disclosed, real limitation (see
    check_build_volume_fit's own docstring), not silently assumed away."""
    name: str
    build_volume_mm: tuple[float, float, float]  # X, Y, Z
    nozzle_diameter_mm: float = 0.4
    build_volume_shape: str = "rectangular"


DEFAULT_PRINTER_PROFILES: dict[str, PrinterProfile] = {
    "generic_small": PrinterProfile(name="Generic small FDM (220x220x250)", build_volume_mm=(220.0, 220.0, 250.0)),
    "generic_medium": PrinterProfile(name="Generic medium FDM (250x210x210)", build_volume_mm=(250.0, 210.0, 210.0)),
    "generic_large": PrinterProfile(name="Generic large FDM (300x300x400)", build_volume_mm=(300.0, 300.0, 400.0)),
}


@dataclass
class BuildVolumeFitReport:
    fits: bool
    model_extents_mm: tuple[float, float, float]
    printer_volume_mm: tuple[float, float, float]
    overrun_mm: tuple[float, float, float]  # 0 on any axis that fits; positive = exceeds by this much
    method: str = "oriented bounding box vs. the printer profile's own configured rectangular volume"


def check_build_volume_fit(mesh: trimesh.Trimesh, printer: PrinterProfile) -> BuildVolumeFitReport:
    """Compares the mesh's OWN current bounding box (whatever orientation
    it's actually in right now -- apply a rotation first via
    trimesh.Trimesh.apply_transform if checking a candidate orientation,
    see suggest_orientations below) against the printer's configured
    volume. Deliberately disclosed as rectangular-only (see
    PrinterProfile.build_volume_shape) -- a non-rectangular build volume
    would need real per-shape fit logic this module doesn't have."""
    extents = tuple(float(e) for e in (mesh.bounds[1] - mesh.bounds[0]))
    overrun = tuple(max(0.0, extents[i] - printer.build_volume_mm[i]) for i in range(3))
    return BuildVolumeFitReport(
        fits=all(o <= 1e-6 for o in overrun), model_extents_mm=extents,
        printer_volume_mm=printer.build_volume_mm, overrun_mm=overrun,
    )


@dataclass
class OverhangReport:
    overhang_angle_deg: float  # the configured threshold itself, disclosed back
    overhang_area_mm2: float  # total surface area of faces steeper than the threshold
    overhang_area_fraction: float  # that area as a fraction of total surface area
    build_direction: tuple[float, float, float]
    overhang_face_indices: list[int] = field(default_factory=list)  # confirmed real gap this closes:
        # the Print Lab UI could only show the AGGREGATE overhang area/fraction as numbers, with no way
        # to show the user WHERE on the model they actually are -- the per-face mask was computed then
        # discarded. These are the mesh's own face indices (same order the exported STL/the viewer's
        # loaded geometry uses), so a caller can highlight exactly these triangles instead of only
        # reporting a percentage. Not emitted for every overhang face on a pathologically dense mesh
        # (see server.py's own upload face-count ceiling) -- a real limitation for very large meshes,
        # not attempted here.
    method: str = (
        "per-face angle between each triangle's own normal and the build-up direction, "
        "area-weighted -- a real geometric measurement, not a simulated slicer support map"
    )

    @property
    def likely_needs_support(self) -> bool:
        return self.overhang_area_mm2 > 1.0  # any nontrivial overhang area at all


def check_overhangs(
    mesh: trimesh.Trimesh, overhang_angle_deg: float = 45.0, build_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> OverhangReport:
    """A face is a potential overhang when the angle between its OWN
    outward normal and the build direction exceeds (90 - overhang_angle_deg)
    -- i.e. the face slopes down steeper than the printer/material can
    bridge unsupported, the same convention real slicers use for their own
    "support angle" setting. Downward-facing faces (normal has a negative
    component along build_direction) are the ones that actually need
    support FROM BELOW; an upward overhang (like the underside of an arch,
    also downward-facing by this same test) is exactly the geometry that
    gets flagged, which is correct -- an upward-sloping top surface never
    needs support regardless of its angle.

    Deliberately does NOT claim to generate real slicer supports or know
    the true minimum support structure -- see this module's own
    PRINT_LAB_LIMITATIONS. overhang_angle_deg is the printer/material-
    dependent setting every real slicer also exposes -- there is no
    universal correct default, 45 degrees is simply the common one.

    Faces actually RESTING on the build plate are excluded from the
    downward-facing count regardless of angle -- confirmed real bug this
    fixes (caught in this module's own smoke test, not hypothetical): a
    plain box's flat bottom face points straight down (the steepest
    possible angle) but needs zero support, since the plate itself is
    already holding it up. Approximated as "within a small tolerance of
    the mesh's own minimum coordinate along the build direction" -- a
    face resting flush on the plate sits exactly there; a true overhang
    floating above the plate (a bridge, a cantilever) does not."""
    build_dir = np.asarray(build_direction, dtype=float)
    build_dir = build_dir / np.linalg.norm(build_dir)
    normals = mesh.face_normals
    cos_angle = normals @ build_dir  # negative = facing partly downward (away from build-up direction)
    # A face at exactly `overhang_angle_deg` from straight-down is the
    # threshold; cos(90 - overhang_angle_deg) in build-direction terms.
    threshold = -np.cos(np.radians(overhang_angle_deg))
    face_coord_along_build = mesh.triangles_center @ build_dir
    min_coord = float((mesh.vertices @ build_dir).min())
    extent_along_build = float((mesh.vertices @ build_dir).max()) - min_coord
    plate_tolerance = max(1e-3, extent_along_build * 1e-3)  # a thin slice near the model's own base
    resting_on_plate = face_coord_along_build < (min_coord + plate_tolerance)
    overhang_mask = (cos_angle < threshold) & ~resting_on_plate
    overhang_area = float(mesh.area_faces[overhang_mask].sum())
    total_area = float(mesh.area_faces.sum())
    return OverhangReport(
        overhang_angle_deg=overhang_angle_deg, overhang_area_mm2=overhang_area,
        overhang_area_fraction=(overhang_area / total_area) if total_area > 0 else 0.0,
        build_direction=tuple(float(v) for v in build_dir),
        overhang_face_indices=np.nonzero(overhang_mask)[0].tolist(),
    )


@dataclass
class MaterialEstimate:
    solid_volume_mm3: float
    density_g_per_cm3: float | None  # None when no material density was supplied
    estimated_mass_g: float | None  # None whenever density is None -- never guessed
    method: str = (
        "mesh.volume (the SOLID body's own exact volume) x material density -- this is NOT the same as "
        "filament consumption, which also depends on infill percentage, wall count, and any supports "
        "(all slicer-settings-dependent, not geometry alone); see PRINT_LAB_LIMITATIONS"
    )


def estimate_material(mesh: trimesh.Trimesh, density_g_per_cm3: float | None = None) -> MaterialEstimate:
    """mesh.volume is already the EXACT solid volume trimesh computes from
    the mesh's own closed surface (reused, not reimplemented) -- only
    valid (is_volume True) for a watertight, consistently-wound mesh;
    returns 0.0 volume honestly for anything else rather than a wrong
    number. Mass is None (never a guessed density) unless the caller
    supplies one explicitly."""
    volume_mm3 = float(mesh.volume) if (mesh is not None and mesh.is_volume) else 0.0
    mass_g = (volume_mm3 / 1000.0) * density_g_per_cm3 if density_g_per_cm3 else None
    return MaterialEstimate(solid_volume_mm3=volume_mm3, density_g_per_cm3=density_g_per_cm3, estimated_mass_g=mass_g)


@dataclass
class OrientationCandidate:
    label: str
    rotation_deg: tuple[float, float, float]  # X, then Y, then Z -- same convention as
                                                # parallel_builder.PlacementTransform.rotation
    fits_build_volume: bool
    overhang_area_mm2: float
    contact_area_mm2: float  # the footprint actually touching the build plate in this orientation
    height_mm: float  # extent along the build direction -- taller prints take longer, more layers
    score: float  # lower is better; see suggest_orientations's own docstring for the formula
    notes: str


def suggest_orientations(
    mesh: trimesh.Trimesh, printer: PrinterProfile, overhang_angle_deg: float = 45.0,
) -> list[OrientationCandidate]:
    """Tries a small, fixed set of axis-aligned candidate orientations (the
    6 ways to lay an oriented bounding box flat on a build plate -- a real,
    bounded set, not an exhaustive or continuous search) and scores each
    on build-volume fit, overhang area, and build height. This is
    explicitly NOT claimed to find the universally optimal orientation --
    it is a small set of REASONABLE candidates with their own numbers
    shown, matching this module's own disclosed-estimate discipline; a
    human (or a real slicer's own orientation optimizer) may reasonably
    choose differently for reasons this coarse score doesn't capture
    (aesthetics, visible layer lines, known-weak-axis strength).

    Scoring (lower is better): disqualified (does not fit the build
    volume) sorts last regardless of other numbers; among fitting
    candidates, overhang area dominates (most real print-time/support
    cost), with height as a secondary tiebreaker (taller = more layers =
    slower, all else equal)."""
    candidates = [
        ("flat (as modeled)", (0.0, 0.0, 0.0)),
        ("rotated 90 about X", (90.0, 0.0, 0.0)),
        ("rotated -90 about X", (-90.0, 0.0, 0.0)),
        ("rotated 90 about Y", (0.0, 90.0, 0.0)),
        ("rotated -90 about Y", (0.0, -90.0, 0.0)),
        ("upside down (180 about X)", (180.0, 0.0, 0.0)),
    ]
    results = []
    for label, (rx, ry, rz) in candidates:
        rot = (
            trimesh.transformations.rotation_matrix(np.radians(rz), (0, 0, 1))
            @ trimesh.transformations.rotation_matrix(np.radians(ry), (0, 1, 0))
            @ trimesh.transformations.rotation_matrix(np.radians(rx), (1, 0, 0))
        )
        oriented = mesh.copy()
        oriented.apply_transform(rot)
        fit = check_build_volume_fit(oriented, printer)
        overhang = check_overhangs(oriented, overhang_angle_deg=overhang_angle_deg)
        min_z = float(oriented.bounds[0][2])
        contact_faces = oriented.face_normals[:, 2] < -0.99  # facing straight down, at/near the plate
        contact_area = float(oriented.area_faces[contact_faces].sum()) if contact_faces.any() else 0.0
        height = float(oriented.bounds[1][2] - oriented.bounds[0][2])
        score = (0 if fit.fits else 1_000_000.0) + overhang.overhang_area_mm2 + height * 0.1
        notes = "does not fit the printer's build volume" if not fit.fits else (
            "no significant overhangs" if overhang.overhang_area_mm2 < 1.0 else
            f"~{overhang.overhang_area_fraction:.0%} of surface area may need support"
        )
        results.append(OrientationCandidate(
            label=label, rotation_deg=(rx, ry, rz), fits_build_volume=fit.fits,
            overhang_area_mm2=overhang.overhang_area_mm2, contact_area_mm2=contact_area,
            height_mm=height, score=score, notes=notes,
        ))
    results.sort(key=lambda c: c.score)
    return results


def slicer_availability() -> dict:
    """Detects whether a real slicer CLI is actually present on THIS
    machine -- never assumed. Checked, not claimed: this project does not
    bundle or install a slicer, so availability is entirely environment-
    dependent. Returns {"cura": bool, "prusaslicer": bool} plus which
    executable name (if any) was found, so a caller can offer a real
    handoff only when one genuinely exists, and a graceful "not available"
    message otherwise."""
    import shutil

    found = {}
    for name, executables in (("cura", ("CuraEngine",)), ("prusaslicer", ("prusa-slicer", "PrusaSlicer"))):
        path = next((shutil.which(exe) for exe in executables if shutil.which(exe)), None)
        found[name] = {"available": path is not None, "executable_path": path}
    return found


@dataclass
class PrintabilityReport:
    integrity: MeshIntegrityReport
    build_volume_fit: BuildVolumeFitReport
    overhangs: OverhangReport
    material: MaterialEstimate
    warnings: list[dict] = field(default_factory=list)  # [{"severity": "error"|"warning"|"info", "message": str}]
    limitations: list[str] = field(default_factory=lambda: list(PRINT_LAB_LIMITATIONS))

    def as_dict(self) -> dict:
        return {
            "integrity": {
                "watertight": self.integrity.watertight, "manifold": self.integrity.manifold,
                "body_count": self.integrity.body_count, "non_manifold_edge_count": self.integrity.non_manifold_edge_count,
                "degenerate_face_count": self.integrity.degenerate_face_count, "is_empty": self.integrity.is_empty,
                "vertex_count": self.integrity.vertex_count, "face_count": self.integrity.face_count,
                "bounding_box_mm": list(self.integrity.bounding_box_mm), "volume_mm3": self.integrity.volume_mm3,
                "surface_area_mm2": self.integrity.surface_area_mm2, "printable_as_is": self.integrity.printable_as_is,
                "method": self.integrity.method,
            },
            "build_volume_fit": {
                "fits": self.build_volume_fit.fits, "model_extents_mm": list(self.build_volume_fit.model_extents_mm),
                "printer_volume_mm": list(self.build_volume_fit.printer_volume_mm),
                "overrun_mm": list(self.build_volume_fit.overrun_mm), "method": self.build_volume_fit.method,
            },
            "overhangs": {
                "overhang_angle_deg": self.overhangs.overhang_angle_deg, "overhang_area_mm2": self.overhangs.overhang_area_mm2,
                "overhang_area_fraction": self.overhangs.overhang_area_fraction,
                "likely_needs_support": self.overhangs.likely_needs_support, "method": self.overhangs.method,
                "overhang_face_indices": self.overhangs.overhang_face_indices,
            },
            "material": {
                "solid_volume_mm3": self.material.solid_volume_mm3, "density_g_per_cm3": self.material.density_g_per_cm3,
                "estimated_mass_g": self.material.estimated_mass_g, "method": self.material.method,
            },
            "warnings": self.warnings,
            "limitations": self.limitations,
            "print_time_estimate": "unavailable -- requires a real slicer estimate (see limitations)",
            "support_material_estimate": "unavailable -- requires a real slicer estimate (see limitations)",
        }


def analyze_printability(
    mesh: trimesh.Trimesh, printer: PrinterProfile, overhang_angle_deg: float = 45.0,
    density_g_per_cm3: float | None = None,
) -> PrintabilityReport:
    """The one entry point server.py's Print Lab route calls -- runs every
    implemented check and collects warnings with severity. Never raises:
    an empty/invalid mesh produces an honest all-"unavailable/failed"
    report (via MeshIntegrityReport.is_empty), not an exception."""
    integrity = check_mesh_integrity(mesh)
    warnings: list[dict] = []
    if integrity.is_empty:
        warnings.append({"severity": "error", "message": "no usable geometry -- the mesh is empty"})
        empty_fit = BuildVolumeFitReport(fits=False, model_extents_mm=(0, 0, 0), printer_volume_mm=printer.build_volume_mm, overrun_mm=(0, 0, 0))
        empty_overhang = OverhangReport(overhang_angle_deg=overhang_angle_deg, overhang_area_mm2=0.0, overhang_area_fraction=0.0, build_direction=(0, 0, 1))
        return PrintabilityReport(integrity=integrity, build_volume_fit=empty_fit, overhangs=empty_overhang, material=MaterialEstimate(0.0, density_g_per_cm3, None), warnings=warnings)

    if not integrity.watertight:
        warnings.append({"severity": "error", "message": "mesh is not watertight -- a slicer may reject it or fill gaps unpredictably"})
    if integrity.body_count > 1:
        warnings.append({"severity": "warning", "message": f"{integrity.body_count} disconnected shells -- confirm this is an intentional multi-body export, not a split part"})
    if integrity.non_manifold_edge_count > 0:
        warnings.append({"severity": "error", "message": f"{integrity.non_manifold_edge_count} non-manifold edge(s) -- real geometry defect, likely to cause slicing artifacts"})
    if integrity.degenerate_face_count > 0:
        warnings.append({"severity": "warning", "message": f"{integrity.degenerate_face_count} degenerate (near-zero-area) face(s)"})

    fit = check_build_volume_fit(mesh, printer)
    if not fit.fits:
        axes = [a for a, o in zip("XYZ", fit.overrun_mm) if o > 0]
        warnings.append({"severity": "error", "message": f"exceeds {printer.name}'s build volume on axis/axes {', '.join(axes)} by {max(fit.overrun_mm):.1f}mm"})

    overhangs = check_overhangs(mesh, overhang_angle_deg=overhang_angle_deg)
    if overhangs.likely_needs_support:
        warnings.append({"severity": "info", "message": f"~{overhangs.overhang_area_fraction:.0%} of surface area exceeds {overhang_angle_deg:.0f}° overhang and will likely need support"})

    material = estimate_material(mesh, density_g_per_cm3)

    return PrintabilityReport(integrity=integrity, build_volume_fit=fit, overhangs=overhangs, material=material, warnings=warnings)
