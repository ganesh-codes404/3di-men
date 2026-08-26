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

    plate = trimesh.creation.extrude_polygon(Polygon(outline), height=params.thickness_mm)
    plate.apply_translation([0, 0, -params.thickness_mm / 2])

    for cx, cy in params.hole_centers():
        hole = trimesh.creation.cylinder(
            radius=params.hole_diameter_mm / 2,
            height=params.thickness_mm * 3,  # guarantee it fully punches through
            sections=32,
        )
        hole.apply_translation([cx, cy, 0])
        plate = plate.difference(hole, engine="manifold")

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
        material=rng.choice(["aluminum", "steel", "polycarbonate"]),
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
        material=rng.choice(["steel", "aluminum", "cast iron"]),
    )


def generate_dataset(out_dir: Path, n: int, seed: int = 0) -> list[dict]:
    """Generate n (mesh, description, params) triples under out_dir. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = []
    for i in range(n):
        params = random_mounting_plate(rng)
        mesh = build_mounting_plate(params)
        stem = f"mounting_plate_{i:04d}"
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
