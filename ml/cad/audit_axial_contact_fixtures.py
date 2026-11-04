"""Read-only audit script -- Part 2 of the "Axial Contact Gap Reproduction &
Design Audit" mission brief. NOT production code, NOT a test, NOT imported
by anything. Builds 5 hand-authored fixtures through the REAL production
pipeline (parallel_builder.build_assembly_from_points, assembly_plan's real
resolve_position/resolve_rotation_deg, llm_decompose's real
_correct_gear_mesh_spacing) and measures the resulting geometry with real,
non-mocked trimesh queries. No LLM call anywhere in this script.

Run with: cd ml/cad && python3 audit_axial_contact_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from assembly_plan import Relationship, resolve_rotation_deg
from llm_decompose import ActionablePoint, _correct_gear_mesh_spacing
from parallel_builder import PlacementTransform, build_assembly_from_points

BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "program_model.joblib"

SAMPLE_N = 2000


def world_mesh_for(result, pid: str) -> tuple[trimesh.Trimesh, tuple, tuple]:
    """Reconstructs the REAL world-placed mesh for one point, using the
    pipeline's own already-resolved position_mm (parallel_builder.py stores
    the RESOLVED position back into PointBuildResult.point.position_mm, not
    the original placeholder) and the real resolve_rotation_deg (pure,
    deterministic, safe to call again for inspection -- same function
    parallel_builder.py itself calls)."""
    pr = next(p for p in result.points if p.point.id == pid or (p.point.id == "" and pid == ""))
    local_mesh = pr.mesh
    position_mm = pr.point.position_mm
    rel = pr.point.relationship
    rotation_deg = resolve_rotation_deg(rel) if rel is not None else (0.0, 0.0, 0.0)
    transform = PlacementTransform.translation(position_mm).compose(PlacementTransform.rotation(rotation_deg))
    placed = transform.apply_to_mesh(local_mesh)
    return placed, position_mm, rotation_deg


def describe_point(result, pid: str) -> dict:
    pr = next(p for p in result.points if p.point.id == pid)
    placed, position_mm, rotation_deg = world_mesh_for(result, pid)
    return {
        "id": pid,
        "status": pr.status,
        "error": pr.error,
        "program": pr.program,
        "local_bounds": pr.mesh.bounds.tolist() if pr.mesh is not None else None,
        "resolved_position_mm": position_mm,
        "resolved_rotation_deg": rotation_deg,
        "world_bounds": placed.bounds.tolist(),
        "is_watertight": bool(placed.is_watertight),
        "_placed_mesh": placed,
    }


def pair_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n: int = SAMPLE_N) -> dict:
    """Sampling-based min surface-to-surface distance (exact closest point
    per sample via trimesh.proximity.closest_point, approximate only in
    WHICH points get queried) plus a watertight containment check in both
    directions. See mission-brief methodology: python-fcl is not installed
    on this machine, so this is the best available substitute, with an
    explicitly reported tolerance."""
    area_a = float(mesh_a.area)
    area_b = float(mesh_b.area)
    tol_a = float(np.sqrt(area_a / n)) if area_a > 0 else float("inf")
    tol_b = float(np.sqrt(area_b / n)) if area_b > 0 else float("inf")
    tolerance = max(tol_a, tol_b)

    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n)
    pts_b, _ = trimesh.sample.sample_surface(mesh_b, n)

    _, dist_a_to_b, _ = trimesh.proximity.closest_point(mesh_b, pts_a)
    _, dist_b_to_a, _ = trimesh.proximity.closest_point(mesh_a, pts_b)
    min_dist = float(min(dist_a_to_b.min(), dist_b_to_a.min()))

    contains_info = {}
    contained_any = False
    if mesh_b.is_watertight:
        in_b = mesh_b.contains(pts_a)
        contains_info["a_points_inside_b"] = int(in_b.sum())
        contained_any = contained_any or bool(in_b.any())
    else:
        contains_info["a_points_inside_b"] = "mesh_b not watertight -- containment check skipped"
    if mesh_a.is_watertight:
        in_a = mesh_a.contains(pts_b)
        contains_info["b_points_inside_a"] = int(in_a.sum())
        contained_any = contained_any or bool(in_a.any())
    else:
        contains_info["b_points_inside_a"] = "mesh_a not watertight -- containment check skipped"

    if contained_any:
        classification = "OVERLAPPING"
    elif min_dist <= tolerance:
        classification = "TOUCHING"
    else:
        classification = "SEPARATED"

    return {
        "sample_n": n,
        "tolerance_mm": round(tolerance, 4),
        "min_surface_distance_mm": round(min_dist, 4),
        "containment": contains_info,
        "classification": classification,
    }


def run(label: str, points: list[ActionablePoint], pairs: list[tuple[str, str]], note: str = "") -> dict:
    print(f"\n{'=' * 90}\nFIXTURE: {label}\n{'=' * 90}")
    if note:
        print(f"note: {note}")
    result = build_assembly_from_points(points, BUNDLE_PATH, max_workers=4)
    out = {"label": label, "note": note, "components": {}, "pairs": {}}
    for p in points:
        pid = p.id
        d = describe_point(result, pid)
        placed = d.pop("_placed_mesh")
        out["components"][pid] = d
        print(f"  [{pid}] status={d['status']} program_ops={[s['op'] for s in d['program']]}")
        print(f"       local_bounds={d['local_bounds']}")
        print(f"       resolved_position_mm={d['resolved_position_mm']} resolved_rotation_deg={d['resolved_rotation_deg']}")
        print(f"       world_bounds={d['world_bounds']} watertight={d['is_watertight']}")
        if d["error"]:
            print(f"       error: {d['error']}")
    meshes = {pid: describe_point(result, pid)["_placed_mesh"] for pid in [p.id for p in points]}
    for a, b in pairs:
        if meshes[a] is None or meshes[b] is None:
            print(f"  PAIR {a} <-> {b}: skipped, missing mesh")
            continue
        pd = pair_distance(meshes[a], meshes[b])
        out["pairs"][f"{a}<->{b}"] = pd
        print(f"  PAIR {a} <-> {b}: {pd['classification']} "
              f"(min_dist={pd['min_surface_distance_mm']}mm, tol={pd['tolerance_mm']}mm, "
              f"containment={pd['containment']})")
    return out


def fixture_1_shaft_to_shaft() -> list[dict]:
    variants = []
    for label_suffix, axis_offset in [("axis_offset=0.0 (schema default)", 0.0),
                                       ("axis_offset=40.0 (naive guess)", 40.0),
                                       ("axis_offset=50.0 (geometrically correct flush)", 50.0)]:
        shaft_a = ActionablePoint(description="a cylinder 10mm diameter, 40mm long", position_mm=(0, 0, 0), id="shaft_a")
        shaft_b = ActionablePoint(
            description="a cylinder 10mm diameter, 60mm long", position_mm=(0, 0, 0), id="shaft_b",
            relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="z", axis_offset=axis_offset),
        )
        variants.append(run(f"1. Two shaft segments end-to-end, {label_suffix}",
                             [shaft_a, shaft_b], [("shaft_a", "shaft_b")],
                             note="intent: flush contact, zero gap, zero overlap"))
    return variants


def fixture_2_shaft_sleeve() -> dict:
    shaft = ActionablePoint(description="a cylinder 10mm diameter, 80mm long", position_mm=(0, 0, 0), id="shaft")
    sleeve = ActionablePoint(
        description="a ring 30mm outer diameter, 10.4mm inner diameter, 20mm tall", position_mm=(0, 0, 0), id="sleeve",
        relationship=Relationship(kind="concentric_with", target_id="shaft", axis="z", axis_offset=0.0),
    )
    out = run("2. Shaft with concentric sliding-fit sleeve", [shaft, sleeve], [("shaft", "sleeve")],
               note="intent: deliberate 0.2mm radial clearance + partial axial overlap (correct real-world behavior, NOT a bug)")
    return out


def fixture_3_standoff_spacer() -> list[dict]:
    variants = []
    for label_suffix, axis_offset in [("axis_offset=0.0 (schema default)", 0.0),
                                       ("axis_offset=15.0 (naive guess)", 15.0),
                                       ("axis_offset=10.0 (geometrically correct flush)", 10.0)]:
        standoff = ActionablePoint(description="a cylinder 8mm diameter, 15mm tall", position_mm=(0, 0, 0), id="standoff")
        spacer = ActionablePoint(
            description="a cylinder 8mm diameter, 5mm tall", position_mm=(0, 0, 0), id="spacer",
            relationship=Relationship(kind="concentric_with", target_id="standoff", axis="z", axis_offset=axis_offset),
        )
        variants.append(run(f"3. Spacer stacked on standoff, {label_suffix}",
                             [standoff, spacer], [("standoff", "spacer")],
                             note="intent: flush contact, zero gap, zero overlap"))
    return variants


def fixture_4_gear_pair() -> dict:
    shaft_a = ActionablePoint(description="a cylinder 10mm diameter, 60mm long", position_mm=(0, 0, 0), id="shaft_a")
    gear_a = ActionablePoint(
        description="a gear profile, module 2.5, 14 teeth, 8mm face width, 5mm bore diameter", position_mm=(0, 0, 0), id="gear_a",
        relationship=Relationship(kind="concentric_with", target_id="shaft_a", axis="z"),
    )
    shaft_b = ActionablePoint(
        description="a cylinder 10mm diameter, 60mm long", position_mm=(0, 0, 0), id="shaft_b",
        relationship=Relationship(kind="parallel_to", target_id="shaft_a", axis="z", lateral_axis="x", lateral_offset=30.0),
    )
    gear_b = ActionablePoint(
        description="a gear profile, module 2.5, 22 teeth, 8mm face width, 5mm bore diameter", position_mm=(0, 0, 0), id="gear_b",
        relationship=Relationship(kind="concentric_with", target_id="shaft_b", axis="z"),
    )
    points = [shaft_a, gear_a, shaft_b, gear_b]

    # Verify the real corrector directly, BEFORE feeding into the full pipeline
    # (which also applies it unconditionally -- confirming both call sites agree).
    corrected = _correct_gear_mesh_spacing(points)
    shaft_b_corrected = next(p for p in corrected if p.id == "shaft_b")
    expected_lateral = 2.5 * (14 + 22) / 2.0
    print(f"\n_correct_gear_mesh_spacing: shaft_b.relationship.lateral_offset = "
          f"{shaft_b_corrected.relationship.lateral_offset} (expected {expected_lateral})")
    assert abs(shaft_b_corrected.relationship.lateral_offset - expected_lateral) < 1e-6, "corrector did not produce the expected value!"

    out = run("4. Two-gear meshing pair (positive control)", points, [("gear_a", "gear_b")],
               note=f"center distance (via lateral_offset) should be exactly {expected_lateral}mm; "
                    f"addendum radii ~20mm (gear_a) + ~30mm (gear_b) = 50mm > {expected_lateral}mm by design -- "
                    "SOME tip interference/near-contact is CORRECT (that is what meshing means), not a bug. "
                    "Gear-tooth contact is intermittent/point-contact as teeth rotate past each other, not a "
                    "continuous touching surface -- the SEPARATED/TOUCHING/OVERLAPPING labels below are reported "
                    "as raw measurements only, not as a pass/fail verdict for this fixture.")
    out["expected_center_distance_mm"] = expected_lateral
    out["corrector_verified"] = True
    # Addendum-radius sanity check, reported not asserted (approximate formula)
    out["addendum_radius_gear_a_mm"] = 2.5 * (14 + 2) / 2.0
    out["addendum_radius_gear_b_mm"] = 2.5 * (22 + 2) / 2.0
    return out


def fixture_5_intentional_gap() -> dict:
    collar_a = ActionablePoint(description="a cylinder 12mm diameter, 20mm long", position_mm=(0, 0, 0), id="collar_a")
    collar_b = ActionablePoint(
        description="a cylinder 12mm diameter, 20mm long", position_mm=(0, 0, 0), id="collar_b",
        relationship=Relationship(kind="concentric_with", target_id="collar_a", axis="z", axis_offset=25.0),
    )
    out = run("5. Two collars with an explicitly intentional 5mm gap", [collar_a, collar_b], [("collar_a", "collar_b")],
               note="intent: deliberate ~5mm gap (half_a 10 + half_b 10 + 5mm intended clearance = 25.0 axis_offset)")
    return out


def main():
    all_results = {}
    all_results["fixture_1_variants"] = fixture_1_shaft_to_shaft()
    all_results["fixture_2"] = fixture_2_shaft_sleeve()
    all_results["fixture_3_variants"] = fixture_3_standoff_spacer()
    all_results["fixture_4"] = fixture_4_gear_pair()
    all_results["fixture_5"] = fixture_5_intentional_gap()

    out_path = Path(__file__).resolve().parent / "audit_results" / "axial_contact_fixtures.json"
    out_path.parent.mkdir(exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "_placed_mesh"}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2, default=str)
    print(f"\n\nFull structured results written to {out_path}")


if __name__ == "__main__":
    main()
