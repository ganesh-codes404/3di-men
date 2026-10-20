"""Stage D counterpart for free-form objects: constructs a real CSG solid
from a FreeformSpec's primitives + boolean operations, via trimesh's own
primitive creators and boolean engine (the same "manifold" engine
mitsugen/builder.py's apply_hollowing already uses) -- never a stand-in
for an unbuilt shape, and never a call into 3di-men's own sandbox.py
tool-composing engine (see mitsugen/__init__.py)."""

from __future__ import annotations

import numpy as np
import trimesh

from mitsugen.builder import BuildResult
from mitsugen.freeform import FreeformSpec, PrimitiveSpec


def _euler_matrix_deg(angles_deg: tuple[float, float, float]) -> np.ndarray:
    """X-then-Y-then-Z fixed-order Euler rotation, in degrees -- the same
    simple convention documented in this module's own primitive schema,
    not a general quaternion."""
    rx, ry, rz = np.radians(angles_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _build_primitive_mesh(p: PrimitiveSpec) -> "trimesh.Trimesh | None":
    """None if this primitive's own fields don't support construction --
    disclosed by the caller, never guessed at."""
    if p.kind == "box":
        if p.size_mm is None:
            return None
        mesh = trimesh.creation.box(extents=list(p.size_mm))
    elif p.kind == "sphere":
        if not p.radius_mm or p.radius_mm <= 0:
            return None
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=p.radius_mm)
    elif p.kind == "cylinder":
        if not p.radius_mm or not p.height_mm or p.radius_mm <= 0 or p.height_mm <= 0:
            return None
        mesh = trimesh.creation.cylinder(radius=p.radius_mm, height=p.height_mm, sections=32)
    elif p.kind == "cone":
        if not p.radius_mm or not p.height_mm or p.radius_mm <= 0 or p.height_mm <= 0:
            return None
        mesh = trimesh.creation.cone(radius=p.radius_mm, height=p.height_mm, sections=32)
        mesh.apply_translation([0.0, 0.0, -p.height_mm / 2.0])  # trimesh's cone spans z=[0, height] -- recenter so position_mm means the same "own center" thing for every primitive kind
    else:
        return None

    transform = np.eye(4)
    transform[:3, :3] = _euler_matrix_deg(p.rotation_deg)
    transform[:3, 3] = p.position_mm
    mesh.apply_transform(transform)
    return mesh


def build_freeform_solid(spec: FreeformSpec) -> BuildResult:
    """Never raises -- any construction failure (an unbuildable primitive,
    a failed boolean op, a non-watertight or empty result) comes back as
    a typed BuildResult(mesh=None, error=...), matching
    mitsugen/builder.py's own discipline for the radial-sweep path."""
    if not spec.primitives:
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error="no primitives to build")

    meshes_by_id: dict[str, trimesh.Trimesh] = {}
    unbuildable: list[str] = []
    for p in spec.primitives:
        mesh = _build_primitive_mesh(p)
        if mesh is None:
            unbuildable.append(p.id)
            continue
        meshes_by_id[p.id] = mesh

    if unbuildable:
        # A primitive the spec called for but couldn't actually build is a
        # requested feature this would otherwise go missing -- fail loudly
        # rather than silently export a solid that's missing a wheel/handle/
        # whatever that primitive was supposed to be, matching the "never a
        # partial/guessed mesh on failure" discipline builder.py documents.
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error=f"primitive(s) could not be built from their own fields: {', '.join(unbuildable)}")

    base_id = spec.primitives[0].id
    result = meshes_by_id[base_id]
    for step in spec.operations:
        target = meshes_by_id[step.target_id]
        try:
            if step.op == "union":
                result = result.union(target, engine="manifold")
            elif step.op == "difference":
                result = result.difference(target, engine="manifold")
            else:
                result = result.intersection(target, engine="manifold")
        except Exception as exc:
            return BuildResult(mesh=None, profiles=[], hollow_applied=False, error=f"boolean {step.op} with '{step.target_id}' failed: {exc}")

    if result is None or result.is_empty:
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error="combined primitive geometry is empty")
    if not result.is_watertight:
        result.fix_normals()
    if not result.is_watertight:
        return BuildResult(mesh=None, profiles=[], hollow_applied=False, error="combined primitive geometry is not watertight -- rejected rather than exported as-is")

    return BuildResult(mesh=result, profiles=[], hollow_applied=False, error=None)
