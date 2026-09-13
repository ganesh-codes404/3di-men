import numpy as np
import pytest
import trimesh

from threedimen.geometry.errors import MeshValidationError
from threedimen.geometry.ingest import ingest_mesh


def test_ingest_valid_mesh(box_mesh):
    result = ingest_mesh(box_mesh, strict=True)
    assert result.validation.ok
    assert result.surface.summary_dict()["vertex_count"] == 8


def test_ingest_recenters_and_normalizes_scale(box_mesh):
    result = ingest_mesh(box_mesh, normalize_scale=True, target_diagonal=1.0, strict=True)
    bb = result.surface.bounding_box
    assert np.isclose(bb.diagonal, 1.0, atol=1e-6)
    assert np.allclose(bb.center, 0.0, atol=1e-9)


def test_ingest_without_scale_normalization_preserves_scale(box_mesh):
    original_diagonal = float(np.linalg.norm(box_mesh.bounds[1] - box_mesh.bounds[0]))
    result = ingest_mesh(box_mesh, normalize_scale=False, strict=True)
    assert np.isclose(result.surface.bounding_box.diagonal, original_diagonal)


def test_normalization_is_invertible(box_mesh):
    result = ingest_mesh(box_mesh, normalize_scale=True, strict=True)
    norm = result.surface.metadata["normalization"]
    stored_vertices = result.surface.patches[0].vertices
    # row order/count is preserved through ingest (only translate+scale are
    # applied), so reconstructed[i] must correspond exactly to box_mesh.vertices[i]
    reconstructed = stored_vertices / norm["scale_factor"] + np.array(norm["original_center"])
    np.testing.assert_allclose(reconstructed, box_mesh.vertices, atol=1e-6)


def test_ingest_empty_mesh_fails_validation_non_strict(empty_mesh):
    result = ingest_mesh(empty_mesh, strict=False)
    assert not result.validation.ok
    assert "zero vertices" in " ".join(result.validation.errors)


def test_ingest_empty_mesh_raises_in_strict_mode(empty_mesh):
    with pytest.raises(MeshValidationError):
        ingest_mesh(empty_mesh, strict=True)


def test_ingest_nan_vertices_fails_validation(nan_mesh):
    result = ingest_mesh(nan_mesh, strict=False)
    assert not result.validation.ok
    assert any("non-finite" in e for e in result.validation.errors)


def test_ingest_repairs_degenerate_faces(degenerate_face_mesh):
    result = ingest_mesh(degenerate_face_mesh, repair=True, strict=True)
    assert any("degenerate" in r for r in result.repairs_applied)
    v, f, _ = result.surface.patches[0].to_mesh_data()
    areas = trimesh.Trimesh(vertices=v, faces=f, process=False).area_faces
    assert np.all(areas > 1e-12)


def test_ingest_without_repair_leaves_degenerate_faces(degenerate_face_mesh):
    result = ingest_mesh(degenerate_face_mesh, repair=False, strict=True)
    assert not any("degenerate" in r for r in result.repairs_applied)


def test_ingest_gltf_axis_conversion(box_mesh, tmp_path):
    glb_path = tmp_path / "box.glb"
    box_mesh.export(glb_path)
    result = ingest_mesh(glb_path, strict=True)
    assert result.surface.metadata["axis_converted"] is True
    assert any("Y-up" in r for r in result.repairs_applied)


def test_ingest_obj_does_not_axis_convert(box_mesh, tmp_path):
    obj_path = tmp_path / "box.obj"
    box_mesh.export(obj_path)
    result = ingest_mesh(obj_path, strict=True)
    assert result.surface.metadata["axis_converted"] is False


def test_ingest_wrong_type_raises():
    with pytest.raises(TypeError):
        ingest_mesh(12345)


def test_ingest_tiny_but_legitimate_mesh_is_not_stripped_to_nothing():
    # Regression test: a fixed absolute degenerate-face-area threshold would
    # misclassify every face of a legitimately tiny mesh as degenerate and
    # strip the mesh down to zero faces (then crash on mesh.bounds). The
    # threshold must scale with the mesh's own size instead.
    tiny = trimesh.creation.box(extents=[1e-9, 1e-9, 1e-9])
    result = ingest_mesh(tiny, normalize_scale=False, strict=True)
    assert result.validation.ok
    assert result.surface.summary_dict()["face_count"] == 12
    assert not any("degenerate" in r for r in result.repairs_applied)
