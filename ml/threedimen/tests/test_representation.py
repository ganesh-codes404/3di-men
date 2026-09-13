import numpy as np

from threedimen.geometry.representation import (
    BoundingBox,
    MeshPatch,
    SurfaceRepresentation,
    Transform,
)


def test_bounding_box_from_points():
    points = np.array([[0, 0, 0], [1, 2, 3], [-1, 0, 0]])
    bb = BoundingBox.from_points(points)
    assert np.allclose(bb.min, [-1, 0, 0])
    assert np.allclose(bb.max, [1, 2, 3])
    assert np.allclose(bb.center, [0, 1, 1.5])


def test_bounding_box_union():
    a = BoundingBox(np.array([0, 0, 0]), np.array([1, 1, 1]))
    b = BoundingBox(np.array([-1, -1, -1]), np.array([0.5, 0.5, 0.5]))
    u = BoundingBox.union([a, b])
    assert np.allclose(u.min, [-1, -1, -1])
    assert np.allclose(u.max, [1, 1, 1])


def test_transform_identity_and_apply():
    t = Transform.identity()
    assert t.is_identity
    points = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(t.apply(points), points)


def test_mesh_patch_from_trimesh_and_back(box_mesh):
    patch = MeshPatch.from_trimesh(box_mesh)
    v, f, n = patch.to_mesh_data()
    assert v.shape == (8, 3)
    assert f.shape == (12, 3)
    bb = patch.bounding_box()
    assert np.allclose(bb.size, [2.0, 3.0, 4.0])


def test_surface_representation_from_trimesh_round_trip(box_mesh):
    surface = SurfaceRepresentation.from_trimesh(box_mesh, note="test")
    assert surface.metadata["note"] == "test"
    mesh2 = surface.to_trimesh()
    assert len(mesh2.vertices) == len(box_mesh.vertices)
    assert len(mesh2.faces) == len(box_mesh.faces)


def test_surface_representation_summary(box_mesh):
    surface = SurfaceRepresentation.from_trimesh(box_mesh)
    summary = surface.summary_dict()
    assert summary["vertex_count"] == 8
    assert summary["face_count"] == 12
    assert summary["patch_kinds"] == ["mesh"]


def test_surface_representation_save_load_round_trip(box_mesh, tmp_path):
    surface = SurfaceRepresentation.from_trimesh(box_mesh, source="unit test")
    json_path = surface.save(tmp_path / "surface")
    assert json_path.exists()
    assert (tmp_path / "surface.npz").exists()

    loaded = SurfaceRepresentation.load(tmp_path / "surface")
    assert loaded.metadata["source"] == "unit test"
    assert loaded.summary_dict() == surface.summary_dict()
    np.testing.assert_allclose(loaded.patches[0].vertices, surface.patches[0].vertices)
    np.testing.assert_array_equal(loaded.patches[0].faces, surface.patches[0].faces)


def test_surface_representation_load_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        SurfaceRepresentation.load(tmp_path / "does_not_exist")


def test_unknown_patch_kind_raises_on_load():
    import pytest

    from threedimen.geometry.representation import SurfacePatch

    with pytest.raises(ValueError):
        SurfacePatch.from_dict({"kind": "nonexistent_kind", "meta": {}}, "patch0", {})
