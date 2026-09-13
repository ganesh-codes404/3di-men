from threedimen.geometry.ingest import ingest_mesh
from threedimen.quality import QualityThresholds, assess_quality


def test_good_mesh_is_valid(box_mesh):
    surface = ingest_mesh(box_mesh, strict=True).surface
    report = assess_quality(surface)
    assert report.valid
    assert report.watertight
    assert report.reasons == []


def test_too_few_vertices_rejected(box_mesh):
    surface = ingest_mesh(box_mesh, strict=True).surface
    report = assess_quality(surface, thresholds=QualityThresholds(min_vertex_count=1000))
    assert not report.valid
    assert any("too few vertices" in r for r in report.reasons)


def test_disconnected_components_counted(sphere_mesh, box_mesh):
    import trimesh

    combined = trimesh.util.concatenate([box_mesh, sphere_mesh.apply_translation([10, 10, 10])])
    surface = ingest_mesh(combined, strict=True).surface
    report = assess_quality(surface)
    assert report.disconnected_components == 2


def test_duplicate_detection(box_mesh):
    surface = ingest_mesh(box_mesh, strict=True).surface
    report1 = assess_quality(surface)
    report2 = assess_quality(surface, existing_duplicate_keys={report1.duplicate_key})
    assert not report2.valid
    assert any("duplicate" in r for r in report2.reasons)


def test_bbox_sanity_rejects_bbox_below_threshold(box_mesh):
    # box_mesh has a real bounding-box diagonal (~5.4); a threshold set above
    # that should reject it as "too small" per the caller's own sanity range,
    # independent of whether the mesh's own geometry is actually degenerate.
    surface = ingest_mesh(box_mesh, normalize_scale=False, strict=True).surface
    report = assess_quality(surface, thresholds=QualityThresholds(min_bbox_diagonal=100.0))
    assert not report.valid
    assert any("bounding box diagonal" in r for r in report.reasons)
