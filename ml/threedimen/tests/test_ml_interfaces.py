from threedimen.geometry.ingest import ingest_mesh
from threedimen.ml import ConvexHullDecoder, PointSampleEncoder


def test_point_sample_encoder_shapes(box_mesh):
    surface = ingest_mesh(box_mesh, strict=True).surface
    encoded = PointSampleEncoder(num_points=128, seed=0).encode(surface)
    assert encoded.points.shape == (128, 3)
    assert encoded.normals.shape == (128, 3)


def test_point_sample_encoder_deterministic_with_seed(box_mesh):
    surface = ingest_mesh(box_mesh, strict=True).surface
    e1 = PointSampleEncoder(num_points=64, seed=7).encode(surface)
    e2 = PointSampleEncoder(num_points=64, seed=7).encode(surface)
    import numpy as np

    np.testing.assert_array_equal(e1.points, e2.points)


def test_encode_decode_round_trip_produces_a_mesh(box_mesh):
    surface = ingest_mesh(box_mesh, strict=True).surface
    encoded = PointSampleEncoder(num_points=200, seed=0).encode(surface)
    decoded = ConvexHullDecoder().decode(encoded)
    summary = decoded.summary_dict()
    assert summary["vertex_count"] > 0
    assert summary["face_count"] > 0
