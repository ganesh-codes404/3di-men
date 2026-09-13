"""Tests for the text-to-shape proof-of-concept model (threedimen.model).
All requiring torch -- if the `train` extra isn't installed, this whole file
is skipped rather than failing the rest of the suite."""

import pytest

torch = pytest.importorskip("torch")

from threedimen.dataset import DatasetStore
from threedimen.geometry.ingest import ingest_mesh
from threedimen.model.chamfer import chamfer_distance
from threedimen.model.device import resolve_device
from threedimen.model.generate import generate
from threedimen.model.networks import PointCloudDecoder, PointCloudEncoder, TextEncoder
from threedimen.model.train import TrainConfig, train
from threedimen.model.vocab import Vocabulary
from threedimen.quality import assess_quality
from threedimen.teacher import MockTeacher


# ---- device -----------------------------------------------------------------


def test_resolve_device_explicit_cpu_always_works():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_matches_cuda_availability():
    resolved = resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolved.type == expected


def test_resolve_device_explicit_cuda_raises_clearly_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("this test asserts the no-GPU error path; a GPU is present here")
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        resolve_device("cuda")


# ---- vocab ----------------------------------------------------------------


def test_vocab_build_and_encode_round_trip():
    vocab = Vocabulary.build(["a small bracket", "a large pump"])
    ids = vocab.encode("a small bracket", max_len=10)
    assert len(ids) == 10
    assert ids[-1] == vocab.pad_id  # padded


def test_vocab_unknown_token_maps_to_unk():
    vocab = Vocabulary.build(["a small bracket"])
    ids = vocab.encode("a completely unseen sentence", max_len=8)
    unk_id = vocab.token_to_id["<unk>"]
    assert unk_id in ids


def test_vocab_serialization_round_trip():
    vocab = Vocabulary.build(["a pump", "a valve"])
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.token_to_id == vocab.token_to_id


# ---- chamfer ----------------------------------------------------------------


def test_chamfer_zero_for_identical_point_sets():
    points = torch.randn(2, 20, 3)
    assert chamfer_distance(points, points.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_chamfer_positive_for_different_point_sets():
    a = torch.randn(2, 20, 3)
    b = torch.randn(2, 20, 3) * 3
    assert chamfer_distance(a, b).item() > 0.0


# ---- network shapes ----------------------------------------------------------------


def test_text_encoder_output_shape():
    encoder = TextEncoder(vocab_size=50, embed_dim=16, latent_dim=8)
    tokens = torch.randint(0, 50, (4, 10))
    out = encoder(tokens)
    assert out.shape == (4, 8)


def test_point_cloud_encoder_output_shape():
    encoder = PointCloudEncoder(hidden_dim=16, latent_dim=8)
    points = torch.randn(4, 64, 3)
    out = encoder(points)
    assert out.shape == (4, 8)


def test_point_cloud_decoder_output_shape():
    decoder = PointCloudDecoder(latent_dim=8, num_points=64, hidden_dim=16)
    latent = torch.randn(4, 8)
    out = decoder(latent)
    assert out.shape == (4, 64, 3)


# ---- end-to-end train + generate ----------------------------------------------------------------


@pytest.fixture
def small_dataset(tmp_path):
    store = DatasetStore(tmp_path / "dataset")
    teacher = MockTeacher(cache_dir=tmp_path / "mock_cache")
    prompts = [
        "a small aluminum bracket",
        "a large steel pump housing",
        "a compact plastic valve",
        "a heavy cast iron motor",
    ]
    for prompt in prompts:
        artifact = teacher.generate(prompt)
        result = ingest_mesh(artifact.mesh_path, strict=True)
        report = assess_quality(result.surface)
        store.add_sample(artifact, result, category="test", quality=report.to_dict())
    return store.root


def _tiny_config(**overrides) -> TrainConfig:
    base = dict(num_points=32, embed_dim=8, hidden_dim=16, latent_dim=8, epochs=3, batch_size=2)
    base.update(overrides)
    return TrainConfig(**base)


def test_train_produces_a_checkpoint(small_dataset, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    result = train(small_dataset, checkpoint_path, _tiny_config())
    assert checkpoint_path.exists()
    assert result["num_samples"] == 4
    assert result["final_recon_loss"] >= 0


def test_train_requires_at_least_two_samples(tmp_path):
    empty_store = DatasetStore(tmp_path / "empty_dataset")
    with pytest.raises(ValueError):
        train(empty_store.root, tmp_path / "model.pt", _tiny_config())


def test_reconstruction_loss_decreases_over_training(small_dataset, tmp_path):
    short = train(small_dataset, tmp_path / "short.pt", _tiny_config(epochs=1, seed=0))
    long_run = train(small_dataset, tmp_path / "long.pt", _tiny_config(epochs=40, seed=0))
    assert long_run["final_recon_loss"] < short["final_recon_loss"]


def test_generate_produces_a_valid_mesh(small_dataset, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    train(small_dataset, checkpoint_path, _tiny_config(epochs=10))

    surface = generate("a brand new prompt never seen in training", checkpoint_path)
    mesh = surface.to_trimesh()
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_generate_is_deterministic_for_the_same_checkpoint_and_prompt(small_dataset, tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    train(small_dataset, checkpoint_path, _tiny_config(epochs=5))

    surface1 = generate("a compact bracket", checkpoint_path)
    surface2 = generate("a compact bracket", checkpoint_path)
    import numpy as np

    np.testing.assert_array_equal(surface1.patches[0].vertices, surface2.patches[0].vertices)


def test_train_and_generate_accept_explicit_cpu_device(small_dataset, tmp_path):
    """Exercises the exact same device-parameterized code path that would
    run on CUDA on a GPU machine -- everything except the literal CUDA
    kernel dispatch, which can't be tested without a GPU."""
    checkpoint_path = tmp_path / "model.pt"
    result = train(small_dataset, checkpoint_path, _tiny_config(epochs=5), device="cpu")
    assert result["device"] == "cpu"

    surface = generate("a compact bracket", checkpoint_path, device="cpu")
    assert len(surface.to_trimesh().vertices) > 0


def test_train_with_device_cuda_raises_clearly_when_unavailable(small_dataset, tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this test asserts the no-GPU error path; a GPU is present here")
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        train(small_dataset, tmp_path / "model.pt", _tiny_config(epochs=1), device="cuda")
