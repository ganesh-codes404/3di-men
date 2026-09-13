import pytest

from threedimen.teacher import MockTeacher, TEACHERS
from threedimen.teacher.hunyuan3d import Hunyuan3DConfigError, Hunyuan3DTeacher


def test_mock_teacher_generates_a_mesh_file():
    artifact = MockTeacher().generate("a small bracket")
    assert artifact.mesh_path.exists()
    assert artifact.source_model == "mock"
    assert artifact.prompt == "a small bracket"


def test_mock_teacher_is_deterministic():
    a1 = MockTeacher().generate("a centrifugal water pump")
    a2 = MockTeacher().generate("a centrifugal water pump")
    assert a1.metadata["shape"] == a2.metadata["shape"]
    assert a1.metadata["deterministic_seed"] == a2.metadata["deterministic_seed"]


def test_mock_teacher_varies_by_prompt():
    a1 = MockTeacher().generate("prompt one")
    a2 = MockTeacher().generate("a completely different prompt")
    assert a1.metadata["deterministic_seed"] != a2.metadata["deterministic_seed"]


def test_teacher_registry_contains_hunyuan3d_and_mock():
    assert "hunyuan3d" in TEACHERS
    assert "mock" in TEACHERS


def test_hunyuan3d_teacher_without_endpoint_raises_config_error(monkeypatch):
    monkeypatch.delenv("THREEDIMEN_HUNYUAN3D_ENDPOINT", raising=False)
    teacher = Hunyuan3DTeacher()
    with pytest.raises(Hunyuan3DConfigError):
        teacher.generate("anything")


def test_hunyuan3d_teacher_rejects_invalid_mode():
    with pytest.raises(Hunyuan3DConfigError):
        Hunyuan3DTeacher(endpoint="https://example.com", mode="not-a-real-mode")


def test_hunyuan3d_teacher_reads_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("THREEDIMEN_HUNYUAN3D_ENDPOINT", "https://example.trycloudflare.com/")
    teacher = Hunyuan3DTeacher()
    assert teacher.endpoint == "https://example.trycloudflare.com"  # trailing slash stripped


class _FakeResponse:
    def __init__(self, json_data, content: bytes = b""):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_hunyuan3d_teacher_records_the_actual_generating_model(monkeypatch, tmp_path):
    """Regression test: the server reports which model actually ran
    (result["part_type"]) -- direct mode really runs Shap-E, not Hunyuan3D.
    source_model must reflect that, not just the teacher class's own name,
    otherwise every direct-mode sample gets mislabeled "hunyuan3d-2.1" in the
    dataset when Shap-E did the work."""
    import threedimen.teacher.hunyuan3d as h3d_module

    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/generate")
        assert json["mode"] == "direct"
        return _FakeResponse({"job_id": "job123"})

    def fake_get(url, timeout=None):
        if url.endswith("/jobs/job123"):
            return _FakeResponse({
                "status": "done",
                "result": {
                    "part_type": "shap-e",
                    "mesh_url": "/mesh/abc.stl",
                    "resolved_description": "Shap-E (direct text-to-3D) generation",
                    "generation_time_seconds": 1.23,
                },
            })
        assert url.endswith("/mesh/abc.stl")
        return _FakeResponse({}, content=b"fake stl bytes")

    monkeypatch.setattr(h3d_module.requests, "post", fake_post)
    monkeypatch.setattr(h3d_module.requests, "get", fake_get)

    teacher = Hunyuan3DTeacher(
        endpoint="https://example.com", mode="direct", cache_dir=tmp_path, poll_interval_seconds=0
    )
    artifact = teacher.generate("a bracket")

    assert artifact.source_model == "shap-e"  # NOT "hunyuan3d-2.1[direct]"
    assert artifact.metadata["requested_mode"] == "direct"
    assert artifact.mesh_path.read_bytes() == b"fake stl bytes"
