import json

import pytest

from threedimen.cli.main import main


def test_cli_teacher_generate(tmp_path, capsys):
    exit_code = main(["teacher", "generate", "a bracket", "--model", "mock", "--out", str(tmp_path)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["source_model"] == "mock"
    assert (tmp_path / __import__("pathlib").Path(out["mesh_path"]).name).exists()


def test_cli_mesh_inspect_and_validate(tmp_path, capsys):
    main(["teacher", "generate", "a bracket", "--model", "mock", "--out", str(tmp_path)])
    mesh_path = next(tmp_path.glob("*.stl"))
    capsys.readouterr()

    exit_code = main(["mesh", "inspect", str(mesh_path)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["validation"]["ok"] is True

    exit_code = main(["mesh", "validate", str(mesh_path)])
    assert exit_code == 0
    capsys.readouterr()


def test_cli_mesh_convert(tmp_path):
    main(["teacher", "generate", "a bracket", "--model", "mock", "--out", str(tmp_path)])
    mesh_path = next(tmp_path.glob("*.stl"))
    out_path = tmp_path / "converted.glb"

    exit_code = main(["mesh", "convert", str(mesh_path), str(out_path)])
    assert exit_code == 0
    assert out_path.exists()


def test_cli_dataset_add_stats_validate(tmp_path, capsys):
    dataset_root = tmp_path / "dataset"
    exit_code = main(
        ["dataset", "add", "--prompt", "a bracket", "--model", "mock", "--category", "bracket", "--root", str(dataset_root)]
    )
    assert exit_code == 0
    capsys.readouterr()

    exit_code = main(["dataset", "stats", "--root", str(dataset_root)])
    assert exit_code == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["sample_count"] == 1

    exit_code = main(["dataset", "validate", "--root", str(dataset_root)])
    assert exit_code == 0


def test_cli_prompt_generate(capsys):
    exit_code = main(["prompt", "generate", "--n", "3", "--category", "pump", "--seed", "1"])
    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert record["category"] == "pump"


def test_cli_dataset_build_accepts_and_rejects(tmp_path, capsys):
    dataset_root = tmp_path / "dataset"
    exit_code = main(
        ["dataset", "build", "--n", "5", "--model", "mock", "--category", "pump,bracket", "--seed", "0", "--root", str(dataset_root)]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    summary = json.loads(out.strip().splitlines()[-1])
    assert summary["requested"] == 5
    assert summary["accepted"] + summary["rejected"] + summary["failed"] == 5
    assert summary["failed"] == 0

    stats_exit = main(["dataset", "stats", "--root", str(dataset_root)])
    assert stats_exit == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["sample_count"] == summary["accepted"]


def test_cli_dataset_build_survives_a_mid_batch_failure(tmp_path, capsys, monkeypatch):
    from threedimen.teacher import MockTeacher, TEACHERS

    real_generate = MockTeacher.generate
    call_count = {"n": 0}

    def flaky_generate(self, prompt):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ConnectionError("simulated dropped connection")
        return real_generate(self, prompt)

    monkeypatch.setattr(MockTeacher, "generate", flaky_generate)
    monkeypatch.setitem(TEACHERS, "mock", MockTeacher)

    dataset_root = tmp_path / "dataset"
    exit_code = main(["dataset", "build", "--n", "4", "--model", "mock", "--root", str(dataset_root)])

    # A failed sample must be reported (non-zero exit), but must NOT abort
    # the rest of the batch -- all 4 prompts should still have been attempted.
    assert exit_code == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["requested"] == 4
    assert summary["failed"] == 1
    assert summary["accepted"] + summary["rejected"] + summary["failed"] == 4


def test_cli_model_train_and_generate(tmp_path, capsys):
    pytest.importorskip("torch")

    dataset_root = tmp_path / "dataset"
    main(
        [
            "dataset", "build", "--n", "6", "--model", "mock",
            "--category", "pump,bracket,valve", "--root", str(dataset_root), "--force",
        ]
    )
    capsys.readouterr()

    checkpoint = tmp_path / "model.pt"
    exit_code = main(
        [
            "model", "train", "--root", str(dataset_root), "--out", str(checkpoint),
            "--epochs", "5", "--num-points", "32", "--batch-size", "2",
            "--latent-dim", "8", "--hidden-dim", "16", "--embed-dim", "8",
        ]
    )
    assert exit_code == 0
    assert checkpoint.exists()
    train_result = json.loads(capsys.readouterr().out)
    assert train_result["num_samples"] == 6

    out_mesh = tmp_path / "generated.stl"
    exit_code = main(
        ["model", "generate", "a compact bracket with holes", "--checkpoint", str(checkpoint), "--out", str(out_mesh)]
    )
    assert exit_code == 0
    assert out_mesh.exists()
    gen_result = json.loads(capsys.readouterr().out)
    assert gen_result["summary"]["vertex_count"] > 0


def test_cli_model_train_reports_dataset_too_small(tmp_path, capsys):
    pytest.importorskip("torch")

    empty_root = tmp_path / "empty_dataset"
    empty_root.mkdir()
    exit_code = main(["model", "train", "--root", str(empty_root), "--out", str(tmp_path / "model.pt")])
    assert exit_code == 1


def test_cli_dataset_export_training_bundle_is_smaller_and_loads_back(tmp_path, capsys):
    import zipfile

    dataset_root = tmp_path / "dataset"
    main(["dataset", "build", "--n", "4", "--model", "mock", "--root", str(dataset_root), "--force"])
    capsys.readouterr()

    bundle_path = tmp_path / "bundle.zip"
    exit_code = main(["dataset", "export-training-bundle", "--root", str(dataset_root), "--out", str(bundle_path)])
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sample_count"] == 4
    assert result["skipped_missing_surface"] == []
    assert bundle_path.exists()
    # the bundle must be meaningfully smaller than the full dataset (it drops raw/processed meshes)
    assert result["bundle_size_bytes"] < result["original_dataset_size_bytes"]

    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(bundle_path) as zf:
        zf.extractall(extract_dir)
    assert (extract_dir / "manifest.json").exists()

    from threedimen.model.data import load_training_pairs

    pairs = load_training_pairs(extract_dir, num_points=16)
    assert len(pairs) == 4


def test_cli_dataset_export_training_bundle_empty_dataset_fails_cleanly(tmp_path, capsys):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    exit_code = main(
        ["dataset", "export-training-bundle", "--root", str(empty_root), "--out", str(tmp_path / "bundle.zip")]
    )
    assert exit_code == 1
