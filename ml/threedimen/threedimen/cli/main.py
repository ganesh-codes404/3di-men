"""Part 8: the `3dimen` CLI.

    3dimen mesh inspect <file>
    3dimen mesh convert <input> <output> [--no-normalize]
    3dimen mesh validate <file>

    3dimen teacher generate "<prompt>" [--model hunyuan3d|mock] [--out DIR]

    3dimen dataset add [<mesh>] --prompt "..." [--model NAME] [--category X] [--root DIR] [--force]
    3dimen dataset build --n N [--category X,Y] [--seed S] [--model NAME] [--root DIR] [--force]
    3dimen dataset validate [--root DIR]
    3dimen dataset stats [--root DIR]
    3dimen dataset export-training-bundle --root DIR --out BUNDLE.zip
    3dimen dataset import --source cap3d|trellis --n N [--seed S] [--category X] [--root DIR] [--force]
        (needs the optional 'external' extra: pip install -e ".[external]")

    3dimen prompt generate [--n N] [--category X,Y] [--seed S]

    3dimen model train --root DIR --out CHECKPOINT [--epochs N] [--num-points N] ...
    3dimen model generate "<prompt>" --checkpoint CHECKPOINT --out FILE

(model train/generate need the optional 'train' extra: pip install -e ".[train]")
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..dataset import DatasetStore
from ..geometry.ingest import ingest_mesh
from ..geometry.mesh_io import export_mesh, load_mesh
from ..prompts import generate_prompts
from ..quality import assess_quality
from ..teacher import TEACHERS
from ..teacher.base import MeshArtifact


def _cmd_mesh_inspect(args: argparse.Namespace) -> int:
    result = ingest_mesh(args.file, strict=False)
    print(
        json.dumps(
            {
                "source": str(args.file),
                "validation": result.validation.to_dict(),
                "repairs_applied": result.repairs_applied,
                "summary": result.surface.summary_dict() if result.validation.ok else None,
            },
            indent=2,
        )
    )
    return 0 if result.validation.ok else 1


def _cmd_mesh_convert(args: argparse.Namespace) -> int:
    result = ingest_mesh(args.input, normalize_scale=not args.no_normalize, strict=True)
    out = export_mesh(result.surface, args.output)
    print(f"wrote {out}")
    return 0


def _cmd_mesh_validate(args: argparse.Namespace) -> int:
    result = ingest_mesh(args.file, strict=False)
    print(json.dumps(result.validation.to_dict(), indent=2))
    return 0 if result.validation.ok else 1


def _cmd_teacher_generate(args: argparse.Namespace) -> int:
    teacher = TEACHERS[args.model]()
    artifact = teacher.generate(args.prompt)
    out_dir = args.out or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / artifact.mesh_path.name
    dest.write_bytes(artifact.mesh_path.read_bytes())
    print(
        json.dumps(
            {
                "prompt": artifact.prompt,
                "source_model": artifact.source_model,
                "generation_time_seconds": artifact.generation_time_seconds,
                "mesh_path": str(dest),
                "metadata": artifact.metadata,
            },
            indent=2,
        )
    )
    return 0


def _cmd_dataset_add(args: argparse.Namespace) -> int:
    store = DatasetStore(args.root)
    if args.mesh is not None:
        result = ingest_mesh(args.mesh, strict=True)
        artifact = MeshArtifact(
            mesh_path=args.mesh, prompt=args.prompt, source_model="external", generation_time_seconds=0.0
        )
    else:
        if not args.model:
            print("either a MESH path or --model must be given", file=sys.stderr)
            return 2
        teacher = TEACHERS[args.model]()
        artifact = teacher.generate(args.prompt)
        result = ingest_mesh(artifact.mesh_path, strict=True)

    report = assess_quality(result.surface)
    if not report.valid and not args.force:
        print(f"rejected by quality filter: {report.reasons}", file=sys.stderr)
        return 1

    sample = store.add_sample(artifact, result, category=args.category, quality=report.to_dict())
    print(json.dumps(sample.to_dict(), indent=2))
    return 0


def _cmd_dataset_build(args: argparse.Namespace) -> int:
    """Generate --n synthetic prompts and run each one through the given
    teacher, adding everything that passes quality filtering. Unlike a plain
    shell loop over `dataset add`, a single failed generation (network
    hiccup, a dropped Kaggle tunnel, a bad mesh) is recorded and skipped --
    it does not abort the rest of the batch, which matters once a single
    real Hunyuan3D generation can take minutes and a run covers dozens of
    them."""
    categories = args.category.split(",") if args.category else None
    prompts = generate_prompts(args.n, categories=categories, seed=args.seed)

    store = DatasetStore(args.root)
    existing = store.load_manifest().samples
    duplicate_keys = {
        s.quality["duplicate_key"] for s in existing if s.quality and s.quality.get("duplicate_key")
    }

    teacher = TEACHERS[args.model]()
    accepted = rejected = failed = 0
    t0 = time.perf_counter()

    for i, p in enumerate(prompts, 1):
        sample_t0 = time.perf_counter()
        try:
            artifact = teacher.generate(p.text)
            result = ingest_mesh(artifact.mesh_path, strict=False)
            if not result.validation.ok:
                failed += 1
                print(f"[{i}/{len(prompts)}] FAILED (invalid mesh) {p.text!r} -> {result.validation.errors}", file=sys.stderr)
                continue

            report = assess_quality(result.surface, existing_duplicate_keys=duplicate_keys)
            if not report.valid and not args.force:
                rejected += 1
                elapsed = time.perf_counter() - sample_t0
                print(f"[{i}/{len(prompts)}] REJECTED ({elapsed:.1f}s) {p.text!r} -> {report.reasons}")
                continue

            store.add_sample(artifact, result, category=p.category, quality=report.to_dict())
            if report.duplicate_key:
                duplicate_keys.add(report.duplicate_key)
            accepted += 1
            elapsed = time.perf_counter() - sample_t0
            print(f"[{i}/{len(prompts)}] ACCEPTED ({elapsed:.1f}s) {p.text!r}")
        except Exception as exc:  # a bad sample must not kill the rest of a long batch
            failed += 1
            print(f"[{i}/{len(prompts)}] FAILED ({type(exc).__name__}) {p.text!r} -> {exc}", file=sys.stderr)

    summary = {
        "requested": len(prompts),
        "accepted": accepted,
        "rejected": rejected,
        "failed": failed,
        "elapsed_seconds": time.perf_counter() - t0,
        "root": str(store.root),
    }
    # Single-line JSON (unlike other commands' indented output) so a script
    # can reliably grab the summary as "the last line of stdout" even though
    # every preceding line is a human-readable per-sample progress message.
    print(json.dumps(summary))
    return 0 if failed == 0 else 1


def _cmd_model_train(args: argparse.Namespace) -> int:
    try:
        from ..model.train import TrainConfig, train
    except ImportError as exc:
        print(f"model training needs the optional extra: pip install -e '.[train]' ({exc})", file=sys.stderr)
        return 2

    config = TrainConfig(
        num_points=args.num_points,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    try:
        result = train(args.root, args.out, config, device=args.device)
    except ValueError as exc:  # e.g. dataset too small
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:  # e.g. --device cuda requested but unavailable
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


def _cmd_model_generate(args: argparse.Namespace) -> int:
    try:
        from ..model.generate import generate as model_generate
    except ImportError as exc:
        print(f"model generation needs the optional extra: pip install -e '.[train]' ({exc})", file=sys.stderr)
        return 2

    try:
        surface = model_generate(args.prompt, args.checkpoint, device=args.device)
    except RuntimeError as exc:  # e.g. --device cuda requested but unavailable
        print(str(exc), file=sys.stderr)
        return 2
    out_path = export_mesh(surface, args.out)
    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "checkpoint": str(args.checkpoint),
                "mesh_path": str(out_path),
                "summary": surface.summary_dict(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_dataset_validate(args: argparse.Namespace) -> int:
    problems = DatasetStore(args.root).validate()
    if problems:
        for p in problems:
            print(p)
        return 1
    print("OK -- all sample files present")
    return 0


def _cmd_dataset_stats(args: argparse.Namespace) -> int:
    print(json.dumps(DatasetStore(args.root).stats(), indent=2))
    return 0


def _cmd_dataset_export_training_bundle(args: argparse.Namespace) -> int:
    """Package just what training actually reads (manifest.json + each
    sample's surface.json/.npz) into one zip -- for real Hunyuan3D output,
    raw.stl + processed.glb dominate dataset size (tens of MB/sample at
    60k-225k vertices) but training never touches them, only the surface
    representation. Uploading this bundle to Kaggle instead of the full
    dataset directory can easily be a 5-10x size reduction."""
    import zipfile

    store = DatasetStore(args.root)
    manifest = store.load_manifest()
    if not manifest.samples:
        print(f"no samples found under {args.root}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest.to_dict(), indent=2))
        for sample in manifest.samples:
            for suffix in (".json", ".npz"):
                rel_path = f"{sample.surface_representation_path}{suffix}"
                src = store.root / rel_path
                if src.exists():
                    zf.write(src, arcname=rel_path)
                elif suffix == ".json":
                    skipped.append(sample.id)

    result = {
        "bundle_path": str(out_path),
        "sample_count": len(manifest.samples),
        "skipped_missing_surface": skipped,
        "bundle_size_bytes": out_path.stat().st_size,
        "original_dataset_size_bytes": sum(f.stat().st_size for f in store.root.rglob("*") if f.is_file()),
    }
    print(json.dumps(result, indent=2))
    return 0


def _cmd_dataset_import(args: argparse.Namespace) -> int:
    """Import a small subset of an existing public text-3D dataset (Cap3D or
    TRELLIS-500K's Objaverse-XL-sourced entries). Each sample goes through
    the exact same ingest/quality pipeline as a teacher-generated one -- a
    real-world mesh with a slightly odd topology is just as capable of
    failing validation as a generated one, and should be handled the same
    way (skip, don't crash the batch)."""
    try:
        if args.source == "cap3d":
            from ..external.cap3d import sample_cap3d
            samples = sample_cap3d(args.n, seed=args.seed, check_license=not args.no_license_check)
        elif args.source == "trellis":
            from ..external.trellis import sample_trellis
            samples = sample_trellis(args.n, seed=args.seed, check_license=not args.no_license_check)
        else:
            print(f"unknown source {args.source!r} -- known: cap3d, trellis", file=sys.stderr)
            return 2
    except ImportError as exc:
        print(f"needs the optional extra: pip install -e '.[external]' ({exc})", file=sys.stderr)
        return 2

    store = DatasetStore(args.root)
    accepted = rejected = failed = 0
    for i, ext in enumerate(samples, 1):
        try:
            result = ingest_mesh(ext.mesh_path, strict=False)
            if not result.validation.ok:
                failed += 1
                print(f"[{i}/{len(samples)}] FAILED (invalid mesh) {ext.uid} -> {result.validation.errors}", file=sys.stderr)
                continue

            report = assess_quality(result.surface)
            if not report.valid and not args.force:
                rejected += 1
                print(f"[{i}/{len(samples)}] REJECTED {ext.uid} -> {report.reasons}")
                continue

            artifact = MeshArtifact(
                mesh_path=ext.mesh_path, prompt=ext.prompt, source_model=ext.source,
                generation_time_seconds=0.0, metadata=ext.metadata,
            )
            store.add_sample(artifact, result, category=args.category, quality=report.to_dict())
            accepted += 1
            print(f"[{i}/{len(samples)}] ACCEPTED {ext.uid}: {ext.prompt[:70]!r}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(samples)}] FAILED ({type(exc).__name__}) {ext.uid} -> {exc}", file=sys.stderr)

    print(json.dumps({"requested": len(samples), "accepted": accepted, "rejected": rejected, "failed": failed}))
    return 0


def _cmd_prompt_generate(args: argparse.Namespace) -> int:
    categories = args.category.split(",") if args.category else None
    for p in generate_prompts(args.n, categories=categories, seed=args.seed):
        print(json.dumps({"text": p.text, "category": p.category, "attributes": p.attributes}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="3dimen")
    top = parser.add_subparsers(dest="group", required=True)

    mesh_parser = top.add_parser("mesh", help="inspect/convert/validate mesh files")
    mesh_sub = mesh_parser.add_subparsers(dest="action", required=True)

    p = mesh_sub.add_parser("inspect", help="ingest a mesh and report its validation + summary")
    p.add_argument("file", type=Path)
    p.set_defaults(func=_cmd_mesh_inspect)

    p = mesh_sub.add_parser("convert", help="ingest and re-export a mesh in another supported format")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--no-normalize", action="store_true", help="skip scale normalization")
    p.set_defaults(func=_cmd_mesh_convert)

    p = mesh_sub.add_parser("validate", help="validate only; exit non-zero on any error")
    p.add_argument("file", type=Path)
    p.set_defaults(func=_cmd_mesh_validate)

    teacher_parser = top.add_parser("teacher", help="call a teacher model directly")
    teacher_sub = teacher_parser.add_subparsers(dest="action", required=True)
    p = teacher_sub.add_parser("generate", help="generate a mesh from a prompt")
    p.add_argument("prompt")
    p.add_argument("--model", default="mock", choices=sorted(TEACHERS))
    p.add_argument("--out", type=Path, default=None, help="output directory (default: cwd)")
    p.set_defaults(func=_cmd_teacher_generate)

    dataset_parser = top.add_parser("dataset", help="manage a 3di-men dataset")
    dataset_sub = dataset_parser.add_subparsers(dest="action", required=True)

    p = dataset_sub.add_parser("add", help="generate (or ingest an existing mesh) and add a sample")
    p.add_argument("mesh", nargs="?", type=Path, default=None, help="existing mesh file; omit to generate via --model")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default=None, choices=sorted(TEACHERS))
    p.add_argument("--category", default=None)
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.add_argument("--force", action="store_true", help="add even if the quality filter rejects it")
    p.set_defaults(func=_cmd_dataset_add)

    p = dataset_sub.add_parser("build", help="generate N samples via a teacher and add everything that passes quality filtering")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--category", default=None, help="comma-separated category names; default = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="mock", choices=sorted(TEACHERS))
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.add_argument("--force", action="store_true", help="add samples even if the quality filter rejects them")
    p.set_defaults(func=_cmd_dataset_build)

    p = dataset_sub.add_parser("validate", help="check every sample's referenced files exist")
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.set_defaults(func=_cmd_dataset_validate)

    p = dataset_sub.add_parser("stats", help="summary counts by category/source/quality")
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.set_defaults(func=_cmd_dataset_stats)

    p = dataset_sub.add_parser(
        "export-training-bundle",
        help="package just manifest.json + each sample's surface.json/.npz into one zip (for uploading to Kaggle etc. without the much larger raw/processed mesh files)",
    )
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.add_argument("--out", type=Path, required=True, help="output zip path")
    p.set_defaults(func=_cmd_dataset_export_training_bundle)

    p = dataset_sub.add_parser(
        "import",
        help="import a small subset of an existing public text-3D dataset (needs pip install -e '.[external]')",
    )
    p.add_argument("--source", required=True, choices=["cap3d", "trellis"], help="which external dataset to pull from")
    p.add_argument("--n", type=int, required=True, help="how many samples to pull")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--category", default=None)
    p.add_argument("--root", type=Path, default=Path("dataset"))
    p.add_argument("--force", action="store_true", help="add even if the quality filter rejects it")
    p.add_argument("--no-license-check", action="store_true", help="skip the best-effort per-object license filter (see external/cap3d.py)")
    p.set_defaults(func=_cmd_dataset_import)

    model_parser = top.add_parser("model", help="train/run the 3di-men proof-of-concept model (needs pip install -e '.[train]')")
    model_sub = model_parser.add_subparsers(dest="action", required=True)

    p = model_sub.add_parser("train", help="train a text-to-shape model on a dataset")
    p.add_argument("--root", type=Path, required=True, help="dataset root (as produced by `dataset add`/`dataset build`)")
    p.add_argument("--out", type=Path, required=True, help="checkpoint output path (e.g. model.pt)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--num-points", type=int, default=2048, help="points sampled per mesh for both training target and output resolution")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="'auto' (default, GPU if available), 'cuda', or 'cpu'")
    p.set_defaults(func=_cmd_model_train)

    p = model_sub.add_parser("generate", help="generate a mesh from a prompt using a trained checkpoint")
    p.add_argument("prompt")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto", help="'auto' (default, GPU if available), 'cuda', or 'cpu'")
    p.set_defaults(func=_cmd_model_generate)

    prompt_parser = top.add_parser("prompt", help="synthetic prompt generation")
    prompt_sub = prompt_parser.add_subparsers(dest="action", required=True)
    p = prompt_sub.add_parser("generate", help="generate N synthetic prompts, one JSON object per line")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--category", default=None, help="comma-separated category names; default = all")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_prompt_generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
