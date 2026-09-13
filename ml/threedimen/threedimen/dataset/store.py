"""On-disk dataset store (Part 4).

Layout:
    <root>/manifest.json            DatasetManifest -- paths only, no embedded geometry
    <root>/samples/<id>/raw<ext>    exactly what the teacher produced, untouched
    <root>/samples/<id>/processed.glb   exported SurfaceRepresentation, for quick viewing
    <root>/samples/<id>/surface.json/.npz   the SurfaceRepresentation itself (authoritative)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..geometry.ingest import IngestResult
from ..geometry.mesh_io import export_mesh
from ..teacher.base import MeshArtifact
from .schema import DatasetManifest, DatasetSample, SampleMetadata


class DatasetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def load_manifest(self) -> DatasetManifest:
        if not self.manifest_path.exists():
            return DatasetManifest()
        return DatasetManifest.from_dict(json.loads(self.manifest_path.read_text()))

    def _write_manifest(self, manifest: DatasetManifest) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest.to_dict(), indent=2))
        tmp.replace(self.manifest_path)  # atomic rename on POSIX

    def add_sample(
        self,
        artifact: MeshArtifact,
        ingest_result: IngestResult,
        *,
        category: str | None = None,
        quality: dict | None = None,
        sample_id: str | None = None,
    ) -> DatasetSample:
        if not ingest_result.validation.ok:
            raise ValueError(f"refusing to add an invalid sample: {ingest_result.validation.errors}")

        sample_id = sample_id or uuid.uuid4().hex
        sample_dir = self.root / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        raw_dest = sample_dir / f"raw{artifact.mesh_path.suffix}"
        raw_dest.write_bytes(artifact.mesh_path.read_bytes())

        processed_path = export_mesh(ingest_result.surface, sample_dir / "processed.glb")
        surface_json_path = ingest_result.surface.save(sample_dir / "surface")

        summary = ingest_result.surface.summary_dict()
        metadata = SampleMetadata(
            category=category,
            bounding_box=summary["bounding_box"],
            vertex_count=summary["vertex_count"],
            face_count=summary["face_count"],
            normalization=ingest_result.surface.metadata.get("normalization", {}),
            generation_source=artifact.source_model,
            generation_time_seconds=artifact.generation_time_seconds,
            extra={"warnings": ingest_result.warnings, "repairs_applied": ingest_result.repairs_applied},
        )
        sample = DatasetSample(
            id=sample_id,
            prompt=artifact.prompt,
            source_model=artifact.source_model,
            raw_mesh_path=str(raw_dest.relative_to(self.root)),
            processed_mesh_path=str(processed_path.relative_to(self.root)),
            surface_representation_path=str(surface_json_path.with_suffix("").relative_to(self.root)),
            metadata=metadata,
            quality=quality,
        )

        manifest = self.load_manifest()
        manifest.samples.append(sample)
        self._write_manifest(manifest)
        return sample

    def stats(self) -> dict:
        manifest = self.load_manifest()
        samples = manifest.samples
        categories: dict[str, int] = {}
        sources: dict[str, int] = {}
        accepted = rejected = 0
        for s in samples:
            key = s.metadata.category or "uncategorized"
            categories[key] = categories.get(key, 0) + 1
            sources[s.source_model] = sources.get(s.source_model, 0) + 1
            if s.quality is not None:
                if s.quality.get("valid"):
                    accepted += 1
                else:
                    rejected += 1
        return {
            "dataset_version": manifest.dataset_version,
            "sample_count": len(samples),
            "by_category": categories,
            "by_source_model": sources,
            "quality_accepted": accepted,
            "quality_rejected": rejected,
            "unscored": len(samples) - accepted - rejected,
        }

    def validate(self) -> list[str]:
        """Check every sample's referenced files actually exist on disk.
        Returns a list of problem descriptions; empty means everything's OK."""
        problems: list[str] = []
        for s in self.load_manifest().samples:
            for label, rel_path in (
                ("raw_mesh_path", s.raw_mesh_path),
                ("processed_mesh_path", s.processed_mesh_path),
            ):
                if not (self.root / rel_path).exists():
                    problems.append(f"sample {s.id}: missing {label} -> {rel_path}")
            surface_json = self.root / f"{s.surface_representation_path}.json"
            if not surface_json.exists():
                problems.append(f"sample {s.id}: missing surface representation -> {surface_json}")
        return problems
