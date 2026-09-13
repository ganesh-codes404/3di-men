"""3di-men core package: surface representation, teacher/data pipeline, dataset tooling.

Separate from ml/cad/ (a narrower CAD-parametric prototype) -- this package is
the general text-to-3D foundation described in the project's Phase 1-5
roadmap (see ../../docs/threedimen-architecture.md):

    text prompt -> teacher model (Hunyuan3D, swappable) -> raw mesh
                -> ingest/validate/normalize -> SurfaceRepresentation
                -> dataset sample (versioned, on disk)
                -> (later) 3di-men's own model, trained on that dataset

Two independent version numbers matter and are bumped separately:
  REPRESENTATION_VERSION -- the SurfaceRepresentation schema (geometry/representation.py)
  DATASET_VERSION        -- the on-disk dataset manifest schema (dataset/schema.py)
Bumping one does not imply bumping the other; a dataset format change doesn't
necessarily mean the representation itself changed, and vice versa.
"""

REPRESENTATION_VERSION = "0.1.0"
DATASET_VERSION = "0.1.0"
PIPELINE_VERSION = "0.1.0"  # the ingest/teacher/quality pipeline as a whole

__all__ = ["REPRESENTATION_VERSION", "DATASET_VERSION", "PIPELINE_VERSION"]
