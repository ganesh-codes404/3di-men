class MeshLoadError(Exception):
    """A mesh file couldn't be read or didn't produce a usable single mesh."""


class MeshValidationError(Exception):
    """A mesh failed validation (see geometry/ingest.py's ValidationReport)."""
