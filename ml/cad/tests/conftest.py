"""ml/cad has no pyproject.toml / installable package -- every script here
gets onto sys.path via `sys.path.insert(0, str(Path(__file__).resolve().parent))`
(see inference.py, benchmark.py, etc.). This mirrors that convention for
pytest, which otherwise has no way to resolve `from data.procedural_generator
import ...` / `from part_families import ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
