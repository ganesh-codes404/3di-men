#!/usr/bin/env python3
"""python evaluate.py --meshes <dir> [--out report.json]

Thin CLI wrapper around evaluation/geometry_eval.py (see
docs/cad-finetuning-technical-report.md Section 6). CPU-only, no GPU or
Hunyuan3D checkout required -- this is the one entry point in ml/cad that
actually runs today.
"""
from scripts.run_eval import main

if __name__ == "__main__":
    main()
