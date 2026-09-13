"""The first 3di-men model: a small text-to-shape proof of concept.

Requires the optional `train` extra (`pip install -e ".[train]"`, adds
PyTorch CPU) -- nothing else in `threedimen` depends on this subpackage, so
the base CLI/dataset/ingest tooling works without it.

See train.py's module docstring for what this model is and, more
importantly, what it is not: a plumbing validator over a genuinely small
dataset, not a claim of generative quality.
"""
