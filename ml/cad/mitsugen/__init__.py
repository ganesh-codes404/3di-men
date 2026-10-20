"""Mitsugen -- the 3D-printing generation product, a sibling to 3di-men's
CAD product, not a mode of it. Two independent construction paths, both
orchestrated by pipeline.py (see its own docstring for routing details):

1. radial_sweep -- the original v1 pipeline for plain bodies of
   revolution. spec.py (prompt -> structured shape spec), slice_planner.py
   (spec -> a bounded sequence of planned slices), profile.py (2D
   cross-section representation + validation), builder.py (slices -> a
   real 3D solid via ring lofting).
2. primitive_composition -- for everything else (a mug with a handle, a
   toy car, a bracket, any non-radially-symmetric or multi-feature
   object). freeform.py (prompt -> a local LLM's own decomposition into
   primitives + boolean operations -- a fresh, independent spec stage,
   not a mode of spec.py), freeform_builder.py (that spec -> a real CSG
   solid via trimesh's boolean engine). This is a constructive
   APPROXIMATION, disclosed as such in every result -- never a claim of
   exact reproduction of an arbitrary object.

Deliberately does NOT import sandbox.py's Sandbox, program_model.py,
compositional_decomposer.py, assembly_plan.py, parallel_builder.py, or
llm_decompose.py -- 3di-men's own constraint-solving CAD engine (and its
own, differently-shaped LLM decomposition) is not Mitsugen's geometry
engine, by explicit design. freeform.py loads the same on-disk .gguf
model and the same llama-cpp-python library llm_decompose.py uses --
"LLM inference" is explicitly shared infrastructure -- but writes its
own system prompt, JSON schema, and parsing logic for a completely
different decomposition target. See pipeline.py's own docstring for
exactly which low-level utilities ARE reused beyond that, and why those
are a different thing: general-purpose trimesh/shapely primitives and
geometry-report machinery, not the CAD assembly solver itself."""
