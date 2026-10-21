"""Mitsugen -- the 3D-printing generation product, a sibling to 3di-men's
CAD product, not a mode of it. Three independent construction paths,
all orchestrated by pipeline.py (see its own docstring for routing
details):

1. radial_sweep -- the original v1 pipeline for plain bodies of
   revolution. spec.py (prompt -> structured shape spec), slice_planner.py
   (spec -> a bounded sequence of planned slices), profile.py (2D
   cross-section representation + validation), builder.py (slices -> a
   real 3D solid via ring lofting).
2. primitive_composition (Route A) -- for composable functional objects
   (a mug with a handle, a toy car, a bracket, a stool with a patterned
   set of legs, any non-radially-symmetric or multi-feature object).
   freeform.py (prompt -> a local LLM's own decomposition into
   primitives + boolean/pattern operations -- a fresh, independent spec
   stage, not a mode of spec.py), freeform_builder.py (that spec -> a
   real CSG solid via trimesh's boolean engine). This is a constructive
   APPROXIMATION, disclosed as such in every result -- never a claim of
   exact reproduction of an arbitrary object.
3. neural_generation (Route B) -- for genuinely organic/complex objects
   (animals, figurines, sculptures) that Route A's primitive vocabulary
   cannot plausibly approximate. neural_adapter.py: an OPTIONAL,
   disclosed-if-unavailable backend (TripoSR, image-to-3D, paired with a
   local text-to-image step) -- see NEURAL_MODEL_RESEARCH.md for the
   evidence behind that choice and RTX4060_NEURAL_SETUP.md for enabling
   it. Never required; a machine without it configured falls through to
   Route A with a disclosed reason, never a bare failure and never a
   silent swap.

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
