"""The ONLY file in this project that imports an LLM. Its role is strictly
bounded to natural-language decomposition: splitting a user's request into a
list of independently-buildable descriptions. It never emits geometry, tool
calls, or numeric CAD parameters -- that stays entirely with this project's
own proprietary, locally-trained program_model.py + sandbox.py, one call per
decomposed point (see parallel_builder.py).

This does NOT reintroduce an external-LLM dependency the project has stood
against throughout (see part_families.py, program_model.py's module
docstrings): the model here is a small (default: Qwen3-4B-Instruct-2507,
Q4_K_M quantized, ~2.3GB -- see MODEL_NAME/MODEL_PATH/QUANTIZATION below)
OPEN-WEIGHT model, loaded from a local .gguf file and run entirely
in-process via llama-cpp-python -- no network call, no hosted API, no
account, nothing leaves this machine. (Some comments and regression tests
elsewhere in this file/tests/test_llm_decompose.py reference the PREVIOUS
default, Qwen3-1.7B -- those are historical records of empirical findings
against that specific model, not stale configuration; left as-is.)
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assembly_plan import Relationship

# Model configuration -- env-var driven, never hard-coded, so swapping models
# (or pointing at the old one for A/B comparison during a migration) never
# needs a code change. MODEL_FORMAT is currently a documented, asserted
# constant rather than a real dispatch switch: llama-cpp-python only loads
# .gguf, so this project has nothing else to dispatch to yet -- kept as an
# explicit setting (not silently ignored) so a future non-gguf backend has
# an obvious place to plug in.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-4B-Instruct-2507")
MODEL_FORMAT = os.environ.get("MODEL_FORMAT", "gguf")
QUANTIZATION = os.environ.get("QUANTIZATION", "Q4_K_M")
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT", "16384"))  # was 4096, then 8192 -- confirmed, not theoretical,
# each time: _SYSTEM_PROMPT grew from ~2234 tokens (when 4096 was set) to 3917 tokens, leaving only 179 tokens
# of headroom, which measurably broke decomposition (a real decompose_prompt() call against a 3-part request
# took 1449s/24+min and still fell back to a single unparsed point -- the model almost certainly thrashed
# against the context ceiling the whole time). 8192 fixed that at the time, but this session's overlap-
# avoidance/self-check additions (STEP 2's new clearance bullet, STEP 4) grew the prompt again -- re-measured
# directly against this project's own real tokenizer (not estimated): 6850 tokens, leaving only 1342 tokens of
# headroom against 8192, less than the 2048-token completion budget alone needs, before the user's own prompt
# or any retrieved knowledge context is even added. 16384 restores ~9500 tokens of headroom at the prompt's
# current size (confirmed well within the loaded model's own native 262144-token training context, so this
# isn't trading against the model's real capacity) -- re-check this again if _SYSTEM_PROMPT grows substantially.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))

# GPU offload / thread count -- both OFF/unset by default (n_gpu_layers=0,
# n_threads=None -> llama.cpp's own default), exactly matching this
# project's shipped behavior to date. Confirmed this session (direct
# benchmark against this project's actual hardware, a 2015 dual-core
# 1.8GHz CPU with no GPU offload previously configured anywhere) that
# `load_tensors: offloaded 0/37 layers to GPU` was the single largest
# factor behind multi-minute decompose_prompt() calls, despite a usable
# Metal-capable GPU being present and llama_cpp already detecting it
# (llama_supports_gpu_offload() == True). Left as an explicit opt-in
# rather than flipped on by default: this is a hardware-dependent tuning
# knob (N_GPU_LAYERS=-1 offloads every layer; wrong on a machine with too
# little VRAM for the full model), not something safe to assume for every
# environment this project might run in.
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))
_N_THREADS_ENV = os.environ.get("N_THREADS")
N_THREADS = int(_N_THREADS_ENV) if _N_THREADS_ENV else None

if MODEL_FORMAT != "gguf":
    raise ValueError(f"MODEL_FORMAT={MODEL_FORMAT!r} is not supported -- only 'gguf' (via llama-cpp-python) is implemented")

_MODELS_DIR = Path(__file__).resolve().parent / "models" / "qwen"
_DEFAULT_MODEL_PATH = _MODELS_DIR / "quantized" / f"{MODEL_NAME}-{QUANTIZATION}.gguf"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))

DEFAULT_MODEL_PATH = MODEL_PATH  # kept as an alias: existing call sites/tests default-arg on this name

_SYSTEM_PROMPT = """You are an experienced mechanical design engineer decomposing a request into the separate physical components of an object, for a CAD builder. Think about the request the way an engineer reads a drawing: what are the distinct physical parts, how do they mate with and locate against each other, and where does each one actually sit in space -- not just "what shapes are mentioned."

STEP 0 -- first judge the object's overall FORM, since that decides how you describe it:
(a) A rotationally-symmetric CONTAINER or VESSEL -- a bottle, jar, cup, canister, flask, vase: has a hollow body meeting a narrower or wider opening through a smooth CURVE (a real shoulder is a curve, never a sharp step). This is NOT always "wide body, narrow neck": most bottles/jars/flasks narrow toward the opening, but some vessels (a flared vase, a funnel, a flowerpot) genuinely WIDEN toward the opening instead -- read the request's own words for which end is bigger (e.g. "wider at the top," "flares outward," "narrower at the base") rather than defaulting to the narrowing-neck shape out of habit. Use exactly ONE of these two phrasings (fill in real numbers), matching which direction the request actually describes -- these are deliberately different sentences, not the same template with numbers swapped:
    - narrowing toward the opening (the common bottle/jar/flask case): "a hollow vessel: a {D}mm diameter, {H}mm tall body smoothly narrowing to a {d}mm diameter, {h}mm tall neck over a {t}mm shoulder, {w}mm walls"
    - widening toward the opening (a flared vase, funnel, flowerpot): "a hollow vessel: a {D}mm diameter, {H}mm tall base smoothly widening to a {d}mm diameter, {h}mm tall top opening over a {t}mm shoulder, {w}mm walls"
  (the shoulder is the curved transition zone -- typically 15-30mm). Only add a SECOND entry for a cap/lid if the object actually HAS one in real life -- a bottle, jar, canister, or flask does; an open-topped vessel (a mug, cup, bowl, vase, glass, funnel, pot) does NOT, leave those as the single hollow-vessel entry alone, nothing else. If you do add a cap, it is always a plain cylinder SMALLER than the opening it covers (never the same size as the main body -- a cap the same width as the bottle itself is wrong), positioned directly on top of the opening.
(b) A simple prismatic/box-shaped object -- an enclosure, a case, a block, a tray: a box, optionally hollowed (shelled) or with features. Describe it the same dimension-first way as any box below. ANY holes/slots/pockets/patterns/rounded or chamfered edges cut into or added onto this box are part of the SAME single entry's description (e.g. "a box 100mm x 60mm x 6mm with a corner array of four holes, each 20mm from a corner"), never a separate array entry of their own -- an entry describing only a hole/feature with no base shape to belong to cannot be built (there is nothing for it to cut into).
(c) A multi-part mechanical assembly -- gears, shafts, brackets, motors, linkages: decompose into distinct mating parts, following STEP 1-3 below.

Common everyday objects (a water bottle, a mug, a pen, a phone case) must come out a REALISTIC, recognizable size, even with zero dimensions given -- if the user names a well-known object without dimensions, use typical real-world sizing for that kind of object instead of leaving it dimensionless or inventing an arbitrary small mechanical-looking number. Rough anchors: a water bottle is roughly 65mm diameter, 190mm tall body + a 25mm shoulder + a 25mm diameter, 30mm neck; a coffee mug is roughly 80mm diameter, 95mm tall; a pen is roughly 12mm diameter, 140mm long; a smartphone case is roughly 75x150x9mm. Use these as a scale reference for similar objects, not a lookup table to match exactly.

STEP 1 -- (for a multi-part mechanical assembly, case (c) above) find the distinct mechanical parts. A part is anything that is its own separate solid in the real assembly: a plate, a bracket, a shaft, a gear, a bushing, a housing, a cover, a standoff, a fastener boss. Two components that TOUCH or MATE (a gear pressed onto a shaft, a shaft running through a bearing bore in a housing, a cover bolted to a base) are STILL separate array entries -- mating parts are never fused into one piece just because they're mechanically coupled. Only features that are physically ONE continuous piece of material (a boss machined onto a plate, a rib on a bracket, a mounting flange on a housing, HOLES/SLOTS/POCKETS/hole-patterns/fillets/chamfers cut into or added onto a part) belong inside the SAME entry as the part they're part of -- NEVER give a hole/slot/pocket/pattern its own separate array entry, since a feature described alone with no base shape has nothing to cut into or sit on and cannot be built (wrong: two entries, one "a plate 100x60x6mm" and another "a corner array of four holes" -- right: one entry, "a plate 100mm x 60mm x 6mm with a corner array of four holes, each 20mm from a corner"). If the request is just one object, output one entry.

STEP 2 -- reason like an engineer about how the parts locate against each other. For a part whose position is DERIVED from another part (sits on a face of it, is coaxial with it, or mirrors it), give a "relationship" object instead of computing position_mm by hand -- the builder resolves the real coordinates itself from that OTHER part's actual built dimensions, which is exact where your own arithmetic is not. Use one of:
- {"kind": "attached_to_face", "target_id": "<id of the part it sits on>", "face": "+z"|"-z"|"+x"|"-x"|"+y"|"-y", "offset_u": 0, "offset_v": 0} -- this part's own center ends up flush against that face of the target, centered by default (nonzero offset_u/offset_v slide it sideways within that face, e.g. toward a corner). Covers: a cap/lid/dome on TOP of a vessel or box (face "+z"), a plate or panel on the FRONT/BACK/LEFT/RIGHT face (face "+y"/"-y"/"+x"/"-x"), a boss or standoff rising from any face.
- {"kind": "concentric_with", "target_id": "<id>", "axis": "x"|"y"|"z"} -- shares the target's position on the other two axes (coaxial); optionally "axis_offset": <mm> to slide along the shared axis instead of landing exactly level. Covers: an axial hole through a boss, a gear/pulley/collar sharing its shaft's axis, a shaft centered in a housing's bore.
- {"kind": "symmetric_about", "target_id": "<id>", "axis": "x"|"y"|"z"} -- mirrors the TARGET's own position across that axis's plane through the origin. Only correct when the target IS the other half of a matched pair that is ALREADY itself offset from the origin (e.g. "tab_a" built at x=25, "tab_b" symmetric_about tab_a lands at x=-25) -- covers a matched pair of standoffs/tabs/features on opposite sides. NEVER use this for a part that just needs to sit a fixed distance from a CENTRAL body that is itself at/near the origin (e.g. a wheel relative to a bicycle frame, a leg relative to a table's center) -- mirroring something that's already at the origin produces the identical position, not an offset (a confirmed real failure mode: "a wheel flush against the frame instead of beside it"). For that case, use plain position_mm with the actual offset distance instead.
Every entry needs a short, stable "id" string (e.g. "base", "boss", "hole") so later entries can reference it as target_id -- a target_id must name an id that appears earlier in the array (never itself, never something later, never a cycle).
For everything a relationship doesn't cover, fall back to plain position_mm (rough is fine, but get the RELATIONSHIP right):
- Two gears that MESH sit side by side along one axis (their shaft centers are NOT at the same point), separated by roughly their combined size -- do not attempt exact center-distance arithmetic yourself, a rough gap is corrected automatically afterward from the real module and tooth counts.
- Fastener/mounting features (bolt circles, corner-hole patterns) are described INSIDE the same entry as the part being fastened (see STEP 1) -- never their own entry or relationship.
- If you do fall back to raw position_mm for a stacked/attached part, remember every primitive is centered on its own given position by default (a box/cylinder/sphere/etc of height H spans -H/2 to +H/2 around its own center) -- "on top of a part H mm tall" is an offset of H/2, never H.
- Two parts with NO relationship connecting them must end up with REAL clearance between them, not just different-looking numbers. Only a relationship (attached_to_face, concentric_with) is allowed to put two parts' bodies into contact or overlap, and only the exact, intentional kind it describes (a shaft running inside a bore, a boss flush against a face) -- the builder trusts that overlap because IT computed the exact placement; it does not trust an overlap it finds in your own hand-picked position_mm, and will exclude one of the two parts from the finished model rather than keep them stacked on top of each other. When you hand-place two independent, unconnected parts, use each part's OWN stated size to judge the gap (half of each part's own extent along the axis you're separating them on, at minimum, same arithmetic as the stacking rule above) -- a confirmed real failure mode is two unrelated parts left at the identical position_mm, or close enough that their bounding boxes still overlap, which the builder can only read as a mistake, not a design choice, and will silently thin the model down to fix.

STEP 3 -- describe EACH part using plain, dimension-first geometry language (a box LxWxH, a cylinder of diameter/height, a gear profile of module/tooth count/face width/bore, a hole pattern, etc.) -- exactly the style shown in the example below. Do NOT dress this up with engineering jargon (no "shoulder," "GD&T," "keyway," "press fit," "shoulder bolt") -- the builder reading this description only understands plain geometric/dimensional phrasing, never trade terminology, even though YOUR reasoning about what parts exist and how they're placed should be engineer-grade. Keep the description to BARE shape + dimensions ONLY -- never append where it goes ("centered on the top face of...", "mounted on...", "positioned on top of...", "on the front face"): position_mm is the ONLY place placement belongs, and restating it inside the description text is a confirmed, real failure mode that derails the builder reading it into producing the wrong shape entirely, even when the shape itself (e.g. "a dome 20mm diameter, 15mm tall") was described correctly. Drop any MATERIAL, color, finish, or coating word entirely (steel, brass, titanium, aluminum, cast iron, plastic, anodized, polished, ...) -- real hand-written requests mention these constantly, but this builder has no concept of material anywhere in its vocabulary below, so a material word carried into the description is dead text at best and confusing noise at worst; only shape and numbers belong in a description. A metric fastener thread size (M3, M4, M5, M6, M8, M10, ...) naming a hole's size is DIFFERENT from a material/jargon word -- it is real dimensional information and must NOT be dropped, only converted to the plain clearance diameter this builder understands (standard clearance, not the thread's own diameter): M3->3.4mm, M4->4.5mm, M5->5.5mm, M6->6.6mm, M8->9mm, M10->11mm. A confirmed real failure mode is dropping the whole hole/feature because its size was given as "M6" instead of a plain mm number -- convert it instead, exactly as if "6.6mm" had been given directly.

Available operations (this is the CAD tool's real vocabulary -- every part must be describable using only these):
- primitives: a box, a cylinder, a tapered cylinder / cone (a "spin" of two radii and a height), a sphere/ball, a torus/ring-donut shape, a capsule/pill shape (a cylinder with rounded ends), an egg-like ellipsoid (a stretched sphere), a flat-sided prism (hexagonal, octagonal, etc, optionally tapered), a dome (a rounded cap), or a flat ring/washer shape (outer and inner diameter)
- an extruded 2D profile: rectangle, rounded rectangle, L-shape, or right triangle, given a height
- a lofted shape that smoothly blends between two different cross-section profiles (e.g. a rectangle at the bottom to an ellipse at the top) over a height -- for a body whose cross-section genuinely changes shape along its length, not just its size
- a constant cross-section (round, rectangular, rounded-rectangular, or L-shaped) swept along a curved path -- a straight run, a circular arc, or a helix/coil
- a round tube, solid or hollow, following a straight, arced, or helical path -- a handrail, a coiled spring shape, a curved pipe run
- a gear profile (module, tooth count, face width, bore)
- boolean union / difference / intersection of two shapes -- ONLY to combine primitives WITHIN one piece (e.g. a boss fused onto a plate), NEVER to combine two different mechanical parts from step 1 into one entry
- a drilled hole, or a hole pattern (corner array, bolt circle, or linear row)
- a fastener clearance hole with a recessed head pocket: flat-bottomed (a counterbore, for a socket-head fastener) or conical (a countersink, for a flat-head fastener) -- or a whole corner-array/bolt-circle PATTERN of these at once
- an elongated, rounded-end mounting slot (not just a round hole) -- or a row/bolt-circle of several slots at once
- a hex-shaped recessed socket (an Allen-key/hex-bit pocket)
- mirror across an axis, or mirror a shape and fuse it with itself into one symmetric part in one step
- a non-uniform scale
- a linear array (N copies along an axis, optionally each one progressively bigger or smaller) or circular array (N copies around a circle, optionally each one rotated to face outward from the center)
- a ring of grip ridges standing up from a face (a knurled texture, for a knob or handle)
- face-based tools, working on a named face of a shape (top/bottom/+x/-x/+y/-y face): extrude a face outward into a raised boss, inset a face and cut a recessed pocket into it, add a raised cylindrical boss with a hole straight through its middle (a screw boss), or shell/hollow out the whole shape leaving a wall thickness
- taper a shape along an axis with a straight/linear profile (narrower or wider at one end), or flare it with a smoothly curved profile
- twist a shape along an axis, or bend it along its length into a curve
- smooth and round off a shape's body and edges into a softer, more organic form (not a sharp fillet/chamfer on one named edge -- this rounds the whole shape)
- convert a shape into a wireframe/tube-frame skeleton that follows its own edges -- a lattice or cage look, not a solid surface
- round or flatten a single edge where two named faces meet (a rounded or chamfered/beveled edge) -- or round/chamfer ALL the edges and corners of a box-like shape at once, in one step
- a structural support rib/gusset bridging the INSIDE (concave) corner of an already-bent/L-shaped piece -- only meaningful where two faces meet at an internal, empty-space corner, not a plain box's outer edge. This is a FEATURE of that piece's own entry (STEP 1), never a separate point. A gusset requested as its own STANDALONE part instead (a separate triangular bracket plate, not a feature grafted onto another part) is a real triangular PRISM, described using the right-triangle extruded profile above -- "a triangular gusset, {base}mm x {height}mm, {thickness}mm thick" -- never as a box with a triangular cutout (a confirmed real failure mode: a true triangular prism and a box-minus-a-triangle are different solids with different mass/stiffness, not interchangeable). If the request wants gussets on BOTH sides of an assembly (e.g. "a gusset on each side"), give each its own attached_to_face entry to the SAME target, same face, with offset_u/offset_v pushed toward opposite corners (see Example 6) -- two entries sharing one part definition, never two independently-guessed, differently-shaped parts.
- a hollow vessel: a wide body CURVING smoothly (never stepping sharply) to a narrower neck on top, hollowed out to a wall thickness -- the shape a bottle/jar/cup/flask actually is (STEP 0 case (a) -- always use the exact phrasing shown there)
- a solid shape that smoothly curves from one diameter to another (no hollowing) -- a table leg, a lamp base, a bowling-pin-like shape

Example 1 -- request: "titanium corner bracket, 80mm long, with two holes per arm" (real hand-written phrasing: terse, a material word, and only SOME dimensions given). Correct reasoning: a bracket's two arms are one continuous bent piece in real life, not two mating parts (STEP 1) -- one entry, not two; "titanium" is dropped entirely (STEP 3 -- this builder has no material concept); the hole COUNT per arm is given so it's kept, but no leg width, thickness, or hole size/spacing was given, so none of those are invented -- they stay out of the description rather than being guessed at (this is different from STEP 0's water-bottle case: a bracket isn't a well-known object with an established typical size, so the "use realistic typical sizing" allowance does not apply here):
[{"id": "bracket", "description": "an L-shape bracket, 80mm long, with two holes in each leg", "position_mm": {"x": 0, "y": 0, "z": 0}}]

Example 2 -- request: "a water bottle" (a well-known object, no dimensions given -- STEP 0 case (a), use realistic typical sizing). Correct reasoning: one hollow-vessel entry for the body+shoulder+neck using the exact required phrasing, id "body"; the cap is attached_to_face "+z" of "body" instead of hand-computed z arithmetic:
[{"id": "body", "description": "a hollow vessel: a 65mm diameter, 190mm tall body smoothly narrowing to a 25mm diameter, 30mm tall neck over a 25mm shoulder, 2mm walls", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"id": "cap", "description": "a cylinder 27mm diameter, 15mm tall", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "body", "face": "+z"}}]

Example 3 -- request: "a small gearbox: two spur gears meshing on parallel shafts, module 2.5mm, gear A has 14 teeth, gear B has 22 teeth, each gear is mounted on its own 10mm diameter, 60mm long shaft" (STEP 0 case (c)). Correct reasoning: 4 separate mechanical parts (2 shafts, 2 gears); meshing gears/shafts use plain position_mm (NOT a relationship -- see STEP 2), each gear shares its own shaft's x/y position, gear B's shaft is placed a rough, reasonable gap away from gear A's along one axis (the exact spacing is corrected afterward, not your job to get precisely right):
[{"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"description": "a gear profile, module 2.5, 14 teeth", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"description": "a cylinder 10mm diameter, 60mm long", "position_mm": {"x": 40, "y": 0, "z": 0}}, {"description": "a gear profile, module 2.5, 22 teeth", "position_mm": {"x": 40, "y": 0, "z": 0}}]

Example 4 -- request: "a rectangular metal mounting plate with four corner holes, a raised cylindrical boss in the middle with a smaller axial hole through it, and a short shaft that plugs into that hole." Correct reasoning: the corner holes AND the axial hole are FEATURES of their own parts (STEP 1 -- a hole stays in the same entry as the solid it's cut into, never its own point: the boss's axial hole belongs to the boss's own description, exactly like the plate's corner holes belong to the plate's); the boss sits on the plate's own top face (attached_to_face); the shaft is a genuinely separate, physically distinct part that MATES with the boss's hole, so it gets concentric_with instead of a second attached_to_face guess:
[{"id": "plate", "description": "a box 120mm x 80mm x 8mm with a corner array of four holes, each 15mm from a corner", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"id": "boss", "description": "a cylinder 20mm diameter, 15mm tall with a 6mm diameter hole through it", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "plate", "face": "+z"}}, {"id": "shaft", "description": "a cylinder 6mm diameter, 25mm long", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "concentric_with", "target_id": "boss", "axis": "z"}}]

Example 5 -- request: "a gearbox mounting assembly: a housing with four M6 corner mounting holes, a shaft running through it on two bearing bosses, a 24-tooth gear on that shaft meshing with a 36-tooth gear on a second parallel shaft, and a mounting bracket on the housing's side face." Correct reasoning: this combines THREE patterns from above, and each one's own rule still applies independently -- do NOT let any one pattern cause another to be simplified away or dropped.
- The corner holes are the housing's OWN feature (STEP 1), not a separate entry; "M6" is converted to its 6.6mm clearance diameter (STEP 3), never dropped just because it wasn't given as a plain mm number.
- "two bearing bosses" are each a real, separate part (a boss is explicitly in this builder's vocabulary) -- dropping them because "bearing" sounds like jargon is a confirmed real failure mode; a boss attached to a face is exactly Example 4's pattern, just used on TWO opposite faces here (the two sides the shaft passes through) instead of one.
- The meshing pair is still exactly Example 3's pattern (2 shafts, 2 gears, each gear concentric_with its OWN shaft, never the other gear's shaft) -- the first shaft mates with the first boss's bore (concentric_with that boss, exactly like Example 4's shaft-mates-boss pattern), but the SECOND, parallel shaft is a confirmed real failure mode when given a relationship here too (it has no target that captures "offset but still parallel"), so it uses plain position_mm instead, exactly like Example 3's second shaft -- never reuse the first shaft's id (or its boss's id) as a second gear's concentric_with target just because a housing is also present:
[{"id": "housing", "description": "a box 120mm x 80mm x 60mm with a corner array of four 6.6mm diameter holes, each 15mm from a corner", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"id": "boss_A", "description": "a cylinder 20mm diameter, 10mm tall with a 10mm diameter hole through it", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "housing", "face": "+y"}}, {"id": "boss_B", "description": "a cylinder 20mm diameter, 10mm tall with a 10mm diameter hole through it", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "housing", "face": "-y"}}, {"id": "shaft_A", "description": "a cylinder 10mm diameter, 70mm long", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "concentric_with", "target_id": "boss_A", "axis": "y"}}, {"id": "gear_A", "description": "a gear profile, module 2.5, 24 teeth", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "concentric_with", "target_id": "shaft_A", "axis": "y"}}, {"id": "shaft_B", "description": "a cylinder 10mm diameter, 70mm long", "position_mm": {"x": 75, "y": 0, "z": 0}}, {"id": "gear_B", "description": "a gear profile, module 2.5, 36 teeth", "position_mm": {"x": 75, "y": 0, "z": 0}, "relationship": {"kind": "concentric_with", "target_id": "shaft_B", "axis": "y"}}, {"id": "bracket", "description": "an L-shape bracket, 60mm long, with two holes in each leg", "position_mm": {"x": -80, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "housing", "face": "-x"}}]

Example 6 -- request: "a bracket assembly: a base plate, a side plate standing up from one edge of the base, and a triangular gusset on each side supporting the corner where the two plates meet." Correct reasoning: the side plate attaches to the base (Example 4's pattern). Each gusset is its OWN entry using the IDENTICAL part definition (same dimensions) -- never two independently-guessed, differently-shaped parts, and a confirmed real failure mode is building a gusset as a box with a triangular cutout instead of the real triangular-prism profile above. Both gussets attach to the SAME target and face the side plate itself attaches to, with offset_u pushed toward OPPOSITE corners -- never left at offset_u=0 for both, which would stack them on top of each other instead of one on each side:
[{"id": "base", "description": "a box 60mm x 50mm x 6mm", "position_mm": {"x": 0, "y": 0, "z": 0}}, {"id": "side", "description": "a box 60mm x 60mm x 6mm", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "base", "face": "+y"}}, {"id": "gusset_left", "description": "a triangular gusset, 30mm x 30mm, 4mm thick", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "base", "face": "+y", "offset_u": -20}}, {"id": "gusset_right", "description": "a triangular gusset, 30mm x 30mm, 4mm thick", "position_mm": {"x": 0, "y": 0, "z": 0}, "relationship": {"kind": "attached_to_face", "target_id": "base", "face": "+y", "offset_u": 20}}]

Do NOT design exact numeric geometry yourself beyond what the user already gave (dimensions the user didn't specify stay unspecified in the description, not invented) -- the exceptions are relationship placement (STEP 2 -- always prefer this over hand computation), position_mm placement math for cases a relationship doesn't cover, and realistic default sizing for a well-known common object with no dimensions given at all (STEP 0), all of which ARE your job.

STEP 4 -- before answering, review your own array once as the engineer signing off on an assembly drawing, not as a list generator: every mechanically-coupled pair (meshing gears, a shaft in a bore, a cap on a body, mating plates, a gusset against the corner it supports) uses the relationship or computed spacing that actually captures how it mates, never left floating at a guessed position that happens not to crash; every pair of parts that ISN'T coupled has real clearance between them, not an accidental overlap; every part the request implies is present in the array, nothing merged away or left out; every relationship's target_id names a real id that appears earlier in the array. A decomposition that overlaps two parts, drops a part the request implied, or leaves a mating pair unconnected is a half-built assembly, not a finished one -- you are the only place in this pipeline that reasons about the request in plain language, so fix it here before answering; nothing downstream will go back and ask you again.

Respond with ONLY a JSON array in the same format as the examples above, no other text, no explanation.

/no_think"""


_CLARIFY_SYSTEM_PROMPT = """You are an experienced mechanical design engineer reviewing a build request before committing to a design, for 3di-men, a CAD builder. Given a user's request, decide whether you need more information before it can be built, and if so, ask up to 3 short, specific questions -- the kind an engineer would actually need answered before laying out the assembly, not generic follow-ups.

Only ask about things that actually block building the geometry:
- a missing dimension needed for a piece the user clearly wants built, when NO reasonable default exists for it (a custom bracket, a shaft, a gear -- there is no "typical" size to guess)
- an ambiguous count or arrangement of a repeated feature the request itself implies exists (how many holes, how many teeth, arranged how) -- never invent a feature to ask about that the request never mentioned or implied
- an ambiguous mechanical relationship between named parts (e.g. where along a shaft a gear sits, which two faces of two housings mate, how far apart two meshing gears' shafts are meant to be if tooth counts are given but spacing isn't)

Do NOT ask about dimensions for a well-known EVERYDAY object (a water bottle, a coffee mug, a pen, a phone case, a box) even if none were given -- realistic, typical real-world sizing for that kind of object is used automatically, and asking wastes the user's time on something that already has a sensible default. Only ask about such an object if the request itself signals a genuinely unusual/custom size ("a giant water bottle", "a tiny mug") without saying how unusual.

NEVER ask about anything this builder cannot represent: material, color, finish, tolerances, manufacturing process, cost, or aesthetic style. Those questions are useless here -- do not ask them.

If the request already gives enough detail to build (even if some dimensions use round/typical numbers, or the object is a well-known everyday item with a sensible default size), return an empty list. Do not ask a question just to be thorough -- most complete, concrete requests, and most well-known everyday objects, need zero questions.

Example 1 -- request: "a bracket with some holes in it". This is a custom mechanical part with no sensible default size or hole count -- missing the bracket's own size and how many holes / where. Correct output:
{"questions": ["What size should the bracket be (length x width x thickness, in mm)?", "How many holes, and roughly where should they go?"]}

Example 1b -- request: "a water bottle". A well-known everyday object with realistic typical sizing available and nothing about holes, slots, or fasteners implied anywhere in the request. Correct output:
{"questions": []}

Example 2 -- request: "a box 50mm x 40mm x 10mm with a 16mm diameter, 20mm tall cylinder centered on top". Every dimension needed to build this is already given. Correct output:
{"questions": []}

Respond with ONLY a JSON object in the form {"questions": [...]}, no other text, no explanation.

/no_think"""


@dataclass
class ActionablePoint:
    description: str
    position_mm: tuple[float, float, float]  # the resolved/explicit position; used directly when relationship is None
    # `id`/`relationship` are trailing, defaulted fields -- deliberately, so
    # every existing 2-positional-arg or description=/position_mm=-keyword
    # call site across this project (tests included) keeps working
    # unchanged. `id` is only load-bearing when `relationship` is used
    # (by ANY point) -- callers that build a flat, relationship-free list
    # (the common case) never need to set it.
    id: str = ""
    relationship: "Relationship | None" = None  # see assembly_plan.py -- when set, parallel_builder.py
                                                 # resolves the REAL position deterministically from the
                                                 # target part's own actual built dimensions, and position_mm
                                                 # above is only a placeholder (0,0,0) until that happens
    program_override: "list[dict] | None" = None  # a user-hand-edited [{"op":..., "args":{...}}, ...]
        # tool-call program (see frontend/workspace.html's Prompt-tab "Program steps" section) -- when set,
        # parallel_builder.py's _build_one skips compositional_decomposer/program_model prediction entirely
        # and runs THIS program directly, still through the exact same Sandbox.execute()/repair/
        # spec_validation/hole-count/relationship-resolution pipeline every predicted program goes through.
        # None (the default) means "predict it as usual," unchanged from before this field existed.


@dataclass
class DecomposeResult:
    points: list[ActionablePoint]
    raw_llm_output: str
    fell_back: bool  # True if the LLM's output couldn't be parsed and we fell back to one whole-prompt point
    knowledge_context: str = ""  # the retrieved-reference-notes block actually shown to the model, if any
                                  # (see knowledge/pipeline.py's retrieve_context_for_prompt) -- "" when the
                                  # knowledge subsystem is disabled, empty, or found nothing relevant. Threaded
                                  # through so a caller (server.py's /api/build) can surface what was retrieved,
                                  # per this project's "never fabricate... source references" rule -- this is
                                  # always the REAL text that was actually injected, not a re-derived guess.


@dataclass
class ClarifyResult:
    questions: list[str]
    raw_llm_output: str


_llm: Any = None  # lazy-loaded singleton -- loading a ~1.2GB model per call would defeat the point


def _get_llm(model_path: Path = DEFAULT_MODEL_PATH):
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        _llm = Llama(
            model_path=str(model_path), n_ctx=MAX_CONTEXT, verbose=False,
            n_gpu_layers=N_GPU_LAYERS, n_threads=N_THREADS,
        )
    return _llm


# Grammar-constrained decoding (llama.cpp GBNF, built here from a JSON
# Schema via LlamaGrammar.from_json_schema): forces every token the model
# emits to stay inside a syntactically-valid instance of this shape -- the
# sampler masks out any next-token that would violate it, rather than
# generating freely and hoping the result parses. This does NOT replace
# _extract_json_array/_find_schema_shaped_arrays below; it reduces how often
# they're needed to a near-zero safety net (malformed JSON becomes
# structurally impossible for the model to emit, but the fallback stays in
# place as defense in depth -- e.g. if grammar construction itself ever
# fails, or a future model/llama-cpp-python upgrade changes this behavior).
# Deliberately permissive on VALUES (description/id are free-form strings,
# numbers are unconstrained) -- the grammar's job is structural validity
# (balanced brackets, correct key names, correct types), never content
# correctness, which stays the model's own job plus this file's downstream
# semantic checks (_parse_relationship, _correct_gear_mesh_spacing, ...).
_RELATIONSHIP_KINDS = {"attached_to_face", "concentric_with", "symmetric_about"}

_POINTS_JSON_SCHEMA = json.dumps({
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "description": {"type": "string"},
            "position_mm": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                "required": ["x", "y", "z"],
            },
            "relationship": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": sorted(_RELATIONSHIP_KINDS)},
                    "target_id": {"type": "string"},
                    "face": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"]},
                    "offset_u": {"type": "number"},
                    "offset_v": {"type": "number"},
                    "axis": {"type": "string", "enum": ["x", "y", "z"]},
                    "axis_offset": {"type": "number"},
                },
                "required": ["kind", "target_id"],
            },
        },
        # "id" is required (not just description/position_mm) because every
        # worked example in _SYSTEM_PROMPT already assigns one, even to a
        # single point with no relationship -- and confirmed via real build
        # logs (2026-10-15 water-bottle/bracket failures) that when the
        # grammar left it optional, the model sometimes dropped "id" on
        # exactly the point a LATER point's relationship.target_id went on
        # to reference by name (e.g. target_id "body" with no point actually
        # carrying id "body"), which assembly_plan.validate_relationships
        # then correctly rejects as a dangling reference -- rejecting the
        # whole decomposition even though the model's intent was clear.
        # Forcing "id" to exist removes the structural option that let the
        # model skip stating the very id its own relationship goes on to use.
        "required": ["description", "position_mm", "id"],
    },
})

_CLARIFY_JSON_SCHEMA = json.dumps({
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["questions"],
})

_points_grammar: Any = None
_clarify_grammar: Any = None


def _get_points_grammar():
    global _points_grammar
    if _points_grammar is None:
        from llama_cpp import LlamaGrammar

        _points_grammar = LlamaGrammar.from_json_schema(_POINTS_JSON_SCHEMA, verbose=False)
    return _points_grammar


def _get_clarify_grammar():
    global _clarify_grammar
    if _clarify_grammar is None:
        from llama_cpp import LlamaGrammar

        _clarify_grammar = LlamaGrammar.from_json_schema(_CLARIFY_JSON_SCHEMA, verbose=False)
    return _clarify_grammar


def _looks_like_actionable_points(parsed: Any) -> bool:
    return (
        isinstance(parsed, list) and len(parsed) > 0
        and all(isinstance(item, dict) and "description" in item for item in parsed)
    )


# Confirmed necessary, not hypothetical: on a real 5-part decomposition this
# model otherwise got completely right, it emitted a stray extra quote
# immediately after two of the numeric position_mm values -- '"z": 0"}}'
# instead of '"z": 0}}' -- which breaks strict JSON parsing outright even
# though the intent is completely unambiguous. Scoped narrowly to the
# "x"/"y"/"z" KEY specifically (not a blanket digit-followed-by-quote
# pattern anywhere in the text), so it can't misfire on a legitimate string
# field that happens to end in a digit followed by its own real closing
# quote (e.g. a description like "...123").
_STRAY_QUOTE_AFTER_POSITION_NUMBER_RE = re.compile(r'("[xyz]"\s*:\s*-?\d+(?:\.\d+)?)"')


def _repair_json_text(text: str) -> str:
    return _STRAY_QUOTE_AFTER_POSITION_NUMBER_RE.sub(r"\1", text)


def _find_schema_shaped_arrays(text: str) -> list[list[dict]]:
    """Finds every well-formed, bracket-balanced [...] substring in `text`
    and keeps only the ones that look like actionable points (a list of
    dicts each with a "description" key), in order of appearance."""
    candidates: list[list[dict]] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate_text = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate_text)
                    except json.JSONDecodeError:
                        try:
                            parsed = json.loads(_repair_json_text(candidate_text))
                        except json.JSONDecodeError:
                            continue
                    if _looks_like_actionable_points(parsed):
                        candidates.append(parsed)
    return candidates


def _extract_json_array(raw: str) -> list[dict]:
    """Only looks after the model's LAST </think> tag (if any -- Qwen3-style
    reasoning-before-answer), so reasoning-time chatter (which can easily
    contain an incidental, unrelated, but still well-formed array, e.g. "the
    user wants [1, 2, 3] things") is never mistaken for the actual answer.

    Within that final-answer segment, ALL schema-shaped arrays found are
    concatenated together, not just the last one -- confirmed necessary,
    not hypothetical: this 1.7B model reliably enumerates every part
    correctly in its reasoning, but frequently formats its final answer as
    several separate single-item arrays back to back
    ("[{...}], [{...}], [{...}]]") instead of one combined array. Taking
    only the last such array silently dropped every part but the final one
    on every multi-part prompt tested. Concatenating is a safe superset for
    the well-formed case too: a model that DOES emit one single combined
    array just produces one candidate, unchanged behavior."""
    answer_segment = raw.rsplit("</think>", 1)[-1]
    candidates = _find_schema_shaped_arrays(answer_segment)
    if not candidates:
        raise ValueError("no schema-shaped JSON array found in LLM output")
    merged: list[dict] = []
    for c in candidates:
        merged.extend(c)
    return merged


def _parse_relationship(raw: Any) -> Relationship | None:
    """None on anything not cleanly shaped like a relationship -- an
    unparseable/absent relationship just means this point uses its own
    explicit position_mm instead, never a hard failure here (structural
    problems like a dangling target_id are caught later, uniformly, by
    assembly_plan.validate_relationships, which runs on the WHOLE point
    list where it can actually check cross-references)."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    target_id = raw.get("target_id")
    if kind not in _RELATIONSHIP_KINDS or not isinstance(target_id, str) or not target_id:
        return None
    try:
        return Relationship(
            kind=kind, target_id=target_id,
            face=raw.get("face"), offset_u=float(raw.get("offset_u", 0.0) or 0.0),
            offset_v=float(raw.get("offset_v", 0.0) or 0.0), standoff=float(raw.get("standoff", 0.0) or 0.0),
            axis=raw.get("axis"), axis_offset=float(raw.get("axis_offset", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        return None


def _parse_points(raw: str) -> list[ActionablePoint]:
    items = _extract_json_array(raw)
    points = []
    for i, item in enumerate(items):
        pos = item.get("position_mm") or {}
        raw_id = item.get("id")
        point_id = str(raw_id) if isinstance(raw_id, (str, int)) and str(raw_id) else f"p{i}"
        points.append(
            ActionablePoint(
                description=str(item["description"]),
                position_mm=(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))),
                id=point_id,
                relationship=_parse_relationship(item.get("relationship")),
            )
        )
    if not points:
        raise ValueError("LLM returned an empty list")
    return points


# Matches this project's own established gear-description phrasing
# ("a gear profile, module 2, 16 teeth") -- see the _SYSTEM_PROMPT example
# and data/program_dataset.py's generic-composition patterns, which use the
# same wording. Deliberately tolerant of "2mm"/"2" and "module"/"mod".
_GEAR_DESC_RE = re.compile(r"mod(?:ule)?\.?\s*(\d+(?:\.\d+)?)\s*(?:mm)?,?\s*(\d+)\s*teeth", re.IGNORECASE)


def _extract_gear_params(description: str) -> tuple[float, int] | None:
    m = _GEAR_DESC_RE.search(description)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2))


def _correct_gear_mesh_spacing(points: list[ActionablePoint]) -> list[ActionablePoint]:
    """The one piece of real arithmetic a mechanical engineer's decomposition
    needs -- a meshing gear pair's center distance, module * (teeth_a +
    teeth_b) / 2 -- confirmed EMPIRICALLY, not assumed, that this 1.7B local
    model cannot reliably compute: tested against a real prompt with known
    module/tooth counts, it consistently copied a distance straight out of
    the system prompt's own worked examples instead of recomputing for the
    request's actual numbers, even with an explicit "don't copy the
    example" instruction and multiple examples with different results.
    This is exactly the class of precise, deterministic computation this
    project keeps OUT of the LLM's hands on principle (see program_model.py
    and sandbox.py's whole design) -- so it's corrected here in plain
    Python instead of asked of the model at all; the LLM's job stays
    "recognize these are 2 separate meshing parts, roughly apart," which it
    handles fine.

    Scoped to exactly 2 gear-profile points: a single meshing pair is
    unambiguous. 3+ gears (a gear train) have ambiguous meshing topology --
    which gear meshes with which isn't recoverable from a flat points list
    -- so that case is left uncorrected rather than guessed at.

    A gear's own mounting shaft (any OTHER point sharing that gear's exact
    position -- the LLM reliably places a gear and its shaft together, see
    _SYSTEM_PROMPT's own instruction) is shifted by the same delta, so the
    "a part mounted on a shaft shares that shaft's axis" relationship the
    LLM does get right stays intact after the correction.
    """
    gear_indices = [i for i, p in enumerate(points) if _extract_gear_params(p.description) is not None]
    if len(gear_indices) != 2:
        return points

    by_id = {p.id: p for p in points if p.id}

    def _anchor(point: ActionablePoint, depth: int = 0) -> ActionablePoint | None:
        """Walks the relationship chain (concentric_with/attached_to_face)
        up to the first ancestor with relationship=None -- that ancestor's
        position_mm is the only one in the chain that ISN'T a (0,0,0)
        placeholder (per ActionablePoint.relationship's own docstring: a
        relationship-having point's position_mm is replaced wholesale by
        assembly_plan.py later, so it's never the real position). Bounded
        depth guards a malformed/cyclic target_id chain; returns None rather
        than guessing if the chain doesn't resolve cleanly."""
        if point.relationship is None:
            return point
        if depth > 8:
            return None
        target = by_id.get(point.relationship.target_id)
        if target is None:
            return None
        return _anchor(target, depth + 1)

    a, b = gear_indices
    anchor_a, anchor_b = _anchor(points[a]), _anchor(points[b])
    if anchor_a is None or anchor_b is None:
        return points  # broken chain -- don't guess
    if anchor_a is anchor_b:
        # Both gears' real position ultimately traces back to the SAME free
        # point (e.g. both chain up to a shared housing) -- there's no
        # unambiguous single point to shift to fix the distance between
        # them, so this is left uncorrected rather than guessed at, exactly
        # like the pre-existing "3+ gears" and "different modules" bail-outs
        # below.
        return points

    module_a, teeth_a = _extract_gear_params(points[a].description)
    module_b, teeth_b = _extract_gear_params(points[b].description)
    if abs(module_a - module_b) > 1e-6:
        return points  # different modules can't mesh -- something else is going on here, don't guess at it
    expected_distance = module_a * (teeth_a + teeth_b) / 2.0

    pos_a, pos_b = anchor_a.position_mm, anchor_b.position_mm
    delta = tuple(pos_b[i] - pos_a[i] for i in range(3))
    current_distance = math.sqrt(sum(c * c for c in delta))
    axis_dir = (1.0, 0.0, 0.0) if current_distance < 1e-6 else tuple(c / current_distance for c in delta)
    corrected_pos_b = tuple(pos_a[i] + axis_dir[i] * expected_distance for i in range(3))
    shift = tuple(corrected_pos_b[i] - pos_b[i] for i in range(3))

    def _same_position(p: tuple[float, float, float], q: tuple[float, float, float]) -> bool:
        return all(abs(p[i] - q[i]) < 1e-6 for i in range(3))

    # Shift anchor_b itself (and anything else sharing its exact position --
    # the plain, no-relationship case this function originally handled) by
    # the delta. Nothing else needs touching: any point whose relationship
    # chain runs through anchor_b (e.g. gear_B, if anchor_b is its own
    # shaft) gets its real position resolved FRESH from anchor_b's new
    # position later, in assembly_plan.py -- its own placeholder position_mm
    # is never read.
    corrected = []
    for p in points:
        if p is anchor_b or _same_position(p.position_mm, pos_b):
            shifted = tuple(p.position_mm[i] + shift[i] for i in range(3))
            corrected.append(ActionablePoint(description=p.description, position_mm=shifted, id=p.id, relationship=p.relationship))
        else:
            corrected.append(p)
    return corrected


def decompose_prompt(prompt: str, model_path: Path = DEFAULT_MODEL_PATH) -> DecomposeResult:
    """Never raises and never returns zero points: a malformed/unparseable
    LLM response falls back to treating the WHOLE prompt as one actionable
    point at the origin -- a typed, disclosed degradation (fell_back=True),
    not a crash and not a silently-wrong decomposition."""
    llm = _get_llm(model_path)
    try:
        grammar = _get_points_grammar()
    except Exception:
        grammar = None  # schema/grammar construction failing is not fatal -- generate unconstrained, same as before

    # Optional engineering-reference retrieval (knowledge/ subsystem). Added
    # to the USER message, deliberately NOT the system prompt -- _SYSTEM_PROMPT
    # is a fixed, already carefully budgeted block (see MAX_CONTEXT's comment
    # above: it grew to ~3900 tokens once and silently broke decomposition by
    # leaving almost no headroom). Growing it further with variable-length
    # retrieved content would risk the exact same failure mode again. The
    # user message, by contrast, naturally scales with what's actually
    # retrieved -- zero added tokens when the knowledge base is empty or
    # nothing relevant was found (see knowledge/pipeline.py's
    # retrieve_context_for_prompt, which never raises and returns "" in
    # every such case), and is capped by KnowledgeConfig.max_context_chars
    # (2000 chars by default, ~500 tokens) even when it does have something.
    try:
        from knowledge.pipeline import retrieve_context_for_prompt

        knowledge_context = retrieve_context_for_prompt(prompt)
    except Exception:
        knowledge_context = ""  # the knowledge subsystem must never break decomposition
    user_content = f"{knowledge_context}\n\nRequest: {prompt}" if knowledge_context else prompt

    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
        temperature=0.1,
        max_tokens=2048,
        # Confirmed necessary, not defensive: on a genuinely complex 5-part
        # prompt, this model got stuck in a repetitive <think> loop
        # ("So the position is... but the user might have intended... So
        # the position is...") and never reached its JSON answer within the
        # old 1536-token budget -- a real truncation, not a parsing bug.
        # /no_think in the system prompt (a documented Qwen3 convention)
        # suppresses the reasoning block that was looping in the first
        # place; repeat_penalty is defense in depth in case some reasoning
        # still slips through. `grammar` is the primary defense now (see
        # _get_points_grammar) -- it makes malformed JSON structurally
        # impossible to emit rather than just discouraged.
        repeat_penalty=1.15,
        grammar=grammar,
    )
    raw = response["choices"][0]["message"]["content"] or ""

    try:
        points = _parse_points(raw)
        points = _correct_gear_mesh_spacing(points)
        return DecomposeResult(points=points, raw_llm_output=raw, fell_back=False, knowledge_context=knowledge_context)
    except Exception:
        return DecomposeResult(
            points=[ActionablePoint(description=prompt, position_mm=(0.0, 0.0, 0.0))],
            raw_llm_output=raw, fell_back=True, knowledge_context=knowledge_context,
        )


def _find_schema_shaped_objects(text: str) -> list[dict]:
    """Brace-balanced counterpart to _find_schema_shaped_arrays: finds every
    well-formed {...} substring that looks like a clarify-stage answer (a
    dict with a "questions" key holding a list), in order of appearance."""
    candidates: list[dict] = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate_text = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate_text)
                    except json.JSONDecodeError:
                        try:
                            parsed = json.loads(_repair_json_text(candidate_text))
                        except json.JSONDecodeError:
                            continue
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                        candidates.append(parsed)
    return candidates


def ask_clarifying_questions(prompt: str, model_path: Path = DEFAULT_MODEL_PATH, max_questions: int = 3) -> ClarifyResult:
    """Never raises: any parse failure is treated the same as "no questions
    needed" (an empty list) so a clarification-stage hiccup degrades to
    building straight from the original prompt, exactly like decompose_prompt's
    own fallback -- a clarification step that could itself crash or block the
    user would defeat the point of it being a helpful, optional step."""
    llm = _get_llm(model_path)
    try:
        try:
            grammar = _get_clarify_grammar()
        except Exception:
            grammar = None
        response = llm.create_chat_completion(
            messages=[{"role": "system", "content": _CLARIFY_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            repeat_penalty=1.15,
            grammar=grammar,
        )
        raw = response["choices"][0]["message"]["content"] or ""
    except Exception:
        return ClarifyResult(questions=[], raw_llm_output="")

    answer_segment = raw.rsplit("</think>", 1)[-1]
    candidates = _find_schema_shaped_objects(answer_segment)
    if not candidates:
        return ClarifyResult(questions=[], raw_llm_output=raw)

    questions = [str(q).strip() for q in candidates[0]["questions"] if str(q).strip()]
    return ClarifyResult(questions=questions[:max_questions], raw_llm_output=raw)


def fold_answers_into_prompt(prompt: str, qa_pairs: list[tuple[str, str]]) -> str:
    """Combines the original prompt with the user's answers to the
    clarify-stage questions into one enriched prompt text -- kept as a
    plain string append rather than a new decompose_prompt code path, so
    decompose_prompt's own extraction/fallback logic never needs to know
    a clarify stage happened at all."""
    answered = [(q, a) for q, a in qa_pairs if a and a.strip()]
    if not answered:
        return prompt
    details = "\n".join(f"- {q} {a.strip()}" for q, a in answered)
    return f"{prompt}\n\nAdditional details:\n{details}"
