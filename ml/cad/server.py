#!/usr/bin/env python3
"""python server.py [--port 5050]

Local web frontend for the LLM-decompose + parallel tool-composing builder
(see llm_decompose.py + parallel_builder.py). One route: POST /api/build
takes a prompt, decomposes it into actionable points via the local Qwen3
model, builds each point in parallel via 3di-men's own proprietary
program_model.py + sandbox.py, and returns the combined mesh plus a
per-point status report. No part-family concept anywhere in this path.
"""
import argparse
import io
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import trimesh
from flask import Flask, jsonify, request, send_file, send_from_directory

from build_logger import log_build
from llm_decompose import ActionablePoint, _parse_relationship, ask_clarifying_questions, fold_answers_into_prompt
from parallel_builder import build_assembly_from_points, build_assembly_from_programs, build_assembly_from_prompt
from sandbox import Sandbox, ToolCall, tool_schema

OUTPUT_DIR = Path(__file__).resolve().parent / "webapp_output"
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parent / "models" / "program_model.joblib"

app = Flask(__name__)


def _export_mesh(mesh) -> str:
    mesh_id = f"{uuid.uuid4().hex}.stl"
    mesh.export(OUTPUT_DIR / mesh_id)
    return f"/api/mesh/{mesh_id}"


@app.route("/")
def index():
    """The landing page -- no build logic here, just an entry point into the
    real tool at /workspace below."""
    return send_from_directory(Path(__file__).resolve().parent / "frontend", "index.html")


@app.route("/workspace")
def workspace():
    """The actual builder UI (prompt flow + manual tool console) that used to
    live at '/' -- moved here so '/' can be a real landing page without
    touching any of the build/LLM/sandbox routes below."""
    return send_from_directory(Path(__file__).resolve().parent / "frontend", "workspace.html")


@app.route("/api/tools")
def api_tools():
    """Reflects the exact same Sandbox tool methods program_model.py's
    predictions and the LLM-decomposed build path both go through -- the
    manual tool console (Blender-style: pick a tool, fill its real
    arguments, target a handle from an earlier step) is a third way to
    reach the identical proprietary builder, never a separate/lesser one."""
    return jsonify({"tools": tool_schema()})


@app.route("/api/clarify", methods=["POST"])
def api_clarify():
    """Runs the LLM's own optional clarify stage: up to a few targeted
    questions if the prompt is genuinely underspecified for building, or an
    empty list if it's already complete -- never blocks /api/build, since
    a caller can always skip straight there with the original prompt."""
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is empty"}), 400

    try:
        result = ask_clarifying_questions(prompt)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({"questions": result.questions})


def _parse_position(raw) -> tuple[float, float, float]:
    if isinstance(raw, dict):
        return (float(raw.get("x", 0.0) or 0.0), float(raw.get("y", 0.0) or 0.0), float(raw.get("z", 0.0) or 0.0))
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0] or 0.0), float(raw[1] or 0.0), float(raw[2] or 0.0))
    return (0.0, 0.0, 0.0)


def _parse_rotation(raw) -> tuple[float, float, float]:
    """Mirrors _parse_position exactly -- degrees about X/Y/Z, applied in
    that fixed order (see PlacementTransform.rotation()). Missing/absent
    defaults to (0,0,0), i.e. no rotation, so every existing caller that
    never sends rotation_deg at all keeps today's translation-only
    behavior unchanged."""
    if isinstance(raw, dict):
        return (float(raw.get("x", 0.0) or 0.0), float(raw.get("y", 0.0) or 0.0), float(raw.get("z", 0.0) or 0.0))
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0] or 0.0), float(raw[1] or 0.0), float(raw[2] or 0.0))
    return (0.0, 0.0, 0.0)


def _parse_point_program_override(raw_program) -> list[dict] | None:
    """Plain {"op":..., "args":{...}} dicts, deliberately NOT sandbox.ToolCall
    objects -- ActionablePoint.program_override stays a plain, JSON/pickle-
    -friendly value, consistent with every other ActionablePoint field, and
    keeps llm_decompose.py (which this type lives in) with zero dependency
    on sandbox.py's tool-call type, matching this project's own standing
    "the LLM/decomposition layer never touches CAD types directly" rule.
    None (not an empty list) when no program was sent at all -- a point
    edited to have an EMPTY program is a different, deliberate thing
    (handled further down) from "nothing was ever provided."""
    if raw_program is None:
        return None
    program = []
    for item in raw_program:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "")).strip()
        if not op:
            continue
        args = item.get("args")
        program.append({"op": op, "args": args if isinstance(args, dict) else {}})
    return program


def _parse_manual_points(raw_points) -> list[ActionablePoint]:
    """Parses the playground's hand-edited/hand-added parts list -- the
    user-facing counterpart to llm_decompose's own point list, built the
    same way a person edits any structured list: rows with an empty
    description (a blank row someone added but never filled in) are
    silently skipped rather than sent to the builder or rejected outright,
    since a stray empty row is an editing artifact, not a real request.

    Confirmed real gap this closes: this used to only read description/
    position_mm/program_override, silently dropping id/relationship even
    though ActionablePoint and build_assembly_from_points both fully
    support them -- any direct API caller sending a relationship-based
    manual points list (not just the frontend's own position-only
    playground) had it discarded with no error, leaving every point at
    its raw (often identical, e.g. all (0,0,0)) position_mm instead of a
    real resolved placement. _parse_relationship is llm_decompose.py's
    own parser for the exact same JSON shape the LLM's decomposition
    output uses -- reused here, not reimplemented, so both entry points
    accept identical relationship syntax."""
    points = []
    for item in raw_points or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        raw_id = item.get("id")
        points.append(ActionablePoint(
            description=description, position_mm=_parse_position(item.get("position_mm")),
            id=str(raw_id) if isinstance(raw_id, str) and raw_id else "",
            relationship=_parse_relationship(item.get("relationship")),
            program_override=_parse_point_program_override(item.get("program_override")),
        ))
    return points


@app.route("/api/build", methods=["POST"])
def api_build():
    body = request.get_json(silent=True) or {}

    # Two ways to reach the builder: the usual LLM-decomposed prompt flow,
    # or a manually curated list of parts from the playground's editable
    # points list -- a user adding/editing/repositioning parts by hand,
    # bypassing the LLM's decomposition guess entirely while still going
    # through the exact same proprietary program_model.py + sandbox.py
    # builder underneath (see build_assembly_from_points).
    manual_points = _parse_manual_points(body.get("points"))

    if not DEFAULT_BUNDLE_PATH.exists():
        return jsonify({"error": f"no trained model at {DEFAULT_BUNDLE_PATH} -- run scripts/train_program_model.py first"}), 500

    t0 = time.perf_counter()
    try:
        if manual_points:
            result = build_assembly_from_points(manual_points, DEFAULT_BUNDLE_PATH)
        else:
            prompt = body.get("prompt", "").strip()
            if not prompt:
                return jsonify({"error": "prompt is empty"}), 400

            # answers is an optional list of {question, answer} from the
            # clarify step -- folded into the prompt text before
            # decomposition, so decompose_prompt's own extraction/fallback
            # logic needs no changes.
            answers = body.get("answers") or []
            qa_pairs = [(a.get("question", ""), a.get("answer", "")) for a in answers if isinstance(a, dict)]
            enriched_prompt = fold_answers_into_prompt(prompt, qa_pairs)
            result = build_assembly_from_prompt(enriched_prompt, DEFAULT_BUNDLE_PATH)
    except Exception as exc:  # surfaced to the UI rather than a bare 500 page
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    elapsed = time.perf_counter() - t0

    mesh_url = None
    assembly_topology = None
    if result.mesh is not None:
        mesh_url = _export_mesh(result.mesh)

        from mesh_quality import mesh_topology_stats

        assembly_topology = mesh_topology_stats(result.mesh)

    points_response = [
        {
            "description": p.point.description,
            "position_mm": list(p.point.position_mm),
            "status": p.status,
            "in_distribution": p.status != "out_of_distribution",
            "nearest_neighbor_similarity": p.nearest_neighbor_similarity,
            "program": p.program,
            "error": p.error,
            "topology": p.topology,
            # Correctness-to-spec check (spec_validation.py) -- distinct from
            # "status": a point can be status="ok" (structurally watertight)
            # while dimension_match_rate < 1.0 (the built geometry's actual
            # size doesn't match what the description itself claimed). None
            # means the description had no numeric "NNmm" claims to check,
            # not a confirmed match -- see ACCURACY_AUDIT.md section 3.
            "dimension_match_rate": p.dimension_match_rate,
            "unmatched_dimensions_mm": p.unmatched_dimensions_mm,
            # Section 8's bounded repair loop (repair.py) -- True only if a
            # repair attempt actually improved the result; every attempt
            # (improved or not) is listed in repair_attempts, never hidden.
            "repair_applied": p.repair_applied,
            "repair_attempts": p.repair_attempts,
            # True only when the repair that fixed this point actually dropped a
            # step (repair.py's TRUNCATE_TO_EARLIER_STEP), not just recomputed an
            # equivalent commutative boolean -- see
            # AssemblyBuildResult.overall_status's own docstring for why this (not
            # repair_applied alone) is what keeps a status="ok" point from hiding
            # a dropped feature behind a clean overall "PASS".
            "repair_dropped_a_step": p.repair_dropped_a_step,
            # "compositional" (compositional_decomposer.py's deterministic
            # clause-by-clause resolution) or "statistical" (program_model.py's
            # TF-IDF/classifier guess) -- disclosed so a caller can tell a
            # literally-parsed program apart from a guessed one.
            "resolution_method": p.resolution_method,
            # Feature validation (spec_validation.py's hole-count check) --
            # GeometryReport.detected_hole_count was already computed by
            # Sandbox.execute() on every build but discarded by every
            # caller before this; claimed_hole_count is None when the
            # description made no hole claim at all.
            "claimed_hole_count": p.claimed_hole_count,
            "detected_hole_count": p.detected_hole_count,
            # Feature validation (spec_validation.py's hole/bore DIAMETER
            # check) -- any claimed diameter (e.g. "a 12mm bore") that no
            # detected hole's own measured diameter matches within
            # tolerance. Empty when every claim matched or there was
            # nothing to check.
            "unmatched_hole_diameters_mm": p.unmatched_hole_diameters_mm,
            # Non-negotiable principle: explicit dimensions override a
            # semantic name (e.g. "NEMA 17") -- disclosure only, the
            # explicit value is already what's built and validated above.
            "semantic_size_conflicts": p.semantic_size_conflicts,
            # True only when this point built a valid mesh on its own but was
            # then left out of mesh_url's combined mesh to resolve a collision
            # with another, unrelated part -- see
            # parallel_builder._find_and_resolve_collisions. The point's own
            # program/topology/mesh above are unaffected; only the combined
            # assembly excludes it.
            "excluded_from_assembly": p.excluded_from_assembly,
        }
        for p in result.points
    ]

    # see build_logger.py's module docstring -- this is the ONLY record of
    # what actually happened for a given request (prompt, the LLM's raw
    # output, and crucially each point's PREDICTED PROGRAM, not just
    # whether it executed "ok") that survives past this response.
    log_build(
        prompt=body.get("prompt", "") or "(manual points)",
        raw_llm_output=getattr(result, "raw_llm_output", None),
        fell_back=result.fell_back_to_single_point,
        points=points_response,
        elapsed_seconds=elapsed,
    )

    return jsonify({
        "mesh_url": mesh_url,
        "fell_back_to_single_point": result.fell_back_to_single_point,
        "assembly_topology": assembly_topology,
        "points": points_response,
        "generation_time_seconds": elapsed,
        # The real, actually-injected engineering-reference text for this
        # build, if the knowledge/ subsystem retrieved anything relevant --
        # "" (never fabricated) when it found nothing or is disabled. See
        # knowledge/pipeline.py's retrieve_context_for_prompt.
        "knowledge_context": getattr(result, "knowledge_context", "") or "",
        # Confirmed real gap this closes: no assembly-level check anywhere
        # previously caught two independently-placed parts ending up
        # stacked on/inside each other -- see parallel_builder.py's
        # _find_and_resolve_collisions/mesh_quality.bounding_box_overlap_fraction.
        # Each entry here describes a collision ALREADY resolved (one part
        # excluded from mesh_url's combined mesh -- see each point's own
        # excluded_from_assembly/status in `points` below), not a pending
        # failure. Always a list, empty when nothing was flagged.
        "collision_warnings": getattr(result, "collision_warnings", []),
        # Confirmed real gap this closes (mission-brief regression: a
        # 251.2x124.0x108.6mm assembly against an approximate requested
        # 150x100mm footprint, with no way to tell which component caused
        # it) -- see parallel_builder.py's _diagnose_envelope_overrun.
        # Empty whenever the prompt made no explicit footprint claim at
        # all, OR the combined assembly stayed within tolerance of one.
        "envelope_warnings": getattr(result, "envelope_warnings", []),
        # The single aggregate PASS/FAIL/INCOMPLETE this pipeline previously
        # never computed anywhere -- see AssemblyBuildResult.overall_status's
        # own docstring. A caller that only checks mesh_url/assembly_topology
        # (which can look perfectly clean even when a required part failed)
        # should check this field first.
        "overall_status": result.overall_status,
    })


def _parse_program(raw_program) -> list[ToolCall]:
    program = []
    for item in raw_program or []:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op", "")).strip()
        if not op:
            continue
        args = item.get("args")
        program.append(ToolCall(op=op, args=args if isinstance(args, dict) else {}))
    return program


@app.route("/api/execute_program", methods=["POST"])
def api_execute_program():
    """The manual tool console's build step: runs a hand-authored list of
    raw Sandbox tool calls directly -- no LLM, no program_model prediction,
    just the same Sandbox.execute() every other path in this project already
    trusts. Sandbox.validate() (called first, inside execute()) rejects any
    op name outside the fixed CREATES/CONSUMES_* sets before any geometry
    runs, so a malformed or malicious op/arg name is a typed "rejected"
    result, never an arbitrary method call."""
    body = request.get_json(silent=True) or {}
    program = _parse_program(body.get("program"))
    if not program:
        return jsonify({"error": "program is empty"}), 400

    try:
        result = Sandbox.execute(program)
    except Exception as exc:  # a genuinely unexpected failure, not a rejected/partial program
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    mesh_url = None
    topology = None
    if result.mesh is not None:
        mesh_url = _export_mesh(result.mesh)

        from mesh_quality import mesh_topology_stats

        topology = mesh_topology_stats(result.mesh)

    return jsonify({
        "status": result.status,
        "executed_steps": result.executed_steps,
        "failing_step": result.failing_step,
        "error": result.error,
        "mesh_url": mesh_url,
        "topology": topology,
    })


@app.route("/api/execute_programs", methods=["POST"])
def api_execute_programs():
    """The console's answer to "the gear and the shaft are separate
    entities, not one fused blob": each entity in `entities` is its own
    independent hand-authored program (own steps, own finalize), built
    separately and then combined by translation (and, now, optional
    rotation) -- exactly the same combine-by-placement logic /api/build
    already uses for LLM-decomposed parts, just fed by raw programs instead
    of program_model predictions. `rotation_deg` is optional per entity and
    defaults to no rotation (see _parse_rotation)."""
    body = request.get_json(silent=True) or {}
    entities = []
    for item in body.get("entities") or []:
        if not isinstance(item, dict):
            continue
        program = _parse_program(item.get("program"))
        if not program:
            continue
        name = str(item.get("name", "")).strip() or "part"
        entities.append((name, program, _parse_position(item.get("position_mm")), _parse_rotation(item.get("rotation_deg"))))
    if not entities:
        return jsonify({"error": "no entities with a program"}), 400

    try:
        result = build_assembly_from_programs(entities)
    except Exception as exc:  # surfaced to the UI rather than a bare 500 page
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    from mesh_quality import mesh_topology_stats

    assembly_mesh_url = None
    assembly_topology = None
    if result.mesh is not None:
        assembly_mesh_url = _export_mesh(result.mesh)
        assembly_topology = mesh_topology_stats(result.mesh)

    return jsonify({
        "assembly_mesh_url": assembly_mesh_url,
        "assembly_topology": assembly_topology,
        "entities": [
            {
                "name": e.name,
                "status": e.status,
                "executed_steps": e.executed_steps,
                "error": e.error,
                "mesh_url": _export_mesh(e.mesh) if e.mesh is not None else None,
                "topology": e.topology,
            }
            for e in result.entities
        ],
    })


# 3MF confirmed directly (not assumed) to actually work with this project's
# installed trimesh -- both export AND reimport, see
# tests/test_server.py::test_exported_mesh_round_trips_through_3mf. Print
# Lab's own export section (printability.py's module docstring) previously
# flagged 3MF as "unverified on this stack" -- it IS supported, so it's
# offered here rather than silently left out.
_MESH_EXPORT_FORMATS = {"stl": "model/stl", "glb": "model/gltf-binary", "obj": "model/obj", "3mf": "model/3mf"}


@app.route("/api/mesh/<mesh_id>")
def api_mesh(mesh_id):
    path = OUTPUT_DIR / mesh_id
    if not path.is_file() or path.suffix != ".stl":
        return jsonify({"error": "not found"}), 404

    # Every build (including every live preview step in the Tool Console,
    # not just a final download) calls _export_mesh() -- eagerly exporting
    # every format on every one of those would triple disk writes for
    # formats nobody asked for yet, and OUTPUT_DIR is only ever cleaned on
    # server restart (see main()). So the on-disk file stays STL-only
    # (all the viewer itself ever needs, via STLLoader) and GLB/OBJ are
    # produced lazily, in memory, only when actually requested here.
    fmt = request.args.get("format", "stl").lower()
    if fmt not in _MESH_EXPORT_FORMATS:
        return jsonify({"error": f"unsupported format {fmt!r}, expected one of {sorted(_MESH_EXPORT_FORMATS)}"}), 400
    if fmt == "stl":
        return send_file(path, mimetype="model/stl", download_name=mesh_id)

    mesh = trimesh.load_mesh(path)
    exported = mesh.export(file_type=fmt)
    # OBJ is a text format (trimesh returns str); GLB is binary (bytes).
    data = exported.encode("utf-8") if isinstance(exported, str) else exported
    buf = io.BytesIO(data)
    buf.seek(0)
    download_name = f"{Path(mesh_id).stem}.{fmt}"
    return send_file(buf, mimetype=_MESH_EXPORT_FORMATS[fmt], download_name=download_name)


@app.route("/print-lab")
def print_lab():
    """The 3D-printing analysis workspace (see printability.py) -- a
    separate page from /workspace, same convention as /workspace itself
    being separate from the landing page at /. Reachable from
    /workspace's own topbar; carries a model over via ?mesh=<mesh_id>
    (see /workspace's own "Open in Print Lab" link) rather than
    requiring a re-upload of something already built in this session."""
    return send_from_directory(Path(__file__).resolve().parent / "frontend", "print_lab.html")


# Deliberately small, named profiles rather than free-form build-volume
# input alone -- a user picks a REAL printer's own configured volume
# (printability.DEFAULT_PRINTER_PROFILES), never a silently-wrong default
# presented as if it were theirs. "custom" accepts an arbitrary volume for
# a printer not in the list.
@app.route("/api/print/printers")
def api_print_printers():
    from printability import DEFAULT_PRINTER_PROFILES

    return jsonify({
        "printers": [
            {"id": pid, "name": p.name, "build_volume_mm": list(p.build_volume_mm), "nozzle_diameter_mm": p.nozzle_diameter_mm}
            for pid, p in DEFAULT_PRINTER_PROFILES.items()
        ],
    })


_UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50MB -- generous for a single-part STL/OBJ, not unbounded
_UPLOAD_ALLOWED_SUFFIXES = {".stl", ".obj"}
_UPLOAD_MAX_FACES = 2_000_000  # a geometry-complexity ceiling independent of raw file size (a
    # pathologically dense small file could still be slow/memory-heavy to analyze)


@app.route("/api/print/upload", methods=["POST"])
def api_print_upload():
    """Imports a mesh file (STL/OBJ) for Print Lab analysis -- the
    "imported meshes" half of Part B's own scope, alongside models
    already built in 3di-men (which reach Print Lab via their existing
    mesh_id, no upload needed at all). Validates size/type/complexity
    BEFORE attempting to parse -- a malformed or oversized file must
    fail cleanly, never hang or crash the server."""
    if "file" not in request.files:
        return jsonify({"error": "no file provided (expected multipart form field 'file')"}), 400
    f = request.files["file"]
    suffix = Path(f.filename or "").suffix.lower()
    if suffix not in _UPLOAD_ALLOWED_SUFFIXES:
        return jsonify({"error": f"unsupported file type {suffix!r}, expected one of {sorted(_UPLOAD_ALLOWED_SUFFIXES)}"}), 400

    data = f.read(_UPLOAD_MAX_BYTES + 1)
    if len(data) > _UPLOAD_MAX_BYTES:
        return jsonify({"error": f"file exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)}MB upload limit"}), 400
    if len(data) == 0:
        return jsonify({"error": "uploaded file is empty"}), 400

    try:
        mesh = trimesh.load_mesh(io.BytesIO(data), file_type=suffix.lstrip("."))
    except Exception as exc:
        return jsonify({"error": f"could not parse this file as a mesh: {exc}"}), 400
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        return jsonify({"error": "parsed file contains no usable geometry"}), 400
    if len(mesh.faces) > _UPLOAD_MAX_FACES:
        return jsonify({"error": f"mesh has {len(mesh.faces):,} faces, exceeding the {_UPLOAD_MAX_FACES:,} analysis limit"}), 400

    mesh_url = _export_mesh(mesh)
    return jsonify({"mesh_url": mesh_url, "vertex_count": len(mesh.vertices), "face_count": len(mesh.faces)})


def _load_mesh_for_print_analysis(mesh_id: str):
    path = OUTPUT_DIR / mesh_id
    if not path.is_file() or path.suffix != ".stl":
        return None
    return trimesh.load_mesh(path)


@app.route("/api/print/analyze", methods=["POST"])
def api_print_analyze():
    """The one entry point the Print Lab UI calls -- see
    printability.analyze_printability for exactly what's measured vs.
    disclosed as unavailable. `rotation_deg` (optional, [x,y,z] degrees,
    same X-then-Y-then-Z convention as parallel_builder.PlacementTransform.rotation)
    lets the UI preview a candidate orientation's own numbers without a
    second upload."""
    from printability import DEFAULT_PRINTER_PROFILES, PrinterProfile, analyze_printability

    body = request.get_json(silent=True) or {}
    mesh_id = body.get("mesh_id")
    if not mesh_id:
        return jsonify({"error": "mesh_id is required"}), 400
    mesh = _load_mesh_for_print_analysis(mesh_id)
    if mesh is None:
        return jsonify({"error": f"no such mesh {mesh_id!r}"}), 404

    printer_id = body.get("printer_id", "generic_small")
    if printer_id == "custom":
        try:
            volume = tuple(float(v) for v in body["build_volume_mm"])
            assert len(volume) == 3
        except Exception:
            return jsonify({"error": "custom printer requires build_volume_mm: [x, y, z]"}), 400
        printer = PrinterProfile(name="Custom", build_volume_mm=volume)
    elif printer_id in DEFAULT_PRINTER_PROFILES:
        printer = DEFAULT_PRINTER_PROFILES[printer_id]
    else:
        return jsonify({"error": f"unknown printer_id {printer_id!r}, expected one of {sorted(DEFAULT_PRINTER_PROFILES)} or 'custom'"}), 400

    rotation_deg = body.get("rotation_deg")
    if rotation_deg:
        rx, ry, rz = (float(v) for v in rotation_deg)
        rot = (
            trimesh.transformations.rotation_matrix(np.radians(rz), (0, 0, 1))
            @ trimesh.transformations.rotation_matrix(np.radians(ry), (0, 1, 0))
            @ trimesh.transformations.rotation_matrix(np.radians(rx), (1, 0, 0))
        )
        mesh = mesh.copy()
        mesh.apply_transform(rot)

    report = analyze_printability(
        mesh, printer,
        overhang_angle_deg=float(body.get("overhang_angle_deg", 45.0)),
        density_g_per_cm3=body.get("density_g_per_cm3"),
    )
    return jsonify(report.as_dict())


@app.route("/api/print/orientations", methods=["POST"])
def api_print_orientations():
    from printability import DEFAULT_PRINTER_PROFILES, PrinterProfile, suggest_orientations

    body = request.get_json(silent=True) or {}
    mesh_id = body.get("mesh_id")
    if not mesh_id:
        return jsonify({"error": "mesh_id is required"}), 400
    mesh = _load_mesh_for_print_analysis(mesh_id)
    if mesh is None:
        return jsonify({"error": f"no such mesh {mesh_id!r}"}), 404

    printer_id = body.get("printer_id", "generic_small")
    printer = DEFAULT_PRINTER_PROFILES.get(printer_id)
    if printer is None:
        return jsonify({"error": f"unknown printer_id {printer_id!r}"}), 400

    candidates = suggest_orientations(mesh, printer, overhang_angle_deg=float(body.get("overhang_angle_deg", 45.0)))
    return jsonify({"orientations": [
        {
            "label": c.label, "rotation_deg": list(c.rotation_deg), "fits_build_volume": c.fits_build_volume,
            "overhang_area_mm2": c.overhang_area_mm2, "contact_area_mm2": c.contact_area_mm2,
            "height_mm": c.height_mm, "notes": c.notes,
        }
        for c in candidates
    ]})


@app.route("/api/print/slicers")
def api_print_slicers():
    from printability import slicer_availability

    return jsonify(slicer_availability())


# ---------------- Mitsugen: slice-based 3D-printing generation ----------------
# A DISTINCT product/pipeline from everything above -- see mitsugen/__init__.py
# for why it deliberately never imports sandbox.Sandbox, program_model.py,
# compositional_decomposer.py, assembly_plan.py, parallel_builder.py, or
# llm_decompose.py. Shares only general-purpose infrastructure: this same
# Flask app, printability.py (reused inside mitsugen/pipeline.py's own Stage
# F), the mesh export route below (_export_mesh/_MESH_EXPORT_FORMATS are
# format-agnostic, not CAD-specific), and -- for prompts that route to
# mitsugen/freeform.py's own primitive-composition path -- the same
# llama-cpp-python library and on-disk .gguf model llm_decompose.py uses,
# via a second, independent Llama instance Mitsugen loads itself.

@app.route("/mitsugen")
def mitsugen_page():
    """Mitsugen -- 3D Printing. A separate workspace page from /workspace
    (3di-men -- CAD), same convention as /print-lab being its own page."""
    return send_from_directory(Path(__file__).resolve().parent / "frontend", "mitsugen.html")


@app.route("/api/mitsugen/generate", methods=["POST"])
def api_mitsugen_generate():
    from mitsugen.pipeline import generate_mitsugen_model

    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        max_slices = int(body.get("max_slices", 12))
    except (TypeError, ValueError):
        return jsonify({"error": "max_slices must be a positive integer"}), 400
    if max_slices < 1:
        return jsonify({"error": "max_slices must be a positive integer"}), 400

    try:
        result = generate_mitsugen_model(
            prompt, max_slices=max_slices,
            tolerance_fraction=float(body.get("tolerance_fraction", 0.05)),
            hollow=bool(body.get("hollow", False)),
            wall_thickness_mm=body.get("wall_thickness_mm"),
            printer_build_volume_mm=tuple(body["printer_build_volume_mm"]) if body.get("printer_build_volume_mm") else None,
        )
    except Exception as exc:
        # Every stage below is documented as "never raises" -- this is
        # defense in depth, not the expected path. Without it, an
        # exception here falls through to Flask's own HTML error page,
        # which the frontend's res.json() then fails to parse as JSON
        # (a real, reported failure -- "Unexpected token '<'") instead
        # of surfacing as a readable error.
        return jsonify({"error": f"generation failed unexpectedly: {exc}"}), 500

    mesh_url = _export_mesh(result.build.mesh) if (result.build and result.build.mesh is not None) else None
    return jsonify({
        "overall_status": result.overall_status,
        "error": result.error,
        "mesh_url": mesh_url,
        "construction_mode": result.construction_mode,
        "spec": {
            "object_description": result.spec.object_description,
            "height_mm": result.spec.height_mm,
            "waypoints_mm": result.spec.waypoints_mm,
            "explicit_mm": result.spec.explicit_mm,
            "inferred_defaults_mm": result.spec.inferred_defaults_mm,
            "max_slices": result.spec.max_slices,
            "unsupported_reason": result.spec.unsupported_reason,
        } if result.spec else None,
        "plan": {
            "actual_slice_count": result.plan.actual_slice_count,
            "requested_max_slices": result.plan.requested_max_slices,
            "approximation_error_fraction": result.plan.approximation_error_fraction,
            "within_tolerance": result.plan.within_tolerance,
            "limitation": result.plan.limitation,
            "slices": [
                {"slice_id": s.slice_id, "order": s.order, "height_fraction": s.height_fraction,
                 "position_mm": s.position_mm, "radius_mm": s.radius_mm}
                for s in result.plan.slices
            ],
        } if result.plan else None,
        "freeform": {
            "object_description": result.freeform_spec.object_description,
            "fell_back": result.freeform_spec.fell_back,
            # Only sent when the fallback actually fired -- this is the
            # single most actionable field for diagnosing WHY (e.g. a
            # model-load failure reads literally "<LLM call failed:
            # ...>" with the real exception message, see
            # mitsugen/freeform.py's own decompose_freeform_prompt).
            # Confirmed real need for this: a live-reported "generates
            # cubes for nearly every prompt" turned out to be the model
            # failing to load on a machine with a different configured
            # model path, invisible from the UI with no way to tell
            # "the model is wrong" apart from "the model tried and gave
            # up" without this.
            "fallback_reason": result.freeform_spec.raw_llm_output if result.freeform_spec.fell_back else None,
            "dropped_primitives": result.freeform_spec.dropped_primitives,
            "auto_unioned_ids": result.freeform_spec.auto_unioned_ids,
            "primitives": [
                {"id": p.id, "kind": p.kind, "position_mm": p.position_mm, "rotation_deg": p.rotation_deg,
                 "size_mm": p.size_mm, "radius_mm": p.radius_mm, "height_mm": p.height_mm,
                 "major_radius_mm": p.major_radius_mm, "minor_radius_mm": p.minor_radius_mm}
                for p in result.freeform_spec.primitives
            ],
            "operations": [{"op": s.op, "target_id": s.target_id} for s in result.freeform_spec.operations],
        } if result.freeform_spec else None,
        "hollow_applied": result.build.hollow_applied if result.build else False,
        "findings": [
            {"requirement": f.requirement, "expected": f.expected, "measured": f.measured, "passed": f.passed, "method": f.method}
            for f in result.findings
        ],
        "printability": result.printability,
    })


def main():
    # Every /api/build call writes a fresh uuid-named .stl that nothing
    # ever deletes -- fine for one session, not across many restarts of a
    # dev server. Cleared here (inside `if __name__ == "__main__"`, not at
    # module level) so it runs exactly once, in the real server process --
    # not on every /api/build's ProcessPoolExecutor worker too. On macOS,
    # ProcessPoolExecutor's default 'spawn' start method re-executes this
    # entire __main__ script in each freshly spawned worker process (a
    # standard, documented multiprocessing behavior, not a bug in the
    # pool); module-level cleanup code used to run again in every one of
    # those workers, racing the main process and its sibling workers over
    # the same files -- confirmed as the actual cause of a real
    # "BrokenProcessPool" crash (a worker's FileNotFoundError on a file
    # another process had already deleted, during that worker's own
    # bootstrap, before it ever started real work).
    for _stale in OUTPUT_DIR.glob("*.stl"):
        _stale.unlink()

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5050)
    # Defaults to loopback-only, unchanged from before -- binding wider is
    # an explicit opt-in (e.g. --host 100.x.y.z for a specific Tailscale
    # IP, or --host 0.0.0.0 for every interface) since this is a Flask
    # DEV server with no authentication: anything that can reach the bound
    # address can trigger a build or read exported meshes. A Tailscale IP
    # is deliberately narrower than 0.0.0.0 -- only reachable from devices
    # on your own private tailnet, not the local Wi-Fi/LAN too.
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: binding to {args.host} -- reachable from any device that can route to this "
              "address. This is a Flask development server with no authentication.")
    print(f"3di-men builder UI: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
