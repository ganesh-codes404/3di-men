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
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request, send_file, send_from_directory

from build_logger import log_build
from llm_decompose import ActionablePoint, ask_clarifying_questions, fold_answers_into_prompt
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


def _parse_manual_points(raw_points) -> list[ActionablePoint]:
    """Parses the playground's hand-edited/hand-added parts list -- the
    user-facing counterpart to llm_decompose's own point list, built the
    same way a person edits any structured list: rows with an empty
    description (a blank row someone added but never filled in) are
    silently skipped rather than sent to the builder or rejected outright,
    since a stray empty row is an editing artifact, not a real request."""
    points = []
    for item in raw_points or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        points.append(ActionablePoint(description=description, position_mm=_parse_position(item.get("position_mm"))))
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
    separately and then combined by translation -- exactly the same
    combine-by-placement logic /api/build already uses for LLM-decomposed
    parts, just fed by raw programs instead of program_model predictions."""
    body = request.get_json(silent=True) or {}
    entities = []
    for item in body.get("entities") or []:
        if not isinstance(item, dict):
            continue
        program = _parse_program(item.get("program"))
        if not program:
            continue
        name = str(item.get("name", "")).strip() or "part"
        entities.append((name, program, _parse_position(item.get("position_mm"))))
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


@app.route("/api/mesh/<mesh_id>")
def api_mesh(mesh_id):
    path = OUTPUT_DIR / mesh_id
    if not path.is_file() or path.suffix != ".stl":
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="model/stl", download_name=mesh_id)


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
    args = ap.parse_args()
    print(f"3di-men builder UI: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
