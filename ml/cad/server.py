#!/usr/bin/env python3
"""python server.py [--port 5050]

Tiny local web frontend for the prompt -> mesh pipeline. Type a prompt in the
browser, it calls the exact same generate_from_prompt() that inference.py's
CLI uses, and shows the resulting STL in an interactive 3D viewer.

No new generation logic here -- this is only a thin HTTP wrapper around
inference.py, so anything true of the CLI (two part families, regex parsing,
CPU-only, milliseconds per generation) is true here too. See README.md.

/api/plan and /api/generate_assembly are the exception: a thin wrapper
around nlp/clarify.py and assembly.py instead, for the frontend's "ask
clarifying questions before building" flow on multi-component prompts.
/api/generate stays untouched for the CLI and the remote-Kaggle backend
toggle, which is a different model/API entirely.
"""
import argparse
import dataclasses
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request, send_file, send_from_directory

from assembly import AssemblySpec, ComponentSpec, build_assembly, validate_assembly
from inference import generate_from_prompt
from nlp.clarify import build_plan
from part_families import get_part_family, list_part_families

OUTPUT_DIR = Path(__file__).resolve().parent / "webapp_output"
OUTPUT_DIR.mkdir(exist_ok=True)
# Every /api/generate and /api/generate_assembly call writes a fresh
# uuid-named .stl that nothing ever deletes -- fine for one session, not
# across many restarts of a dev server. Clearing on startup (not on every
# request) keeps disk usage bounded without touching a mesh a client might
# still have an open download link to mid-session.
for _stale in OUTPUT_DIR.glob("*.stl"):
    _stale.unlink()

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(Path(__file__).resolve().parent / "frontend", "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    prompt = (request.get_json(silent=True) or {}).get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is empty"}), 400

    try:
        mesh, info = generate_from_prompt(prompt)
    except Exception as exc:  # surfaced to the UI rather than a bare 500 page
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    mesh_id = f"{uuid.uuid4().hex}.stl"
    mesh.export(OUTPUT_DIR / mesh_id)
    info["mesh_url"] = f"/api/mesh/{mesh_id}"
    return jsonify(info)


@app.route("/api/plan", methods=["POST"])
def api_plan():
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is empty"}), 400
    family_overrides = body.get("family_overrides") or {}

    known_families = set(list_part_families())
    bad_overrides = {k: v for k, v in family_overrides.items() if v not in known_families}
    if bad_overrides:
        return jsonify({"error": f"unknown family in family_overrides: {bad_overrides} -- known families: {sorted(known_families)}"}), 400

    try:
        plan = build_plan(prompt, family_overrides=family_overrides)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({
        "components": [dataclasses.asdict(c) for c in plan.components],
        "questions": [dataclasses.asdict(q) for q in plan.questions],
        "needs_clarification": plan.needs_clarification,
    })


@app.route("/api/generate_assembly", methods=["POST"])
def api_generate_assembly():
    body = request.get_json(silent=True) or {}
    raw_components = body.get("components") or []
    if not raw_components:
        return jsonify({"error": "no components"}), 400

    known_families = set(list_part_families())
    for c in raw_components:
        missing = {"id", "family", "params"} - c.keys()
        if missing:
            return jsonify({"error": f"component missing required field(s): {sorted(missing)}"}), 400
        if c["family"] not in known_families:
            return jsonify({"error": f"unknown family {c['family']!r} -- known families: {sorted(known_families)}"}), 400

    t0 = time.perf_counter()
    try:
        components = []
        for c in raw_components:
            family = c["family"]
            params = get_part_family(family).params_cls(**c["params"])
            components.append(
                ComponentSpec(
                    component_id=c["id"], family=family, clause_text=c.get("clause_text", ""),
                    params=params, count=int(c.get("count", 1)),
                )
            )
        spec = AssemblySpec(prompt=body.get("prompt", ""), components=tuple(components))
        result = validate_assembly(build_assembly(spec))
    except Exception as exc:  # surfaced to the UI rather than a bare 500 page
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    elapsed = time.perf_counter() - t0

    if result.mesh is None:
        return jsonify({"error": "assembly produced no geometry"}), 500

    mesh_id = f"{uuid.uuid4().hex}.stl"
    result.mesh.export(OUTPUT_DIR / mesh_id)

    # one representative geometry report per LOGICAL component (all
    # instances of a repeated component share the same locally-built mesh,
    # so instance 0 is representative of every instance)
    component_summaries = []
    for spec_component in spec.components:
        report_key = f"{spec_component.component_id}[0]"
        report = result.geometry_reports.get(report_key)
        component_summaries.append({
            "id": spec_component.component_id,
            "family": spec_component.family,
            "count": spec_component.count,
            "resolved_description": spec_component.params.to_description(),
            "watertight": report.watertight if report is not None else None,
        })

    return jsonify({
        "mesh_url": f"/api/mesh/{mesh_id}",
        "components": component_summaries,
        "overlapping_pairs": list(result.overlapping_pairs),
        "generation_time_seconds": elapsed,
    })


@app.route("/api/mesh/<mesh_id>")
def api_mesh(mesh_id):
    path = OUTPUT_DIR / mesh_id
    if not path.is_file() or path.suffix != ".stl":
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="model/stl", download_name=mesh_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5050)
    args = ap.parse_args()
    print(f"CAD prototype UI: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
