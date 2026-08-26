#!/usr/bin/env python3
"""python server.py [--port 5050]

Tiny local web frontend for the prompt -> mesh pipeline. Type a prompt in the
browser, it calls the exact same generate_from_prompt() that inference.py's
CLI uses, and shows the resulting STL in an interactive 3D viewer.

No new generation logic here -- this is only a thin HTTP wrapper around
inference.py, so anything true of the CLI (two part families, regex parsing,
CPU-only, milliseconds per generation) is true here too. See README.md.
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request, send_file, send_from_directory

from inference import generate_from_prompt

OUTPUT_DIR = Path(__file__).resolve().parent / "webapp_output"
OUTPUT_DIR.mkdir(exist_ok=True)

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
