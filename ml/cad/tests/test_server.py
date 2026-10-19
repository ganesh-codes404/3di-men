"""Tests server.py's routes directly via Flask's test client -- covers only
/api/tools and /api/execute_program here (the manual tool console's
backend), since neither touches the LLM or program_model.py and can run
fast in this suite. /api/build and /api/clarify are exercised end-to-end
manually against the real model (see the session's own verification runs),
not duplicated here."""

import pytest

pytest.importorskip("flask")

import server


@pytest.fixture()
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def test_api_tools_returns_the_full_manual_tool_roster(client):
    res = client.get("/api/tools")
    assert res.status_code == 200
    tools = res.get_json()["tools"]
    ops = [t["op"] for t in tools]
    assert "add_box" in ops
    assert "fillet_edge" in ops
    assert "finalize" not in ops  # not a modeling op -- handled by the console itself


def test_api_execute_program_builds_a_box_and_fillets_an_edge(client):
    res = client.post("/api/execute_program", json={
        "program": [
            {"op": "add_box", "args": {"length": 40, "width": 30, "height": 20, "center": [0, 0, 0]}},
            {"op": "fillet_edge", "args": {"target": 0, "direction1": "+z", "direction2": "+x", "radius": 5}},
            {"op": "finalize", "args": {"target": 1}},
        ]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["mesh_url"] is not None
    assert data["topology"]["is_watertight"] is True


def test_api_execute_program_rejects_an_unknown_op(client):
    res = client.post("/api/execute_program", json={"program": [{"op": "delete_system32", "args": {}}]})
    assert res.status_code == 200  # a typed rejection, not a 500 or an arbitrary call
    data = res.get_json()
    assert data["status"] == "rejected"
    assert data["mesh_url"] is None


def test_api_execute_program_errors_cleanly_on_empty_program(client):
    res = client.post("/api/execute_program", json={"program": []})
    assert res.status_code == 400


def test_api_execute_program_reports_partial_status_without_crashing(client):
    res = client.post("/api/execute_program", json={
        "program": [
            {"op": "add_box", "args": {"length": 40, "width": 30, "height": 20, "center": [0, 0, 0]}},
            # references a handle that doesn't exist -- validate() catches this structurally
            {"op": "translate", "args": {"target": 99, "dx": 1, "dy": 0, "dz": 0}},
            {"op": "finalize", "args": {"target": 1}},
        ]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "rejected"
    assert data["error"] is not None


def test_api_execute_programs_builds_separate_entities_and_a_combined_assembly(client):
    res = client.post("/api/execute_programs", json={
        "entities": [
            {
                "name": "Gear",
                "program": [
                    {"op": "add_gear_profile", "args": {"module_mm": 2, "tooth_count": 16, "face_width_mm": 10, "bore_diameter_mm": 8}},
                    {"op": "finalize", "args": {"target": 0}},
                ],
                "position_mm": {"x": 0, "y": 0, "z": 0},
            },
            {
                "name": "Shaft",
                "program": [
                    {"op": "add_cylinder", "args": {"radius": 4, "height": 60, "center": [0, 0, 0], "axis": "z"}},
                    {"op": "finalize", "args": {"target": 0}},
                ],
                "position_mm": {"x": 40, "y": 0, "z": 0},
            },
        ]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert data["assembly_mesh_url"] is not None
    assert len(data["entities"]) == 2
    names = [e["name"] for e in data["entities"]]
    assert names == ["Gear", "Shaft"]
    for e in data["entities"]:
        assert e["status"] == "ok"
        assert e["mesh_url"] is not None  # each entity's OWN mesh, separate from the combined assembly
        assert e["topology"]["is_watertight"] is True
    # two parts 40mm apart along x, each with its own real extent -- must not overlap in the combined assembly
    assert data["assembly_topology"]["bounding_box_mm"][0] > 40


def test_api_execute_programs_errors_cleanly_with_no_valid_entities(client):
    res = client.post("/api/execute_programs", json={"entities": [{"name": "empty", "program": []}]})
    assert res.status_code == 400


def test_api_execute_programs_reports_a_failing_entity_without_dropping_the_others(client):
    res = client.post("/api/execute_programs", json={
        "entities": [
            {"name": "Good", "program": [
                {"op": "add_box", "args": {"length": 20, "width": 20, "height": 20, "center": [0, 0, 0]}},
                {"op": "finalize", "args": {"target": 0}},
            ], "position_mm": {"x": 0, "y": 0, "z": 0}},
            {"name": "Bad", "program": [
                {"op": "add_box", "args": {"length": 20, "width": 20, "height": 20, "center": [0, 0, 0]}},
                {"op": "translate", "args": {"target": 99, "dx": 1, "dy": 0, "dz": 0}},
                {"op": "finalize", "args": {"target": 1}},
            ], "position_mm": {"x": 50, "y": 0, "z": 0}},
        ]
    })
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["entities"]) == 2
    statuses = {e["name"]: e["status"] for e in data["entities"]}
    assert statuses["Good"] == "ok"
    assert statuses["Bad"] == "rejected"
    assert data["assembly_mesh_url"] is not None  # the good entity alone still forms a combined assembly


def test_parse_point_program_override_roundtrips_plain_dicts():
    program = server._parse_point_program_override([
        {"op": "add_box", "args": {"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}},
        {"op": "finalize", "args": {"target": 0}},
    ])
    assert program == [
        {"op": "add_box", "args": {"length": 10, "width": 10, "height": 10, "center": [0, 0, 0]}},
        {"op": "finalize", "args": {"target": 0}},
    ]


def test_parse_point_program_override_returns_none_when_absent():
    # None (not []) distinguishes "nothing was sent" from "sent as an
    # empty program" -- see ActionablePoint.program_override's own
    # docstring for why that distinction matters to parallel_builder.py.
    assert server._parse_point_program_override(None) is None


def test_parse_point_program_override_skips_malformed_steps():
    program = server._parse_point_program_override([
        {"op": "add_box", "args": {"length": 10}},
        {"args": {"length": 10}},  # missing "op" entirely -- skipped
        "not even a dict",  # skipped
        {"op": "finalize"},  # missing "args" -- defaults to {}
    ])
    assert program == [
        {"op": "add_box", "args": {"length": 10}},
        {"op": "finalize", "args": {}},
    ]


def test_exported_mesh_round_trips_through_stl():
    """Export/reimport integrity (mission brief Phase 9, Test 11): a built
    mesh exported via server._export_mesh (the same STL export every real
    build uses) must reimport with its dimensions/volume/watertightness
    preserved -- confirms the actual exported ARTIFACT is correct, not
    just the in-memory mesh object before export ever touched disk."""
    import numpy as np
    import trimesh

    from sandbox import Sandbox, ToolCall

    prog = [
        ToolCall("add_box", {"length": 40, "width": 30, "height": 20, "center": [0, 0, 0]}),
        ToolCall("finalize", {"target": 0}),
    ]
    result = Sandbox.execute(prog)
    assert result.status == "ok"
    original = result.mesh

    mesh_url = server._export_mesh(original)
    exported_path = server.OUTPUT_DIR / mesh_url.rsplit("/", 1)[-1]
    assert exported_path.is_file()

    reimported = trimesh.load_mesh(exported_path)
    assert reimported.is_watertight
    assert np.allclose(sorted(reimported.extents), sorted(original.extents))
    assert abs(reimported.volume - original.volume) < 1e-3
    exported_path.unlink()  # this test's own artifact, not a real build's -- clean up after itself


def test_exported_mesh_round_trips_through_3mf():
    """Print Lab export (printability.py's own module docstring): 3MF
    support was investigated, not assumed -- confirmed directly that
    this project's installed trimesh can both WRITE and READ BACK a
    valid 3MF file (export alone succeeding is not enough evidence; a
    library can write a file it cannot itself parse back correctly)."""
    import numpy as np
    import trimesh

    from sandbox import Sandbox, ToolCall

    prog = [
        ToolCall("add_box", {"length": 40, "width": 30, "height": 20, "center": [0, 0, 0]}),
        ToolCall("finalize", {"target": 0}),
    ]
    original = Sandbox.execute(prog).mesh

    exported = original.export(file_type="3mf")
    assert isinstance(exported, (bytes, bytearray)) and len(exported) > 0

    import io

    scene = trimesh.load(io.BytesIO(exported), file_type="3mf")
    reimported = scene.to_geometry()
    assert reimported.is_watertight
    assert np.allclose(sorted(reimported.extents), sorted(original.extents))
    assert abs(reimported.volume - original.volume) < 1e-3


# ---------------- _parse_manual_points -- confirmed real bug this covers,
# found live-testing the exact motor-mount scenario against the real server:
# a manual "points" payload with id/relationship fields had BOTH silently
# dropped, so every point fell back to its raw (often identical, e.g. all
# (0,0,0)) position_mm with no relationship resolved at all -- and a later
# gear-mesh correction step then moved EVERY point that happened to share
# that coincidental position together, not just the one it was meant for.
# Neither the LLM nor program_model.py is involved in this function at all,
# so (unlike /api/build's own route) this is tested directly, fast. ----------

def test_parse_manual_points_preserves_id_and_relationship():
    points = server._parse_manual_points([
        {"id": "base", "description": "a box 150mm x 100mm x 10mm", "position_mm": {"x": 0, "y": 0, "z": 0}},
        {
            "id": "boss", "description": "a cylinder 30mm diameter, 20mm tall", "position_mm": {"x": 0, "y": 0, "z": 0},
            "relationship": {"kind": "attached_to_face", "target_id": "base", "face": "+z"},
        },
    ])
    assert len(points) == 2
    assert points[0].id == "base"
    assert points[1].id == "boss"
    assert points[1].relationship is not None
    assert points[1].relationship.kind == "attached_to_face"
    assert points[1].relationship.target_id == "base"
    assert points[1].relationship.face == "+z"


def test_parse_manual_points_ignores_an_unparseable_relationship_without_rejecting_the_point():
    """Mirrors _parse_relationship's own contract (llm_decompose.py): a
    malformed/absent relationship just means this point falls back to its
    own explicit position_mm, never a hard failure here."""
    points = server._parse_manual_points([
        {
            "description": "a box 40mm x 40mm x 40mm", "position_mm": {"x": 5, "y": 0, "z": 0},
            "relationship": {"kind": "not_a_real_kind", "target_id": "x"},
        },
    ])
    assert len(points) == 1
    assert points[0].relationship is None
    assert points[0].position_mm == (5.0, 0.0, 0.0)


def test_parse_manual_points_defaults_a_missing_id_to_empty_string():
    """Matches ActionablePoint's own default (id: str = "") -- downstream
    id auto-generation (parallel_builder._unique_point_ids) expects this,
    not None."""
    points = server._parse_manual_points([{"description": "a box 10mm x 10mm x 10mm", "position_mm": {"x": 0, "y": 0, "z": 0}}])
    assert points[0].id == ""
