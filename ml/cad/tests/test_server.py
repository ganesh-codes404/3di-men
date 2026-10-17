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
