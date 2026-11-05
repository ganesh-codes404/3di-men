"""showcases.py -- the single source of truth for the landing page's F1 car
(see that module's own docstring for why it exists). Structural/shape
checks only: the actual geometry was verified by diffing a live build of
this exact data against the old hand-maintained frontend/index.html copy
it replaced (byte-identical topology and per-entity resolved positions --
see the commit that introduced this module), not re-derived here."""

from showcases import SHOWCASES, get_showcase


def test_f1car_is_registered():
    assert "f1car" in SHOWCASES
    assert get_showcase("f1car") is not None


def test_get_showcase_returns_none_for_an_unknown_key():
    assert get_showcase("not_a_real_showcase") is None


def test_f1car_has_34_uniquely_named_entities():
    entities = get_showcase("f1car")["entities"]
    assert len(entities) == 34
    names = [e["name"] for e in entities]
    assert len(names) == len(set(names))  # no duplicates


def test_every_entity_has_a_well_formed_program_ending_in_finalize():
    entities = get_showcase("f1car")["entities"]
    for e in entities:
        assert e["name"]
        assert isinstance(e["position_mm"], dict) and set(e["position_mm"]) >= {"x", "y", "z"}
        assert e["program"], f"{e['name']} has an empty program"
        assert e["program"][-1]["op"] == "finalize", f"{e['name']}'s program doesn't end in finalize"


def test_nose_carries_the_abutted_to_relationship_fix():
    entities = get_showcase("f1car")["entities"]
    nose = next(e for e in entities if e["name"] == "Nose")
    rel = nose.get("relationship")
    assert rel is not None
    assert rel["kind"] == "abutted_to"
    assert rel["target_id"] == "ChassisFront"
    assert rel["abutment_direction"] == "+"


def test_f1car_declares_a_y_up_axis():
    # a deliberate, disclosed exception to this project's usual Z-up
    # convention -- see showcases.py's own comment on SHOWCASES.
    assert get_showcase("f1car")["upAxis"] == "y"
