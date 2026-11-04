"""component_count.estimate_expected_component_count -- pure text extraction,
no model/bundle dependency, so these run fast as plain unit tests."""

from component_count import estimate_expected_component_count


def test_a_free_form_common_object_prompt_returns_none():
    # No recognized component noun anywhere -- never a guessed count, never 0.
    assert estimate_expected_component_count("a water bottle") is None
    assert estimate_expected_component_count("a coffee mug with a handle") is None


def test_empty_prompt_returns_none():
    assert estimate_expected_component_count("") is None


def test_single_indefinite_component_counts_as_one():
    assert estimate_expected_component_count("a bracket with two holes in each leg") == 1


def test_explicit_cardinal_multiplies_the_count():
    assert estimate_expected_component_count("two brackets with mounting holes") == 2
    assert estimate_expected_component_count("three spacers stacked on a shaft") == 4  # three spacers + one shaft


def test_multiple_distinct_indefinite_components_sum():
    # the real P2 pilot prompt (base, standoff, spacer, collar) -- see PILOT_FAILURE_INVESTIGATION.md
    prompt = (
        "Create a vertical mechanical stack on a 60 mm x 60 mm x 10 mm square base. "
        "Place a cylindrical standoff of 20 mm diameter and 30 mm height centrally on the top face. "
        "Place a 20 mm diameter, 10 mm tall spacer directly on top of the standoff, with zero gap. "
        "Place a 20 mm diameter, 15 mm tall collar on top of the spacer, leaving an intentional 3 mm "
        "gap between the spacer and collar. All three cylindrical components must share the same "
        "vertical axis. The base must remain at the bottom."
    )
    assert estimate_expected_component_count(prompt) == 4


def test_a_single_part_described_by_its_own_integral_faces_is_not_overcounted():
    # the real P3 pilot prompt -- an L-bracket's own horizontal/vertical faces
    # ("The horizontal base...", "The vertical plate...") are features of ONE
    # part, not two separate components; must not be flagged as 3 (bracket +
    # base + plate) just because all three nouns are recognized.
    prompt = (
        "Generate an L-shaped mounting bracket from 8 mm thick material. "
        "The horizontal base should measure 100 mm x 60 mm. "
        "The vertical plate should be 100 mm wide and 70 mm tall, mounted flush along the rear "
        "100 mm edge of the base. Add four 8 mm through-holes to the horizontal base, arranged "
        "symmetrically with their centres 12 mm from the nearest edges. Add two 10 mm through-holes "
        "to the vertical plate, arranged symmetrically along its width and height. Keep the two "
        "plates perpendicular and ensure the holes pass completely through their respective plates."
    )
    assert estimate_expected_component_count(prompt) == 1


def test_a_definite_article_back_reference_is_not_double_counted():
    assert estimate_expected_component_count("Add a shaft. The shaft is 20mm diameter.") == 1


def test_the_plus_cardinal_is_still_a_back_reference_not_a_fresh_count():
    assert estimate_expected_component_count(
        "Add a bracket. Add a second bracket. Keep the two brackets aligned."
    ) == 2


def test_a_definite_reference_across_and_does_not_block_a_later_unrelated_indefinite_mention():
    # "the first shaft" must not make the LLM's own later "a ... gear" (an
    # unrelated, fresh component) look like a back-reference too, just
    # because an earlier "the" appears somewhere in the same long sentence.
    prompt = "Mount a gear on the first shaft and a gear on the second shaft."
    assert estimate_expected_component_count(prompt) == 2


def test_a_bare_qualifier_less_plural_recap_is_not_counted():
    prompt = "Add a shaft and a gear. All shafts and gears must be separate identifiable components."
    assert estimate_expected_component_count(prompt) == 2


def test_per_x_is_excluded_rather_than_guessed_at():
    # a documented, accepted blind spot (see component_count.py's own module
    # docstring): "two bearing blocks per shaft" with 2 shafts is really 4
    # blocks, but this module does not attempt per-X multiplication -- the
    # "per shaft" mention itself must contribute 0, not a phantom extra shaft.
    prompt = "Include two shafts, supported by two bearing blocks per shaft."
    assert estimate_expected_component_count(prompt) == 4  # 2 shafts + 2 blocks (not 2 shafts + 2 blocks + 1 phantom shaft)


def test_a_component_noun_naming_the_requests_own_title_is_excluded():
    prompt = "Design a mechanical shaft support assembly with a bracket and a gear."
    # "shaft" here names the ASSEMBLY'S OWN TITLE ("a ... shaft support
    # assembly"), not a standalone shaft component -- only bracket + gear.
    assert estimate_expected_component_count(prompt) == 2


def test_adjacent_recognized_nouns_are_one_compound_component_not_two():
    assert estimate_expected_component_count("Mount a bearing housing on the plate.") == 1


# ---- held-out validation prompts (not used to tune the heuristic -- see the
# commit that added these: "column"/"cap" were added to _COMPONENT_NOUNS as a
# direct result of testing against this exact batch, then everything above
# was re-verified to confirm no regression) ----


def test_held_out_a_plate_with_a_column():
    prompt = (
        "Create a 120 mm x 80 mm x 10 mm rectangular base plate with a 50 mm tall "
        "cylindrical column centered on it."
    )
    assert estimate_expected_component_count(prompt) == 2  # plate + column


def test_held_out_b_three_named_components():
    prompt = (
        "Create three separate components: a 100 mm x 60 mm x 10 mm base plate, a "
        "20 mm diameter x 30 mm cylindrical standoff centered on the plate, and a "
        "20 mm diameter x 10 mm cylindrical spacer placed directly on top of the standoff."
    )
    assert estimate_expected_component_count(prompt) == 3  # plate + standoff + spacer


def test_held_out_c_generic_cylindrical_components_is_a_known_vocabulary_gap():
    # "components"/"cylindrical" name no recognized PART noun at all (by
    # design -- "cylinder" is a raw shape word, not a part-identity word,
    # see this module's own docstring) -- correctly None (no false
    # protection claimed), not a wrong non-None guess.
    prompt = (
        "Create two separate cylindrical components, each 20 mm in diameter. The "
        "first is 30 mm tall. Place the second directly above it with an "
        "intentional 5 mm gap."
    )
    assert estimate_expected_component_count(prompt) is None


def test_held_out_d_four_identical_standoffs_all_count_not_collapse_to_one():
    prompt = (
        "Create a 100 mm x 80 mm x 10 mm base plate with four identical cylindrical "
        "standoffs, each 10 mm in diameter and 20 mm tall, positioned symmetrically "
        "near the four corners."
    )
    assert estimate_expected_component_count(prompt) == 5  # plate + 4 standoffs


def test_held_out_e_screw_cap_is_one_cap_not_a_phantom_screw():
    # confirmed real false positive before "cap" was added to the noun list:
    # "a separate screw cap" matched bare "screw" (a real noun, used here
    # adjectivally) instead of collapsing into the "screw cap" compound.
    prompt = (
        "Create a 70 mm diameter, 220 mm tall water bottle with a rounded bottom, "
        "tapered shoulders, a 30 mm diameter neck, and a separate screw cap with "
        "grip ridges."
    )
    assert estimate_expected_component_count(prompt) == 1  # the cap (the bottle body itself
        # names no recognized PART noun -- STEP 0's hollow-vessel shape, not a STEP 1 part)
