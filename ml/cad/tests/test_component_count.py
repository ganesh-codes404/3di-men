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
