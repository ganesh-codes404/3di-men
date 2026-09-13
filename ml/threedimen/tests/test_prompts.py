import pytest

from threedimen.prompts import CATEGORIES, generate_prompt, generate_prompts


@pytest.mark.parametrize("category", sorted(CATEGORIES))
def test_generate_prompt_for_every_category(category):
    p = generate_prompt(category=category)
    assert p.category == category
    assert p.text
    assert p.attributes["noun"] in CATEGORIES[category].nouns


def test_generate_prompt_unknown_category_raises():
    with pytest.raises(ValueError):
        generate_prompt(category="not_a_real_category")


def test_generate_prompts_count():
    prompts = generate_prompts(15, seed=0)
    assert len(prompts) == 15


def test_generate_prompts_deterministic_with_same_seed():
    a = generate_prompts(10, seed=42)
    b = generate_prompts(10, seed=42)
    assert [p.text for p in a] == [p.text for p in b]


def test_generate_prompts_varies_with_different_seed():
    a = generate_prompts(10, seed=1)
    b = generate_prompts(10, seed=2)
    assert [p.text for p in a] != [p.text for p in b]


def test_generate_prompts_respects_category_filter():
    prompts = generate_prompts(20, categories=["pump", "valve"], seed=0)
    assert {p.category for p in prompts} <= {"pump", "valve"}


def test_generate_prompts_unknown_category_in_list_raises():
    with pytest.raises(ValueError):
        generate_prompts(5, categories=["pump", "not_real"])
