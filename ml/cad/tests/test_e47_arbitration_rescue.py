"""Phase 8 Part B -- confirms E47 is actually fixed, not merely
disclosed. E47's confirmed root cause (Phase 7's own lock-in test,
test_shared_unit_dimension_parsing.test_e47_root_cause_is_operation_
selection_not_dimension_parsing): op_arbitration.arbitrate() correctly
detects when NEITHER the classifier's nor RAG's default-top-5 candidate
satisfies the text's own structured requirements (NO_VALID_PLAN), but
predict_program_hybrid used to silently execute the classifier's own
known-invalid candidate anyway, with nothing beyond the `arbitration`
metadata field ever disclosing the gap.

The fix (program_model._rescue_valid_alternative + the widened-search
branch in predict_program_hybrid) does two things, tested separately
below: (1) genuinely attempts a wider search for a valid alternative
before conceding, and (2) still returns NO_VALID_PLAN honestly -- never
forcing a weakly-supported substitute -- when nothing above the
project's own existing ood_similarity_threshold confidence floor
exists. Both are required: a "fix" that silently grabbed ANY technically
valid but barely-related training example would trade one wrong-but-
confident answer for a different one."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sklearn")

from sandbox import ToolCall


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    from data.program_dataset import sample_program_dataset, write_program_dataset
    from program_model import ProgramModelConfig, train_program_model

    root = tmp_path_factory.mktemp("tiny_rescue_dataset")
    samples = sample_program_dataset({"plate": 20, "shaft": 20}, n_generic=20, seed=0)
    write_program_dataset(root, samples, seed=0)
    bundle, _ = train_program_model(root, ProgramModelConfig(seed=0))
    return bundle


def _fake_predict_program(classifier_ops):
    def _impl(bundle, text, forced_ops=None):
        ops = forced_ops if forced_ops is not None else list(classifier_ops)
        return [ToolCall(op=o, args={}) for o in ops]
    return _impl


def test_e47_widened_search_rescues_a_confident_alternative(tiny_bundle):
    """When arbitrate()'s own default pass returns NO_VALID_PLAN, but a
    wider search turns up a genuinely requirement-satisfying candidate
    at a similarity ABOVE the existing ood_similarity_threshold floor,
    predict_program_hybrid must use it -- not silently keep executing
    the known-invalid classifier sequence."""
    from program_model import predict_program_hybrid

    text = "a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore"
    invalid_classifier_ops = ("add_box", "add_box", "boolean_difference", "finalize")  # a rectangular pocket, not a round bore
    invalid_rag_ops = ("add_box", "chamfer_edge", "finalize")  # also no bore feature at all
    rescue_ops = ("add_box", "counterbore_hole", "finalize")  # genuinely satisfies the bore requirement
    rescue_sim = tiny_bundle.config.ood_similarity_threshold + 0.10  # comfortably above the floor

    with (
        patch("program_model.predict_program", side_effect=_fake_predict_program(invalid_classifier_ops)),
        patch("program_model.retrieve_nearest_op_sequence_requirement_aware", return_value=(invalid_rag_ops, 0.9)),
        patch("program_model._rescue_valid_alternative", return_value=(rescue_ops, rescue_sim)),
    ):
        program, used_rag, result = predict_program_hybrid(tiny_bundle, text)

    assert result.outcome.name != "NO_VALID_PLAN"
    assert used_rag is True
    assert tuple(c.op for c in program) == rescue_ops
    assert any("widened retrieval" in line for line in result.reasoning)


def test_e47_stays_honestly_unresolved_when_rescue_is_too_weak(tiny_bundle):
    """The other half: a wider search that only turns up a candidate
    BELOW the confidence floor must NOT be used -- NO_VALID_PLAN must be
    reported honestly rather than papered over with a barely-related
    substitute. The classifier's own (still-invalid) sequence remains
    the disclosed default, exactly as before this fix, but now with an
    explicit record that a wider search was genuinely attempted first."""
    from program_model import predict_program_hybrid

    text = "a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore"
    invalid_classifier_ops = ("add_box", "add_box", "boolean_difference", "finalize")
    invalid_rag_ops = ("add_box", "chamfer_edge", "finalize")
    weak_rescue_ops = ("add_box", "counterbore_hole", "finalize")
    weak_rescue_sim = tiny_bundle.config.ood_similarity_threshold - 0.05  # below the floor

    with (
        patch("program_model.predict_program", side_effect=_fake_predict_program(invalid_classifier_ops)),
        patch("program_model.retrieve_nearest_op_sequence_requirement_aware", return_value=(invalid_rag_ops, 0.9)),
        patch("program_model._rescue_valid_alternative", return_value=(weak_rescue_ops, weak_rescue_sim)),
    ):
        program, used_rag, result = predict_program_hybrid(tiny_bundle, text)

    assert result.outcome.name == "NO_VALID_PLAN"
    assert used_rag is False
    assert tuple(c.op for c in program) == invalid_classifier_ops
    assert any("below the" in line and "floor" in line for line in result.reasoning)


def test_e47_no_rescue_candidate_found_at_all_stays_no_valid_plan(tiny_bundle):
    """When the widened search finds nothing valid whatsoever (not even
    a weak candidate), the outcome must still be the honest NO_VALID_PLAN,
    not a crash and not a silent substitution."""
    from program_model import predict_program_hybrid

    text = "a bearing housing, 40mm x 40mm x 20mm, with a 12.2mm bore"
    invalid_classifier_ops = ("add_box", "add_box", "boolean_difference", "finalize")
    invalid_rag_ops = ("add_box", "chamfer_edge", "finalize")

    with (
        patch("program_model.predict_program", side_effect=_fake_predict_program(invalid_classifier_ops)),
        patch("program_model.retrieve_nearest_op_sequence_requirement_aware", return_value=(invalid_rag_ops, 0.9)),
        patch("program_model._rescue_valid_alternative", return_value=None),
    ):
        program, used_rag, result = predict_program_hybrid(tiny_bundle, text)

    assert result.outcome.name == "NO_VALID_PLAN"
    assert used_rag is False
    assert tuple(c.op for c in program) == invalid_classifier_ops
    assert any("no valid alternative at all" in line for line in result.reasoning)


def test_e47_rescue_reports_the_winning_candidates_own_similarity_not_the_blanket_nearest_neighbor():
    """_rescue_valid_alternative's own contract: the reported similarity
    must belong to the WINNING valid candidate's own best occurrence,
    never retrieve_nearest_op_sequence_requirement_aware's blanket
    "closest neighbor of any kind" number (sims[0]) -- a high-confidence
    match to a totally different, invalid neighbor must never be
    mistaken for confidence in the rescue candidate itself."""
    import numpy as np
    from scipy import sparse

    from program_model import ProgramModelBundle, ProgramModelConfig, _rescue_valid_alternative

    # Two training examples: the nearer one (similarity 0.9) has an
    # INVALID sequence for the query text (no hole feature at all); the
    # farther one (similarity 0.3) has a VALID sequence. The function
    # must return the valid sequence's own similarity (0.3), not 0.9.
    vectorizer = type("FakeVectorizer", (), {"transform": staticmethod(lambda texts: sparse.csr_matrix(np.array([[1.0, 0.0]])))})()
    train_vectors = sparse.csr_matrix(np.array([[0.9, 0.0], [0.3, 0.0]]))
    bundle = ProgramModelBundle(
        vectorizer=vectorizer, slot_op_classifiers=[], slot_op_label_encoder=None,
        arg_estimators={}, arg_label_encoders={}, config=ProgramModelConfig(seed=0),
        train_text_vectors=train_vectors,
        train_op_sequences=[("add_box", "finalize"), ("add_box", "counterbore_hole", "finalize")],
    )
    result = _rescue_valid_alternative(bundle, "a block with a 10mm bore", k=2)
    assert result is not None
    ops, sim = result
    assert ops == ("add_box", "counterbore_hole", "finalize")
    assert sim == pytest.approx(0.3)
