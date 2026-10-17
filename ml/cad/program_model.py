"""The proprietary, self-trained, locally-run program-composition model --
predicts a SEQUENCE of sandbox.py tool calls rather than a flat family+params
record. No external LLM API anywhere in this file.

Architecture: K=10 "unrolled slots" on scikit-learn, not a neural sequence
model -- a mechanical unrolling of the exact pattern parser_model.py already
validated (classify -> look up per-field estimators), repeated per slot, each
slot's models taking TF-IDF(text) plus a fixed-size summary of already-
decided prior slots as input. See the design conversation this was built
from for why a real seq2seq/transformer was rejected: not training-speed
cost, but constrained decoding (handle references must never point at an
undefined/consumed step) and precise numeric generation, neither of which a
token-sampling decoder gives for free. Handles are made a bounded categorical
choice by construction (every reference is stored as a small RELATIVE
OFFSET -- "N steps back" -- never an absolute index), which is what lets
plain classifiers handle them with no masking/grammar machinery at all.

Known, honest limitation carried over from parser_model.py's reasoning, now
sharper: because family-derived training programs have a FIXED op-sequence
shape per family (only arguments vary), this model's op-CHOICE prediction
for the 6 known families is easy (high accuracy expected) almost by
construction -- the real test of this architecture is the small, separately-
tracked generic_composition eval bucket (program_eval.py), which is honestly
expected to score lower and should never be averaged with the family bucket.

Training uses TEACHER FORCING (ground-truth prior-slot values as training
features, not the model's own predictions) -- simpler to implement, at the
cost of a known train/inference mismatch (an early wrong prediction can't
self-correct downstream at inference time). Documented, not hidden.

Out-of-distribution disclosure: predict_program() will still return SOME
program for any text input, including wildly novel prompts ("a spaceship")
that share no real vocabulary with the 6 trained families -- a classifier
has to predict something, and it stays confidently wrong rather than
abstaining. That is not "handling" a novel prompt in any real sense; it's
guessing while looking sure of itself. predict_program_checked() and
nearest_neighbor_similarity() add an honest, separate signal for this
("have I seen anything like this text," not "is the output geometrically
valid" -- that's Sandbox.execute()'s job) so a caller can disclose a guess
instead of presenting it as understanding. This does not make the model
understand more -- genuinely open-ended natural-language-to-CAD synthesis
for arbitrary real-world objects is out of reach for a TF-IDF + scikit-learn
model trained on a few thousand mechanical-part examples, no LLM involved.
That gap is honest, not a bug to patch away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

K_SLOTS = 10
STOP = "STOP"

# name -> ("numeric" | "categorical" | "handle_offset")
ARG_SCHEMA: dict[str, list[tuple[str, str]]] = {
    "add_box": [("length", "numeric"), ("width", "numeric"), ("height", "numeric"),
                ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_cylinder": [("radius", "numeric"), ("height", "numeric"),
                      ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "add_extrusion": [("profile_kind", "categorical"), ("param_a", "numeric"), ("param_b", "numeric"), ("param_c", "numeric"),
                       ("height", "numeric"), ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_gear_profile": [("module_mm", "numeric"), ("tooth_count", "count"), ("face_width_mm", "numeric"),
                          ("bore_diameter_mm", "numeric"), ("pressure_angle_deg", "numeric"),
                          ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "boolean_union": [("a", "handle_offset"), ("b", "handle_offset")],
    "boolean_difference": [("a", "handle_offset"), ("b", "handle_offset")],
    "boolean_intersection": [("a", "handle_offset"), ("b", "handle_offset")],
    "translate": [("target", "handle_offset"), ("dx", "numeric"), ("dy", "numeric"), ("dz", "numeric")],
    "rotate": [("target", "handle_offset"), ("axis", "categorical"), ("angle_deg", "numeric")],
    "mirror": [("target", "handle_offset"), ("axis", "categorical")],
    "scale": [("target", "handle_offset"), ("factor_x", "numeric"), ("factor_y", "numeric"), ("factor_z", "numeric")],
    "drill_hole": [("target", "handle_offset"), ("radius", "numeric"), ("depth", "numeric"),
                    ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "drill_hole_pattern": [("target", "handle_offset"), ("pattern", "categorical"), ("count", "count"),
                            ("radius", "numeric"), ("depth", "numeric"), ("axis", "categorical"),
                            ("margin", "numeric"), ("circle_diameter", "numeric"), ("z", "numeric")],
    "linear_pattern": [("target", "handle_offset"), ("axis", "categorical"), ("count", "count"), ("spacing", "numeric")],
    "circular_pattern": [("target", "handle_offset"), ("count", "count"), ("radius", "numeric")],
    "extrude_face": [("target", "handle_offset"), ("direction", "categorical"), ("distance", "numeric")],
    "inset_pocket": [("target", "handle_offset"), ("direction", "categorical"), ("depth", "numeric"), ("margin", "numeric")],
    "shell": [("target", "handle_offset"), ("wall_thickness", "numeric")],
    "smooth_subdivide": [("target", "handle_offset"), ("subdivisions", "count"), ("iterations", "count")],
    "add_wireframe_from_edges": [("target", "handle_offset"), ("edge_radius", "numeric"), ("sections", "count")],
    "weld_vertices": [("target", "handle_offset"), ("tolerance", "numeric")],
    "add_revolve": [("bottom_radius", "numeric"), ("top_radius", "numeric"), ("height", "numeric"),
                     ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_shouldered_cylinder": [("bottom_radius", "numeric"), ("bottom_height", "numeric"), ("top_radius", "numeric"),
                                 ("top_height", "numeric"), ("transition_height", "numeric"),
                                 ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_sphere": [("radius", "numeric"), ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_torus": [("major_radius", "numeric"), ("minor_radius", "numeric"),
                  ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "add_capsule": [("radius", "numeric"), ("height", "numeric"),
                     ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "add_ellipsoid": [("radius_x", "numeric"), ("radius_y", "numeric"), ("radius_z", "numeric"),
                       ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_polygon_frustum": [("sides", "count"), ("bottom_radius", "numeric"), ("top_radius", "numeric"), ("height", "numeric"),
                             ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "add_dome": [("radius", "numeric"), ("height", "numeric"),
                 ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "add_ring": [("outer_radius", "numeric"), ("inner_radius", "numeric"), ("height", "numeric"),
                 ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "add_loft": [("bottom_profile", "categorical"), ("bottom_param_a", "numeric"), ("bottom_param_b", "numeric"),
                 ("bottom_param_c", "numeric"), ("top_profile", "categorical"), ("top_param_a", "numeric"),
                 ("top_param_b", "numeric"), ("top_param_c", "numeric"), ("height", "numeric"),
                 ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_sweep": [("profile_kind", "categorical"), ("param_a", "numeric"), ("param_b", "numeric"), ("param_c", "numeric"),
                  ("path_kind", "categorical"), ("path_param_a", "numeric"), ("path_param_b", "numeric"), ("path_param_c", "numeric"),
                  ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "add_tube": [("outer_radius", "numeric"), ("inner_radius", "numeric"), ("path_kind", "categorical"),
                 ("path_param_a", "numeric"), ("path_param_b", "numeric"), ("path_param_c", "numeric"),
                 ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric")],
    "taper": [("target", "handle_offset"), ("axis", "categorical"), ("bottom_factor", "numeric"), ("top_factor", "numeric")],
    "flare": [("target", "handle_offset"), ("axis", "categorical"), ("bottom_factor", "numeric"), ("top_factor", "numeric")],
    "twist": [("target", "handle_offset"), ("axis", "categorical"), ("angle_deg", "numeric")],
    "bend": [("target", "handle_offset"), ("axis", "categorical"), ("bend_direction", "categorical"), ("angle_deg", "numeric")],
    "fillet_edge": [("target", "handle_offset"), ("direction1", "categorical"), ("direction2", "categorical"), ("radius", "numeric")],
    "chamfer_edge": [("target", "handle_offset"), ("direction1", "categorical"), ("direction2", "categorical"), ("distance", "numeric")],
    "counterbore_hole": [("target", "handle_offset"), ("direction", "categorical"), ("offset_u", "numeric"), ("offset_v", "numeric"),
                          ("hole_radius", "numeric"), ("head_radius", "numeric"), ("head_depth", "numeric"), ("head_style", "categorical")],
    "drill_slot": [("target", "handle_offset"), ("direction", "categorical"), ("offset_u", "numeric"), ("offset_v", "numeric"),
                   ("radius", "numeric"), ("slot_length", "numeric"), ("slot_axis", "categorical"), ("depth", "numeric")],
    "add_rib": [("target", "handle_offset"), ("direction1", "categorical"), ("direction2", "categorical"), ("size", "numeric")],
    "round_box_edges": [("target", "handle_offset"), ("radius", "numeric")],
    "chamfer_box_edges": [("target", "handle_offset"), ("distance", "numeric")],
    "mirror_merge": [("target", "handle_offset"), ("axis", "categorical")],
    "add_boss_with_bore": [("target", "handle_offset"), ("direction", "categorical"), ("offset_u", "numeric"), ("offset_v", "numeric"),
                            ("boss_radius", "numeric"), ("boss_height", "numeric"), ("bore_radius", "numeric")],
    "counterbore_hole_pattern": [("target", "handle_offset"), ("direction", "categorical"), ("pattern", "categorical"), ("count", "count"),
                                  ("hole_radius", "numeric"), ("head_radius", "numeric"), ("head_depth", "numeric"), ("head_style", "categorical"),
                                  ("margin", "numeric"), ("circle_diameter", "numeric")],
    "add_slot_pattern": [("target", "handle_offset"), ("direction", "categorical"), ("pattern", "categorical"), ("count", "count"),
                          ("radius", "numeric"), ("slot_length", "numeric"), ("slot_axis", "categorical"), ("depth", "numeric"),
                          ("margin", "numeric"), ("circle_diameter", "numeric")],
    "add_knurl_pattern": [("target", "handle_offset"), ("direction", "categorical"), ("offset_u", "numeric"), ("offset_v", "numeric"),
                           ("radius", "numeric"), ("height", "numeric"), ("ridge_count", "count")],
    "radial_array_oriented": [("target", "handle_offset"), ("count", "count"), ("radius", "numeric")],
    "linear_array_scaled": [("target", "handle_offset"), ("axis", "categorical"), ("count", "count"), ("spacing", "numeric"), ("scale_step", "numeric")],
    "add_hex_socket": [("target", "handle_offset"), ("direction", "categorical"), ("offset_u", "numeric"), ("offset_v", "numeric"),
                        ("across_flats", "numeric"), ("depth", "numeric")],
    "finalize": [("target", "handle_offset")],
    STOP: [],
}
OPS = list(ARG_SCHEMA.keys())


def _flatten_args(op: str, args: dict[str, Any], slot_index: int) -> dict[str, float | str]:
    """Sandbox args (center as a [x,y,z] list, handles as absolute indices)
    -> flat scalar args (center_x/y/z, handles as relative "N steps back"
    offsets) -- the shape every per-argument estimator expects."""
    flat: dict[str, float | str] = {}
    for name, kind in ARG_SCHEMA[op]:
        if name in ("center_x", "center_y", "center_z"):
            axis_idx = {"center_x": 0, "center_y": 1, "center_z": 2}[name]
            flat[name] = float(args.get("center", [0, 0, 0])[axis_idx])
        elif kind == "handle_offset":
            flat[name] = float(slot_index - args[name])
        else:
            flat[name] = args[name]
    return flat


def _unflatten_args(op: str, flat: dict[str, float | str], slot_index: int) -> dict[str, Any]:
    args: dict[str, Any] = {}
    center = [0.0, 0.0, 0.0]
    for name, kind in ARG_SCHEMA[op]:
        if name in ("center_x", "center_y", "center_z"):
            axis_idx = {"center_x": 0, "center_y": 1, "center_z": 2}[name]
            center[axis_idx] = float(flat[name])
        elif kind == "handle_offset":
            args[name] = int(round(slot_index - flat[name]))
        elif kind == "count":
            # a Ridge regressor's continuous output must be rounded to a
            # valid int before use (range(), list indexing, etc. all reject
            # a bare float) -- clamped to >= 1 since a 0-or-negative count
            # is never valid for any op that takes one
            args[name] = max(1, int(round(float(flat[name]))))
        elif kind == "numeric":
            args[name] = float(flat[name])
        else:
            args[name] = flat[name]
    if any(n in ("center_x", "center_y", "center_z") for n, _ in ARG_SCHEMA[op]):
        args["center"] = center
    return args


@dataclass
class ProgramModelConfig:
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    k_slots: int = K_SLOTS
    seed: int = 0
    # Cosine-similarity floor (TF-IDF, nearest training example) below which
    # a prompt is flagged out-of-distribution rather than trusted -- see
    # nearest_neighbor_similarity()'s docstring for what this catches and
    # its honest calibration limits. 0.20 is a coarse cut from a tiny
    # hand-labeled probe (24 known-in-distribution vs. 8 known-OOD prompts,
    # see the design conversation this was calibrated from), not a
    # statistically rigorous threshold -- expect some misses in both
    # directions, not a guarantee.
    ood_similarity_threshold: float = 0.20


@dataclass
class ProgramModelBundle:
    vectorizer: Any
    slot_op_classifiers: list[Any]  # length k_slots
    slot_op_label_encoder: Any  # shared LabelEncoder over OPS (fixed vocabulary)
    arg_estimators: dict[str, dict[str, Any]]  # op -> arg_name -> fitted estimator
    arg_label_encoders: dict[str, dict[str, Any]]  # op -> categorical arg_name -> LabelEncoder
    config: ProgramModelConfig
    train_text_vectors: Any = None  # sparse TF-IDF matrix over training texts -- nearest_neighbor_similarity()'s reference set
    train_op_sequences: list[tuple[str, ...]] = field(default_factory=list)  # row-aligned with train_text_vectors;
                                                                              # real ops only, STOP/padding trimmed --
                                                                              # the retrieval corpus predict_program()'s
                                                                              # RAG path draws an op-sequence SHAPE from
    trained_on: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ProgramModelBundle":
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return bundle


def _prior_slot_features(op_sequence: list[str], upto: int, k_slots: int, op_encoder) -> list[float]:
    """Fixed-size one-hot-per-slot summary of slots [0, upto) -- slots >=
    upto are all-zero (unknown/future), matching both training (teacher
    forcing: ground-truth ops for slots < upto) and inference (the model's
    own previously predicted ops)."""
    n_ops = len(op_encoder.classes_)
    features = [0.0] * (k_slots * n_ops)
    for j in range(upto):
        op_idx = op_encoder.transform([op_sequence[j]])[0]
        features[j * n_ops + op_idx] = 1.0
    return features


def _pad_program_ops(program, k_slots: int) -> list[str]:
    ops = [c.op for c in program][:k_slots]
    ops += [STOP] * (k_slots - len(ops))
    return ops


def train_program_model(dataset_root: Path, config: ProgramModelConfig | None = None) -> tuple[ProgramModelBundle, dict]:
    """Fits on the TRAIN split only. Train-set fit diagnostics are NOT a
    real-accuracy claim -- see program_eval.py for the honest, 3-bucket
    evaluation methodology."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import LabelEncoder

    from data.program_dataset import read_program_split

    config = config or ProgramModelConfig()
    train_samples = read_program_split(dataset_root, "train")
    if not train_samples:
        raise ValueError(f"no train samples found under {dataset_root}")

    texts = [s.text for s in train_samples]
    vectorizer = TfidfVectorizer(max_features=config.tfidf_max_features, ngram_range=config.tfidf_ngram_range)
    X_text_sparse = vectorizer.fit_transform(texts)  # kept sparse for nearest_neighbor_similarity(); densified below for training
    X_text = X_text_sparse.toarray()

    op_encoder = LabelEncoder()
    op_encoder.fit(OPS)  # fixed vocabulary, not fit from data -- every op must be representable

    padded_ops = [_pad_program_ops(s.program, config.k_slots) for s in train_samples]

    # ---- per-slot op classifiers ----
    slot_op_classifiers = []
    for slot in range(config.k_slots):
        prior_features = np.array([_prior_slot_features(ops, slot, config.k_slots, op_encoder) for ops in padded_ops])
        X = np.hstack([X_text, prior_features])
        y = op_encoder.transform([ops[slot] for ops in padded_ops])
        clf = LogisticRegression(max_iter=1000, random_state=config.seed)
        if len(set(y)) < 2:
            clf = _ConstantClassifier(y[0])
        else:
            clf.fit(X, y)
        slot_op_classifiers.append(clf)

    # ---- per-op argument estimators, pooled across all slots where that op occurs ----
    arg_estimators: dict[str, dict[str, Any]] = {}
    arg_label_encoders: dict[str, dict[str, Any]] = {}
    train_metrics: dict[str, Any] = {"n_train": len(train_samples), "ops": {}}

    for op in OPS:
        if op == STOP or not ARG_SCHEMA[op]:
            continue
        rows_X, rows_flat_args = [], []
        for sample_idx, sample in enumerate(train_samples):
            for slot, call in enumerate(sample.program[: config.k_slots]):
                if call.op != op:
                    continue
                prior_features = _prior_slot_features(padded_ops[sample_idx], slot, config.k_slots, op_encoder)
                X = np.concatenate([X_text[sample_idx], prior_features, [float(slot)]])
                rows_X.append(X)
                rows_flat_args.append(_flatten_args(op, call.args, slot))

        if not rows_X:
            continue
        X_op = np.array(rows_X)
        arg_estimators[op] = {}
        arg_label_encoders[op] = {}
        op_metrics = {}
        for name, kind in ARG_SCHEMA[op]:
            y_raw = [row[name] for row in rows_flat_args]
            if kind == "categorical":
                encoder = LabelEncoder()
                y = encoder.fit_transform([str(v) for v in y_raw])
                arg_label_encoders[op][name] = encoder
                estimator = _ConstantClassifier(y[0]) if len(set(y)) < 2 else LogisticRegression(max_iter=1000, random_state=config.seed)
            else:
                y = [float(v) for v in y_raw]
                estimator = Ridge(random_state=config.seed)
            if hasattr(estimator, "fit"):
                estimator.fit(X_op, y)
            arg_estimators[op][name] = estimator
            op_metrics[name] = float(estimator.score(X_op, y)) if hasattr(estimator, "score") else 1.0
        train_metrics["ops"][op] = op_metrics

    train_op_sequences = [tuple(c.op for c in s.program[: config.k_slots]) for s in train_samples]

    bundle = ProgramModelBundle(
        vectorizer=vectorizer, slot_op_classifiers=slot_op_classifiers, slot_op_label_encoder=op_encoder,
        arg_estimators=arg_estimators, arg_label_encoders=arg_label_encoders, config=config,
        train_text_vectors=X_text_sparse, train_op_sequences=train_op_sequences,
        trained_on={"dataset_root": str(dataset_root), "train_count": len(train_samples)},
    )
    return bundle, train_metrics


class _ConstantClassifier:
    """Fallback for an (op, slot) or (op, arg) cell with only one observed
    class in training data -- sklearn classifiers require >= 2 classes to
    fit; this just always predicts the one seen value/probability 1.0."""

    def __init__(self, constant):
        self.constant = constant
        self.classes_ = [constant]

    def predict(self, X):
        import numpy as np

        return np.full(len(X), self.constant)

    def predict_proba(self, X):
        import numpy as np

        return np.ones((len(X), 1))

    def score(self, X, y):
        return 1.0


import re

_DIMENSION_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)


def _override_primary_shape_dimensions(op: str, args: dict, text: str) -> dict:
    """When the text explicitly states the SHAPE-CREATING op's own primary
    dimensions (the common case -- "a cylinder 20mm diameter, 15mm tall"),
    use those literal numbers directly instead of the trained regressors'
    approximation. Confirmed as a real, severe gap, not hypothetical: on
    the exact text "a cylinder 20mm diameter, 15mm tall with a 6mm
    diameter hole through it", the regressor predicted height=52.6mm for a
    stated 15mm -- a >3x error -- because TF-IDF-driven regression has no
    actual mechanism to read the digits out of the text, only to correlate
    word patterns with typical training values. Literal numbers, when
    present, are unambiguous ground truth and should always win.

    Deliberately scoped to ONLY the first op (the one that creates the
    base shape, i.e. one of sandbox.py's _CREATES set) and to its OWN
    first len(numeric_fields) literal "Nmm" values in the text, in
    ARG_SCHEMA's declared order -- not every op in a multi-step program.
    Mapping a flat list of literal numbers to EVERY op's args across a
    whole multi-op sequence (e.g. which number is a hole's margin vs its
    depth) is a much harder, more error-prone problem; the base shape's
    own overall size is both the most reliably extractable case (its
    numbers are almost always stated first, before any feature detail)
    and the most visible error to a user if wrong, so it is the one
    handled here. If the text doesn't have enough explicit numbers to
    cover this op's own dimensions, nothing is overridden -- never a
    partial/guessed mapping.

    A field whose name contains "radius" is halved from the literal
    value: this project's own training-text convention (see
    data/program_dataset.py's patterns) consistently states a DIAMETER in
    text ("20mm diameter") while the matching tool argument is a radius
    -- confirmed directly across _pattern_dome/_pattern_sphere/
    _pattern_shouldered_cylinder/etc., not assumed."""
    from sandbox import _CREATES

    if op not in _CREATES:
        return args
    numeric_fields = [name for name, kind in ARG_SCHEMA[op] if kind == "numeric" and name not in ("center_x", "center_y", "center_z")]
    if not numeric_fields:
        return args
    literals = [float(m) for m in _DIMENSION_RE.findall(text)]
    if len(literals) < len(numeric_fields):
        return args

    overridden = dict(args)
    for name, value in zip(numeric_fields, literals[: len(numeric_fields)]):
        overridden[name] = value / 2.0 if "radius" in name else value
    return overridden


def predict_program(bundle: ProgramModelBundle, text: str, forced_ops: list[str] | None = None) -> list:
    """K-slot autoregressive loop: predicts one op per slot (features =
    TF-IDF + summary of the model's OWN previously predicted slots, not
    ground truth -- this is real inference, unlike training's teacher
    forcing), stopping at STOP or finalize.

    `forced_ops`, when given, is a RETRIEVED op-sequence (see
    retrieve_nearest_op_sequence()/predict_program_rag() below) to use
    INSTEAD of the per-slot classifiers' own independent choice at each
    slot -- the RAG path. Argument prediction is unchanged either way:
    every op (chosen or forced) still goes through the same trained
    per-argument regressors, conditioned on the actual prior ops at that
    point, exactly as training's teacher forcing was conditioned."""
    import numpy as np

    from sandbox import ToolCall

    X_text = bundle.vectorizer.transform([text]).toarray()[0]
    op_encoder = bundle.slot_op_label_encoder
    predicted_ops: list[str] = []
    program: list[ToolCall] = []

    for slot in range(bundle.config.k_slots):
        if forced_ops is not None:
            op = forced_ops[slot] if slot < len(forced_ops) else STOP
        else:
            prior_features = _prior_slot_features(predicted_ops + [STOP] * (bundle.config.k_slots - len(predicted_ops)), slot, bundle.config.k_slots, op_encoder)
            X = np.concatenate([X_text, prior_features]).reshape(1, -1)
            clf = bundle.slot_op_classifiers[slot]
            op_idx = clf.predict(X)[0]
            op = op_encoder.inverse_transform([op_idx])[0]
        predicted_ops.append(op)

        if op == STOP:
            break

        args: dict[str, float | str] = {}
        if ARG_SCHEMA[op]:
            prior_features_for_args = _prior_slot_features(predicted_ops[:-1] + [STOP] * (bundle.config.k_slots - len(predicted_ops) + 1), slot, bundle.config.k_slots, op_encoder)
            X_arg = np.concatenate([X_text, prior_features_for_args, [float(slot)]]).reshape(1, -1)
            for name, kind in ARG_SCHEMA[op]:
                estimator = bundle.arg_estimators.get(op, {}).get(name)
                if estimator is None:
                    args[name] = 0.0 if kind != "categorical" else ""
                    continue
                if kind == "categorical":
                    idx = estimator.predict(X_arg)[0]
                    encoder = bundle.arg_label_encoders[op][name]
                    args[name] = encoder.inverse_transform([idx])[0]
                else:
                    args[name] = float(estimator.predict(X_arg)[0])
            args = _override_primary_shape_dimensions(op, args, text)

        full_args = _unflatten_args(op, args, slot) if ARG_SCHEMA[op] else {}
        program.append(ToolCall(op=op, args=full_args))

        if op == "finalize":
            break

    return program


def _nearest_neighbor_index_and_similarity(bundle: ProgramModelBundle, text: str) -> tuple[int, float]:
    """Shared by nearest_neighbor_similarity() so it reads off the same
    rank-1 result retrieve_nearest_op_sequence()'s top-k scan already
    computes, rather than vectorizing `text` and scanning
    train_text_vectors a second time."""
    idxs, sims = _top_k_neighbors(bundle, text, k=1)
    if not idxs:
        return -1, 0.0
    return idxs[0], sims[0]


def _top_k_neighbors(bundle: ProgramModelBundle, text: str, k: int) -> tuple[list[int], list[float]]:
    if bundle.train_text_vectors is None:
        raise ValueError("bundle has no stored train_text_vectors -- retrain to enable OOD detection/retrieval")
    import numpy as np

    x = bundle.vectorizer.transform([text])
    sims = (bundle.train_text_vectors @ x.T).toarray().ravel()
    if len(sims) == 0:
        return [], []
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]  # argpartition doesn't sort within the partition -- sort just these k
    return [int(i) for i in top], [float(sims[i]) for i in top]


def retrieve_nearest_op_sequence(bundle: ProgramModelBundle, text: str, k: int = 5) -> tuple[tuple[str, ...], float]:
    """The retrieval half of the RAG path: finds the `k` closest training
    examples by TF-IDF cosine similarity and returns the MAJORITY-VOTE
    op-sequence among them (ties broken toward the higher-similarity
    occurrence) -- never a hallucinated sequence, always one actually
    observed in training -- plus the single best match's similarity
    score (for the unchanged OOD disclosure). predict_program_rag() uses
    this sequence's SHAPE in place of the per-slot classifiers' own
    independent guess; the trained per-argument regressors still
    determine every actual numeric/categorical value, exactly as in the
    non-retrieval path.

    Confirmed empirically (a classifier-robustness probe across every
    generic-composition pattern, see data/program_dataset.py's
    _with_positional_noise for the finding that motivated this) that the
    per-slot classifiers' own op CHOICE is the least robust part of this
    architecture under realistic phrasing drift. Also confirmed empirically
    that PURE 1-NN retrieval (k=1) overcorrects: it fixed the fragile
    generic/hand-written cases but measurably regressed the 6 well-
    trodden part families (exact_op_sequence_accuracy 0.916 -> 0.841 on
    synthetic_test), where a single nearest neighbor is noisier than the
    per-slot classifiers' own decision boundary fit across hundreds of
    examples per family. Majority vote across k=5 neighbors is a direct
    fix for that: for a dense, well-covered query it converges on the
    same consensus shape the classifier already learned; for a sparse or
    drifted query it still recovers the right shape because the top-k
    neighbors, even if individually a bit further away, are still real
    examples of the same underlying pattern more often than not."""
    if not bundle.train_op_sequences:
        raise ValueError("bundle has no stored train_op_sequences -- retrain to enable retrieval")
    idxs, sims = _top_k_neighbors(bundle, text, k)
    if not idxs:
        return (STOP,), 0.0

    votes: dict[tuple[str, ...], float] = {}
    for rank, idx in enumerate(idxs):
        seq = bundle.train_op_sequences[idx]
        # weight by similarity (not a plain count) so a close match outvotes
        # several merely-nearby ones, with rank as a tiny tiebreaker
        votes[seq] = votes.get(seq, 0.0) + sims[rank] + (1e-6 * (len(idxs) - rank))
    winner = max(votes, key=votes.get)
    return winner, sims[0]


def predict_program_rag(bundle: ProgramModelBundle, text: str) -> list:
    """predict_program(), but with the op-sequence SHAPE drawn from
    retrieval (retrieve_nearest_op_sequence()) rather than guessed
    independently slot-by-slot."""
    ops, _sim = retrieve_nearest_op_sequence(bundle, text)
    return predict_program(bundle, text, forced_ops=list(ops))


def predict_program_hybrid(bundle: ProgramModelBundle, text: str) -> tuple[list, bool]:
    """Arbitrates between the classifier's own independent op-sequence and
    RAG's retrieved one, rather than trusting either unconditionally.
    Returns (program, used_rag).

    Measured (this project's own honest 3-bucket eval, retrained bundle):
    pure RAG (k=5) beats the classifier-only path on hand_written (+12.5
    points) and generic_composition (+5.8 points) -- exactly the prompts
    with thin or novel phrasing, where RAG's "always a real observed
    sequence" guarantee matters -- but loses 3.3 points on synthetic_test,
    the 6 deeply-trained part families, where a classifier that has fit a
    boundary across hundreds of examples per family is less noisy than
    looking at only the 5 nearest points.

    The two methods AGREEING is itself a strong, free confidence signal:
    when they land on the same op-sequence, there is no tradeoff to
    arbitrate, so either result is equally trustworthy (this returns the
    already-computed classifier program rather than recomputing). When
    they DISAGREE, that disagreement is exactly the classifier's known
    failure signature -- it chose an op combination that doesn't
    correspond to any single real training example -- so RAG wins,
    PROVIDED the winning neighbor's own similarity to `text` clears
    bundle.config.ood_similarity_threshold (the same floor
    nearest_neighbor_similarity() already uses to flag a prediction
    out_of_distribution elsewhere -- reused here, not a second invented
    number).

    Confirmed real gap this closes (a mission-brief audit, not
    hypothetical): disagreement alone used to force RAG's sequence
    unconditionally, with no floor on how similar that "winning" neighbor
    actually was. For a text with no genuinely close training neighbor at
    all, the winner is only the LEAST-dissimilar of five already-poor
    matches -- forcing its op-sequence onto an unrelated text is exactly
    how an unrequested cylinder or rounded block (an op present in some
    training pattern that happens to share surface vocabulary with the
    query -- e.g. "round"/"block"/"smooth" -- not the shape actually
    requested) could reach a built model. Traced to real training
    patterns that can do this: data/program_dataset.py's
    _pattern_round_box_edges, its boss-with-bore patterns, and its
    "smoothed... into a softer, more organic form" wording.

    Below that floor, the classifier's own (unforced) sequence is used
    instead -- not because it's guaranteed correct, but because it's at
    least derived from THIS text's own per-slot read, not a forced
    transplant from an unrelated training example. Either way,
    in_distribution (predict_program_checked) is computed from this same
    real nearest-neighbor similarity, so a low-confidence result is still
    honestly disclosed, never silently promoted by this gate."""
    classifier_program = predict_program(bundle, text)
    classifier_ops = tuple(c.op for c in classifier_program)
    rag_ops, rag_sim = retrieve_nearest_op_sequence(bundle, text)
    if classifier_ops == tuple(rag_ops):
        return classifier_program, False
    if rag_sim < bundle.config.ood_similarity_threshold:
        return classifier_program, False
    return predict_program(bundle, text, forced_ops=list(rag_ops)), True


def nearest_neighbor_similarity(bundle: ProgramModelBundle, text: str) -> float:
    """Cosine similarity between `text`'s TF-IDF vector and the single
    closest training example. This is an honest "have I ever seen anything
    resembling this text" signal -- deliberately NOT the slot classifiers'
    own predict_proba, because a softmax classifier is well known to stay
    confidently wrong on inputs far from anything it was trained on (proven
    concretely here: "a spaceship" and "a coffee mug" -- zero token overlap
    with training vocabulary -- produced byte-identical, high-confidence,
    wrong flange-shaped predictions before this check existed). TfidfVectorizer
    L2-normalizes every row by default, so cosine similarity between two
    already-unit-length vectors is a plain dot product -- no extra
    normalization needed here.

    A LOW score does not mean predict_program's output is geometrically
    invalid -- Sandbox.execute() will typically still report status="ok",
    since nothing about a confident-but-baseless guess trips the executor's
    structural or watertightness checks. It means the prediction has no
    real support in training data and should be disclosed as a guess, not
    presented as understanding the prompt -- see ProgramPrediction /
    predict_program_checked() below for the disclosed-guess wrapper."""
    _idx, sim = _nearest_neighbor_index_and_similarity(bundle, text)
    return sim


@dataclass
class ProgramPrediction:
    program: list  # list[sandbox.ToolCall]
    nearest_neighbor_similarity: float
    in_distribution: bool  # nearest_neighbor_similarity >= bundle.config.ood_similarity_threshold


def predict_program_checked(bundle: ProgramModelBundle, text: str) -> ProgramPrediction:
    """predict_program_rag() plus the honest out-of-distribution disclosure
    from nearest_neighbor_similarity() -- this is the entry point an
    application layer should call for arbitrary/untrusted user prompts,
    the same way Sandbox.execute()'s status field is meant to be checked
    before a caller trusts a predicted program's geometry. Kept as a
    SEPARATE, never-merged signal from execution status, same discipline
    program_eval.py already applies to its 3 report buckets: in_distribution
    answers "have I seen anything like this text," status answers "did the
    predicted program build a valid mesh" -- neither implies the other.

    Uses the agreement-gated HYBRID by default (predict_program_hybrid) --
    not pure RAG and not the classifier alone -- see that function's
    docstring for the measured reasoning: it keeps RAG's gains on novel/
    thin-density prompts while recovering the classifier's own strength on
    the 6 deeply-trained part families, rather than trading one off
    against the other unconditionally."""
    if bundle.train_op_sequences:
        program, _used_rag = predict_program_hybrid(bundle, text)
        sim = nearest_neighbor_similarity(bundle, text)
    else:
        # older bundle saved before retrieval existed -- fall back to the
        # original independent per-slot path rather than hard-failing.
        program = predict_program(bundle, text)
        sim = nearest_neighbor_similarity(bundle, text)
    return ProgramPrediction(
        program=program,
        nearest_neighbor_similarity=sim,
        in_distribution=sim >= bundle.config.ood_similarity_threshold,
    )
