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
    "translate": [("target", "handle_offset"), ("dx", "numeric"), ("dy", "numeric"), ("dz", "numeric")],
    "rotate": [("target", "handle_offset"), ("axis", "categorical"), ("angle_deg", "numeric")],
    "drill_hole": [("target", "handle_offset"), ("radius", "numeric"), ("depth", "numeric"),
                    ("center_x", "numeric"), ("center_y", "numeric"), ("center_z", "numeric"), ("axis", "categorical")],
    "drill_hole_pattern": [("target", "handle_offset"), ("pattern", "categorical"), ("count", "count"),
                            ("radius", "numeric"), ("depth", "numeric"), ("axis", "categorical"),
                            ("margin", "numeric"), ("circle_diameter", "numeric"), ("z", "numeric")],
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

    bundle = ProgramModelBundle(
        vectorizer=vectorizer, slot_op_classifiers=slot_op_classifiers, slot_op_label_encoder=op_encoder,
        arg_estimators=arg_estimators, arg_label_encoders=arg_label_encoders, config=config,
        train_text_vectors=X_text_sparse,
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


def predict_program(bundle: ProgramModelBundle, text: str) -> list:
    """K-slot autoregressive loop: predicts one op per slot (features =
    TF-IDF + summary of the model's OWN previously predicted slots, not
    ground truth -- this is real inference, unlike training's teacher
    forcing), stopping at STOP or finalize."""
    import numpy as np

    from sandbox import ToolCall

    X_text = bundle.vectorizer.transform([text]).toarray()[0]
    op_encoder = bundle.slot_op_label_encoder
    predicted_ops: list[str] = []
    program: list[ToolCall] = []

    for slot in range(bundle.config.k_slots):
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

        full_args = _unflatten_args(op, args, slot) if ARG_SCHEMA[op] else {}
        program.append(ToolCall(op=op, args=full_args))

        if op == "finalize":
            break

    return program


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
    if bundle.train_text_vectors is None:
        raise ValueError("bundle has no stored train_text_vectors -- retrain to enable OOD detection")
    x = bundle.vectorizer.transform([text])
    sims = (bundle.train_text_vectors @ x.T).toarray().ravel()
    return float(sims.max()) if len(sims) else 0.0


@dataclass
class ProgramPrediction:
    program: list  # list[sandbox.ToolCall]
    nearest_neighbor_similarity: float
    in_distribution: bool  # nearest_neighbor_similarity >= bundle.config.ood_similarity_threshold


def predict_program_checked(bundle: ProgramModelBundle, text: str) -> ProgramPrediction:
    """predict_program() plus the honest out-of-distribution disclosure
    from nearest_neighbor_similarity() -- this is the entry point an
    application layer should call for arbitrary/untrusted user prompts,
    the same way Sandbox.execute()'s status field is meant to be checked
    before a caller trusts a predicted program's geometry. Kept as a
    SEPARATE, never-merged signal from execution status, same discipline
    program_eval.py already applies to its 3 report buckets: in_distribution
    answers "have I seen anything like this text," status answers "did the
    predicted program build a valid mesh" -- neither implies the other."""
    return ProgramPrediction(
        program=predict_program(bundle, text),
        nearest_neighbor_similarity=(sim := nearest_neighbor_similarity(bundle, text)),
        in_distribution=sim >= bundle.config.ood_similarity_threshold,
    )
