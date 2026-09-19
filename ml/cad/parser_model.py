"""The proprietary, self-trained, locally-run prompt-understanding model
(Phase 4) -- predicts structured part-family + parameters directly, never
raw geometry. No external LLM API involved anywhere in this file.

Architecture choice: TF-IDF + scikit-learn classifiers/regressors, not a
neural net. Reasoning (see the plan doc / conversation this was designed in):
the task is low-dimensional structured prediction over small discrete grids
(random_mounting_plate/random_flange sample most fields from 3-5 choices),
which is a classical-ML-shaped problem, not one that benefits from learned
embeddings -- and it avoids adding a torch dependency to a package that
doesn't have one today.

Known, honest limitation (do not paper over this in eval output): because
numeric fields are drawn from small fixed grids during synthetic training, a
bag-of-words model's numeric predictions are closer to *recognizing a seen
value* than *extracting a novel one*. It will score well on the synthetic
test split and measurably worse on hand-written prompts with unseen numbers.
nlp/prompt_parser.py's regex `_find_number` already generalizes to any digit
string, which this model fundamentally cannot -- this model's real value-add
is part-family classification and categorical fields, not numeric
extraction. See parser_eval.py, which reports these separately rather than
collapsing them into one score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from part_families import FieldKind, get_part_family, list_part_families

_CATEGORICAL_KINDS: tuple[FieldKind, ...] = ("categorical", "count")


@dataclass
class ParserModelConfig:
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    seed: int = 0


@dataclass
class ParserModelBundle:
    vectorizer: Any  # sklearn.feature_extraction.text.TfidfVectorizer
    part_family_clf: Any  # sklearn classifier
    field_estimators: dict[str, dict[str, Any]]  # family -> field -> fitted estimator
    label_encoders: dict[str, dict[str, Any]]  # family -> categorical/count field -> LabelEncoder
    config: ParserModelConfig
    trained_on: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ParserModelBundle":
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return bundle


@dataclass
class LearnedParseResult:
    part_type: str
    params: Any  # MountingPlateParams | FlangeParams -- same union nlp.prompt_parser.ParseResult uses
    part_type_confidence: float
    field_confidences: dict[str, float | None]  # None for regression (numeric) fields


def _make_field_estimator(kind: FieldKind, seed: int):
    if kind in _CATEGORICAL_KINDS:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=seed)
    from sklearn.linear_model import Ridge

    return Ridge(random_state=seed)


def train_parser_model(dataset_root: Path, config: ParserModelConfig | None = None) -> tuple[ParserModelBundle, dict]:
    """Fits on the TRAIN split only. Returns (bundle, train_metrics) -- the
    metrics here are train-set fit diagnostics, NOT a claim of real accuracy;
    use parser_eval.py's evaluate_parser() against val/test/hand-written data
    for that."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    from data.prompt_store import read_split

    config = config or ParserModelConfig()
    train_samples = read_split(dataset_root, "train")
    if not train_samples:
        raise ValueError(f"no train samples found under {dataset_root} -- did generate_prompt_dataset.py run?")

    texts = [s.text for s in train_samples]
    vectorizer = TfidfVectorizer(max_features=config.tfidf_max_features, ngram_range=config.tfidf_ngram_range)
    X = vectorizer.fit_transform(texts)

    family_labels = [s.part_family for s in train_samples]
    part_family_clf = _make_field_estimator("categorical", config.seed)
    part_family_clf.fit(X, family_labels)

    field_estimators: dict[str, dict[str, Any]] = {}
    label_encoders: dict[str, dict[str, Any]] = {}
    train_metrics: dict[str, Any] = {"n_train": len(train_samples), "families": {}}

    for family in list_part_families():
        spec = get_part_family(family)
        family_mask = [i for i, s in enumerate(train_samples) if s.part_family == family]
        if not family_mask:
            continue
        X_family = X[family_mask]
        field_estimators[family] = {}
        label_encoders[family] = {}
        family_metrics: dict[str, float] = {}

        for f in spec.fields:
            y_raw = [train_samples[i].params[f.name] for i in family_mask]
            if f.kind in _CATEGORICAL_KINDS:
                encoder = LabelEncoder()
                y = encoder.fit_transform([str(v) for v in y_raw])
                label_encoders[family][f.name] = encoder
            else:
                y = [float(v) for v in y_raw]

            estimator = _make_field_estimator(f.kind, config.seed)
            estimator.fit(X_family, y)
            field_estimators[family][f.name] = estimator
            family_metrics[f.name] = float(estimator.score(X_family, y))  # train-set fit only

        train_metrics["families"][family] = family_metrics

    bundle = ParserModelBundle(
        vectorizer=vectorizer,
        part_family_clf=part_family_clf,
        field_estimators=field_estimators,
        label_encoders=label_encoders,
        config=config,
        trained_on={"dataset_root": str(dataset_root), "train_count": len(train_samples), "families": list_part_families()},
    )
    return bundle, train_metrics


def predict(bundle: ParserModelBundle, text: str) -> LearnedParseResult:
    X = bundle.vectorizer.transform([text])

    family_probs = bundle.part_family_clf.predict_proba(X)[0]
    family_idx = family_probs.argmax()
    part_type = bundle.part_family_clf.classes_[family_idx]
    part_type_confidence = float(family_probs[family_idx])

    spec = get_part_family(part_type)
    resolved: dict[str, Any] = {}
    field_confidences: dict[str, float | None] = {}

    for f in spec.fields:
        estimator = bundle.field_estimators[part_type][f.name]
        if f.kind in _CATEGORICAL_KINDS:
            encoder = bundle.label_encoders[part_type][f.name]
            probs = estimator.predict_proba(X)[0]
            idx = probs.argmax()
            value_str = encoder.inverse_transform([idx])[0]
            resolved[f.name] = f.python_type(value_str) if f.python_type is not str else value_str
            field_confidences[f.name] = float(probs[idx])
        else:
            resolved[f.name] = float(estimator.predict(X)[0])
            field_confidences[f.name] = None

    params = spec.params_cls(**resolved)
    return LearnedParseResult(
        part_type=part_type,
        params=params,
        part_type_confidence=part_type_confidence,
        field_confidences=field_confidences,
    )
