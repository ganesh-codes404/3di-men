"""User-facing entry point for the learned (scikit-learn) prompt parser --
the trained-model counterpart to nlp/prompt_parser.py's regex parser.

Deliberately NOT wired into inference.py's default path yet: the regex
parser stays the default until real A/B accuracy numbers exist (see
parser_eval.py / scripts/eval_parser_model.py). This module exists so both
parsers can be called through an equally simple interface for that
comparison, and so a future phase can promote this one once it's earned it.
"""

from __future__ import annotations

from pathlib import Path

from parser_model import LearnedParseResult, ParserModelBundle, predict as _predict

_cached_bundle: ParserModelBundle | None = None
_cached_path: Path | None = None


def load_bundle(checkpoint_path: str | Path) -> ParserModelBundle:
    return ParserModelBundle.load(Path(checkpoint_path))


def predict(text: str, checkpoint_path: str | Path) -> LearnedParseResult:
    """Loads (and caches) the bundle at checkpoint_path, then predicts.
    For batch use (e.g. eval scripts), prefer loading the bundle once with
    load_bundle() and calling parser_model.predict(bundle, text) directly."""
    global _cached_bundle, _cached_path
    path = Path(checkpoint_path)
    if _cached_bundle is None or _cached_path != path:
        _cached_bundle = load_bundle(path)
        _cached_path = path
    return _predict(_cached_bundle, text)
