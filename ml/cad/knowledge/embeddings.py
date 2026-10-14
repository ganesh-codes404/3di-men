"""The embedding backend behind knowledge retrieval -- a thin wrapper
around scikit-learn's TfidfVectorizer, the SAME embedding mechanism
program_model.py already uses in production (see that file's
train_program_model -- TfidfVectorizer(max_features=..., ngram_range=...)).

Kept behind an explicit interface (EmbeddingBackend) specifically so a
future neural backend (sentence-transformers or similar) can be swapped in
without touching index.py/retrieval.py -- not because one is currently
needed. See the knowledge/__init__.py module docstring for why TF-IDF is
the right DEFAULT on this project's hardware (CPU-only, no GPU offload in
use anywhere, ~0.1-0.4 tok/s local LLM generation already measured)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from knowledge.config import KnowledgeConfig


class EmbeddingBackend(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None: ...

    @abstractmethod
    def transform(self, texts: list[str]) -> Any: ...

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> Any: ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool: ...

    def feature_names(self) -> list[str] | None:
        """Per-term vocabulary, for match-transparency (retrieval.py's
        matched_terms). None for a backend that has no such concept (e.g. a
        future dense neural embedding) -- retrieval degrades to an empty
        matched_terms tuple in that case, never an error."""
        return None


class TfidfEmbeddingBackend(EmbeddingBackend):
    def __init__(self, config: KnowledgeConfig | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        config = config or KnowledgeConfig()
        self._vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range,
            stop_words="english",
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("cannot fit an embedding backend on zero texts")
        self._vectorizer.fit(texts)
        self._fitted = True

    def fit_transform(self, texts: list[str]) -> Any:
        if not texts:
            raise ValueError("cannot fit an embedding backend on zero texts")
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix

    def transform(self, texts: list[str]) -> Any:
        if not self._fitted:
            raise ValueError("embedding backend has not been fit yet")
        return self._vectorizer.transform(texts)

    def vocabulary_size(self) -> int:
        return len(self._vectorizer.vocabulary_) if self._fitted else 0

    def feature_names(self) -> list[str] | None:
        if not self._fitted:
            return None
        return list(self._vectorizer.get_feature_names_out())
