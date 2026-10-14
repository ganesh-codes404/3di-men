"""Env-var-driven configuration, matching llm_decompose.py's own
MODEL_NAME/MODEL_PATH/MAX_CONTEXT pattern -- every setting has a sensible
default, nothing requires editing Python to reconfigure, and the whole
subsystem can be switched off without touching any call site (section 18:
"Make every optional subsystem disableable through configuration")."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CAD_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class KnowledgeConfig:
    enabled: bool = os.environ.get("KNOWLEDGE_ENABLED", "1") not in ("0", "false", "False", "")

    # Outside application source code (section 4), gitignored (see .gitignore)
    # since ingested documents may be the user's own licensed material.
    store_dir: Path = Path(os.environ.get("KNOWLEDGE_STORE_DIR", str(_CAD_ROOT / "knowledge_store")))

    # Chunking. Defaults chosen for a TF-IDF retriever reading into an
    # already context-constrained local LLM (llm_decompose.py's own system
    # prompt is ~3900 tokens against an 8192 budget -- see that file's
    # MAX_CONTEXT comment) -- chunks must stay small enough that a handful
    # of them is a reasonable addition to the user prompt, not a second
    # system-prompt-sized block. ~180 words (~1000 chars) is comfortably
    # under 512 tokens even for dense technical text; 15% overlap keeps a
    # procedure's lead-in from being orphaned at a chunk boundary without
    # meaningfully inflating storage.
    chunk_target_chars: int = int(os.environ.get("KNOWLEDGE_CHUNK_CHARS", "1000"))
    chunk_overlap_chars: int = int(os.environ.get("KNOWLEDGE_CHUNK_OVERLAP_CHARS", "150"))
    chunk_min_chars: int = int(os.environ.get("KNOWLEDGE_CHUNK_MIN_CHARS", "40"))  # drop near-empty scraps

    # TF-IDF embedding (see embeddings.py) -- mirrors program_model.py's own
    # TfidfVectorizer(max_features=..., ngram_range=...) tuning, not a
    # separately-invented configuration surface.
    tfidf_max_features: int = int(os.environ.get("KNOWLEDGE_TFIDF_MAX_FEATURES", "20000"))
    tfidf_ngram_range: tuple[int, int] = (1, 2)

    # Retrieval budget. retrieval.py never returns more than top_k chunks,
    # and ALSO stops once max_context_chars worth of text has been
    # accumulated -- a query that matches many long chunks still gets a
    # bounded answer. ~2000 chars (~500 tokens) for the whole retrieved
    # block is deliberately conservative against the same context-overflow
    # failure mode llm_decompose.py already hit once this session.
    default_top_k: int = int(os.environ.get("KNOWLEDGE_TOP_K", "5"))
    min_score: float = float(os.environ.get("KNOWLEDGE_MIN_SCORE", "0.05"))
    max_context_chars: int = int(os.environ.get("KNOWLEDGE_MAX_CONTEXT_CHARS", "2000"))

    # OCR fallback for scanned PDFs (section 4: "optional fallback...do not
    # make OCR mandatory"). Off by default -- pytesseract + a system
    # tesseract binary is a real, non-Python-only dependency this project
    # has never required; a scanned PDF is reported as unprocessable
    # instead (see document_parser.py) unless this is explicitly enabled.
    enable_ocr: bool = os.environ.get("KNOWLEDGE_ENABLE_OCR", "0") not in ("0", "false", "False", "")

    @property
    def documents_dir(self) -> Path:
        return self.store_dir / "documents"

    @property
    def index_path(self) -> Path:
        return self.store_dir / "index" / "knowledge_index.joblib"


DEFAULT_CONFIG = KnowledgeConfig()
