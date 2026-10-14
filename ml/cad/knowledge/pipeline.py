"""The ONE integration point between the knowledge subsystem and the real
prompt-to-geometry pipeline (llm_decompose.py's decompose_prompt). Kept in
its own module rather than folded into llm_decompose.py directly so that
file's existing, already carefully-tuned prompt logic (see its own
MAX_CONTEXT-overflow postmortem comment) gets exactly one new call added to
it, and so this subsystem's own heavier imports (scikit-learn, lazily
fitz/bs4) never load at all when KNOWLEDGE_ENABLED=0.

retrieve_context_for_prompt() is the single function llm_decompose.py
calls. It NEVER raises and returns "" (meaning "add nothing") for every one
of: the subsystem disabled, no index on disk yet, an empty index, or a
query that matched nothing above the relevance floor -- all exactly the
same outcome to the caller, by design, so decompose_prompt needs no special
casing and behaves identically to how it did before this subsystem
existed when there's nothing useful to add (section 3's graceful-
degradation requirement; section 18's "every optional subsystem
disableable through configuration")."""

from __future__ import annotations

from knowledge.config import KnowledgeConfig
from knowledge.retrieval import format_context_block, retrieve

_cached_index = None
_cached_index_mtime: float | None = None
_cached_index_config_key: tuple | None = None


def _config_key(config: KnowledgeConfig) -> tuple:
    return (str(config.store_dir), config.default_top_k, config.min_score, config.max_context_chars)


def _get_index(config: KnowledgeConfig):
    """Lazily loads + caches the index for this process, reloading only
    when the on-disk index file's mtime changes (e.g. the CLI ingested a
    new document in a separate process since this one last loaded it) --
    not a fresh joblib.load() on every single prompt."""
    global _cached_index, _cached_index_mtime, _cached_index_config_key
    from knowledge.index import KnowledgeIndex  # deferred: pulls in sklearn, only when actually needed

    if not config.index_path.exists():
        return None
    mtime = config.index_path.stat().st_mtime
    key = _config_key(config)
    if _cached_index is None or _cached_index_mtime != mtime or _cached_index_config_key != key:
        _cached_index = KnowledgeIndex.load(config=config)
        _cached_index_mtime = mtime
        _cached_index_config_key = key
    return _cached_index


def retrieve_context_for_prompt(prompt: str, config: KnowledgeConfig | None = None) -> str:
    config = config or KnowledgeConfig()
    if not config.enabled:
        return ""
    try:
        index = _get_index(config)
        if index is None or not index.chunks:
            return ""
        results = retrieve(index, prompt, config=config)
        return format_context_block(results)
    except Exception:
        # A knowledge-subsystem failure (a corrupt index file, a missing
        # optional dependency, anything) must never break prompt
        # decomposition -- that pipeline worked without this subsystem
        # before, and must keep working exactly the same way if this
        # subsystem misbehaves.
        return ""
