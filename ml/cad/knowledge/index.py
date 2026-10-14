"""Persistent chunk index -- joblib-backed, mirroring program_model.py's
own ProgramModelBundle.save/load pattern exactly (see that class's
docstring) rather than inventing a second persistence convention in this
project.

Re-embeds the WHOLE corpus on every add/remove rather than doing
incremental vector updates. This is a deliberate simplicity choice (section
3: "a small, transparent implementation" over cleverness) that's cheap at
this subsystem's actual expected scale -- a personal reference library of
dozens to a few hundred documents, a few thousand chunks -- where a full
TF-IDF refit is a sub-second scikit-learn call, not a bottleneck worth
engineering around. If this subsystem's corpus ever grows large enough for
that to matter, both the embedding backend (embeddings.py) and this index
are built behind narrow interfaces specifically so the strategy can change
without touching retrieval.py or ingestion.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knowledge.config import KnowledgeConfig
from knowledge.embeddings import EmbeddingBackend, TfidfEmbeddingBackend
from knowledge.schemas import Chunk, DocumentRecord


@dataclass
class KnowledgeIndex:
    config: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    embedding_backend: EmbeddingBackend = field(default_factory=TfidfEmbeddingBackend)
    chunk_vectors: Any = None  # sparse matrix, rows aligned 1:1 with self.chunks

    def has_document(self, document_id: str) -> bool:
        return document_id in self.documents

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(self.documents.values(), key=lambda d: d.imported_at)

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.document_id == document_id]

    def add_document(self, record: DocumentRecord, chunks: list[Chunk]) -> None:
        if self.has_document(record.document_id):
            raise ValueError(f"document {record.document_id} is already indexed -- remove it first to re-add")
        self.documents[record.document_id] = record
        self.chunks.extend(chunks)
        self._rebuild_vectors()

    def remove_document(self, document_id: str) -> bool:
        """True if the document was present and removed. False (a no-op,
        not an error) if it wasn't indexed -- removal of something already
        absent is not a failure condition (section 4's "remove a document"
        workflow should be safely re-runnable)."""
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        self._rebuild_vectors()
        return True

    def _rebuild_vectors(self) -> None:
        self.embedding_backend = TfidfEmbeddingBackend(self.config)
        if not self.chunks:
            self.chunk_vectors = None
            return
        self.chunk_vectors = self.embedding_backend.fit_transform([c.text for c in self.chunks])

    def rebuild(self) -> None:
        """Section 4's "rebuild the index" workflow -- re-derives
        chunk_vectors from the chunks currently held, without touching
        documents/chunks themselves. Useful after a config change (e.g. a
        different tfidf_max_features) or to recover from a suspected
        vector/chunk misalignment."""
        self._rebuild_vectors()

    def integrity_check(self) -> list[str]:
        """Section 4's "integrity checks" requirement. Returns a list of
        problems found -- empty means healthy. Deterministic structural
        checks only, never a judgment about CONTENT quality."""
        issues: list[str] = []
        chunk_doc_ids = {c.document_id for c in self.chunks}
        missing_docs = chunk_doc_ids - set(self.documents.keys())
        for doc_id in missing_docs:
            issues.append(f"chunks reference document_id {doc_id!r} which is not in self.documents")

        for doc_id, record in self.documents.items():
            actual = len(self.chunks_for_document(doc_id))
            if actual != record.chunk_count:
                issues.append(
                    f"document {doc_id!r} ({record.filename}): chunk_count={record.chunk_count} "
                    f"but {actual} chunks are actually indexed"
                )

        if self.chunks:
            if self.chunk_vectors is None:
                issues.append(f"{len(self.chunks)} chunks are indexed but chunk_vectors is None")
            elif self.chunk_vectors.shape[0] != len(self.chunks):
                issues.append(
                    f"chunk_vectors has {self.chunk_vectors.shape[0]} rows but there are {len(self.chunks)} chunks"
                )
        elif self.chunk_vectors is not None:
            issues.append("no chunks are indexed but chunk_vectors is not None")

        seen_chunk_ids: set[str] = set()
        for c in self.chunks:
            if c.chunk_id in seen_chunk_ids:
                issues.append(f"duplicate chunk_id {c.chunk_id!r}")
            seen_chunk_ids.add(c.chunk_id)

        return issues

    def save(self, path: Path | None = None) -> None:
        import joblib

        path = path or self.config.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | None = None, config: KnowledgeConfig | None = None) -> "KnowledgeIndex":
        import joblib

        config = config or KnowledgeConfig()
        path = path or config.index_path
        if not path.exists():
            return KnowledgeIndex(config=config)  # an empty, valid index -- not an error
        return joblib.load(path)
