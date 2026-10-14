"""Section-aware chunking (section 5): packs a ParsedDocument's blocks
(already paragraph/heading-structured by document_parser.py) into chunks
bounded by KnowledgeConfig.chunk_target_chars, without splitting a block
(a paragraph, list item, or table cell) across a chunk boundary unless a
single block alone exceeds the target size. Concept-type and knowledge-
domain tagging are both closed-vocabulary keyword lookups against
schemas.py's authored taxonomy -- never free-text generation -- and default
to "general"/empty specifically so an unmatched chunk is honestly
unclassified rather than mis-tagged (section 5: "infer metadata
conservatively... do not fabricate")."""

from __future__ import annotations

import hashlib
import re

from knowledge.config import KnowledgeConfig
from knowledge.document_parser import ParsedBlock, ParsedDocument
from knowledge.schemas import Chunk, ConceptType

# Checked in order -- first match wins, so more specific/higher-confidence
# patterns are listed first. Each entry requires a reasonably distinctive
# phrase, not a single common word, to avoid over-tagging ordinary prose.
_CONCEPT_TYPE_PATTERNS: list[tuple[ConceptType, re.Pattern]] = [
    ("formula", re.compile(r"\bformula\b|\bequation\b|[A-Za-z]\w*\s*=\s*[-\w./() +*]+", re.IGNORECASE)),
    ("troubleshooting_procedure", re.compile(
        r"\bto (fix|resolve|avoid this|correct this)\b|\btroubleshoot|\bsolution:|\bif this happens\b",
        re.IGNORECASE,
    )),
    ("failure_mode", re.compile(
        r"\b(fails?|failure|can cause|risk of|common (mistake|error)|pitfall)\b", re.IGNORECASE
    )),
    ("manufacturing_rule", re.compile(
        r"\b(3d print(ing)?|CNC|injection mold|sheet.?metal|machin(e|ing)|draft angle|minimum wall thickness)\b",
        re.IGNORECASE,
    )),
    ("constraint", re.compile(
        r"\bmust (not )?(be|exceed)\b|\bminimum\b|\bmaximum\b|\btolerance\b|\bat least\b|\bno more than\b",
        re.IGNORECASE,
    )),
    ("geometric_operation", re.compile(
        r"\b(extrud|revolv|sweep|loft|boolean (union|difference|intersection)|fillet|chamfer|shell|offset)\w*\b",
        re.IGNORECASE,
    )),
    ("modeling_procedure", re.compile(
        r"^\s*(step\s*\d|\d+[.)])\s|\bfirst,|\bnext,|\bthen,|\bfollow these steps\b", re.IGNORECASE
    )),
    ("design_principle", re.compile(
        r"\b(design principle|rule of thumb|best practice|as a general rule)\b", re.IGNORECASE
    )),
    ("definition", re.compile(
        r"\bis defined as\b|\brefers to\b|\bis (a|an|the)\b.{0,60}\bterm\b|^[A-Z][\w -]{1,40}\s+is\s+(a|an|the)\s",
    )),
    ("example", re.compile(r"\bfor example\b|\be\.g\.|\bconsider a\b|\bcase study\b", re.IGNORECASE)),
]

# Section 6's own domain bullet lists, used directly as keyword banks --
# authored taxonomy, not inferred. A chunk may match zero, one, or several.
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "solid_modeling": (
        "extrusion", "revolution", "sweep", "loft", "boolean union", "boolean difference",
        "boolean intersection", "fillet", "chamfer", "shell", "offset", "surface operation",
        "mesh valid", "solid valid", "primitive",
    ),
    "parametric_cad": (
        "sketch", "geometric constraint", "dimensional constraint", "feature tree",
        "dependency relationship", "reference geometry", "feature order", "parameter update",
        "regenerat",
    ),
    "mechanical_design": (
        "shaft", "flange", "mounting plate", "bracket", "housing", "enclosure", "fastener",
        "boss", "bearing", "gear", "thread", "fit and clearance", "wall thickness",
    ),
    "product_modeling": (
        "silhouette", "proportion", "primary form", "secondary form", "tertiary form",
        "edge treatment", "surface continuity", "hard-surface", "visual hierarchy",
    ),
    "assembly_reasoning": (
        "component hierarchy", "coordinate frame", "local transform", "global transform",
        "attachment point", "alignment constraint", "mating surface", "interference",
        "assembly validation",
    ),
    "manufacturing": (
        "3d printing", "cnc machin", "injection mold", "sheet-metal", "sheet metal",
        "manufacturing tolerance", "material-aware",
    ),
}


def _infer_concept_type(text: str) -> ConceptType:
    for concept_type, pattern in _CONCEPT_TYPE_PATTERNS:
        if pattern.search(text):
            return concept_type
    return "general"


def _infer_domains(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(domain for domain, keywords in _DOMAIN_KEYWORDS.items() if any(kw in lowered for kw in keywords))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_oversized_block(text: str, target_chars: int) -> list[str]:
    """Only invoked for a single block that alone exceeds target_chars (a
    rare, long un-paragraphed wall of text) -- splits at sentence
    boundaries as a fallback so a chunk size limit is still an actual
    limit, never on an ordinary block."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + 1 + len(sent) > target_chars:
            pieces.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        pieces.append(current)
    return pieces or [text]


def chunk_document(
    parsed: ParsedDocument, document_id: str, source_path: str, config: KnowledgeConfig | None = None,
    source_id: str | None = None, source_url: str | None = None, source_revision: str | None = None,
) -> list[Chunk]:
    config = config or KnowledgeConfig()
    raw_pieces: list[tuple[str, str | None, int | None]] = []  # (text, section, page)
    for block in parsed.blocks:
        if len(block.text) > config.chunk_target_chars:
            for piece in _split_oversized_block(block.text, config.chunk_target_chars):
                raw_pieces.append((piece, block.section, block.page))
        else:
            raw_pieces.append((block.text, block.section, block.page))

    chunks: list[Chunk] = []
    buffer_parts: list[str] = []
    buffer_section: str | None = None
    buffer_pages: list[int] = []
    buffer_len = 0

    def overlap_tail(text: str) -> str:
        if config.chunk_overlap_chars <= 0 or len(text) <= config.chunk_overlap_chars:
            return ""
        tail = text[-config.chunk_overlap_chars :]
        # start the overlap at a word boundary so it reads cleanly
        space_idx = tail.find(" ")
        return tail[space_idx + 1 :] if space_idx != -1 else tail

    def flush():
        nonlocal buffer_parts, buffer_section, buffer_pages, buffer_len
        text = "\n\n".join(buffer_parts).strip()
        if len(text) >= config.chunk_min_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{len(chunks)}",
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=text,
                    title=parsed.title,
                    section=buffer_section,
                    page_start=min(buffer_pages) if buffer_pages else None,
                    page_end=max(buffer_pages) if buffer_pages else None,
                    source_path=source_path,
                    content_hash=_content_hash(text),
                    concept_type=_infer_concept_type(text),
                    domains=_infer_domains(text),
                    source_id=source_id, source_url=source_url, source_revision=source_revision,
                )
            )
        elif text and chunks:
            # too small to stand alone (section 5: avoid near-empty chunks)
            # -- folded into the previous chunk rather than dropped.
            prev = chunks[-1]
            merged_text = f"{prev.text}\n\n{text}"
            chunks[-1] = Chunk(
                chunk_id=prev.chunk_id, document_id=prev.document_id, chunk_index=prev.chunk_index,
                text=merged_text, title=prev.title, section=prev.section,
                page_start=prev.page_start,
                page_end=max(filter(None, [prev.page_end, *buffer_pages]), default=prev.page_end),
                source_path=prev.source_path, content_hash=_content_hash(merged_text),
                concept_type=_infer_concept_type(merged_text), domains=_infer_domains(merged_text),
                source_id=prev.source_id, source_url=prev.source_url, source_revision=prev.source_revision,
            )
        buffer_parts, buffer_pages, buffer_len = [], [], 0

    for text, section, page in raw_pieces:
        section_changed = buffer_parts and section != buffer_section
        would_overflow = buffer_len + len(text) > config.chunk_target_chars
        if buffer_parts and (section_changed or would_overflow):
            carried = overlap_tail(buffer_parts[-1]) if would_overflow and not section_changed else ""
            flush()
            if carried:
                buffer_parts = [carried]
                buffer_len = len(carried)
        buffer_section = section
        buffer_parts.append(text)
        if page is not None:
            buffer_pages.append(page)
        buffer_len += len(text)
    flush()

    return chunks
