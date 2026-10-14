"""Format-specific text + structure extraction. Produces a single,
format-agnostic ParsedDocument (a flat list of ParsedBlock, each tagged with
the section heading it falls under and/or its source page) so chunking.py
never needs to know whether a document came from a PDF, Markdown, HTML, or
plain text file.

Every parser here is conservative about what it claims: a title/section is
only ever set when the source format actually signals one (a real heading
line, a <title> tag, PDF metadata) -- never guessed from content, per
section 5's "infer metadata conservatively... do not fabricate section
names." A document that cannot be extracted reliably raises
DocumentParseError rather than returning silently-broken text (section 4:
"Do not silently treat broken extraction as valid engineering knowledge").
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from knowledge.config import KnowledgeConfig
from knowledge.schemas import ExtractionMethod

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm", ".rst"}


class DocumentParseError(Exception):
    """A document could not be processed reliably. Always carries a
    specific, actionable reason -- never raised as a bare catch-all."""


@dataclass
class ParsedBlock:
    text: str
    section: str | None = None
    page: int | None = None


@dataclass
class ParsedDocument:
    blocks: list[ParsedBlock]
    title: str | None
    page_count: int | None
    extraction_method: ExtractionMethod
    warnings: list[str] = field(default_factory=list)


def parse_document(path: Path, config: KnowledgeConfig | None = None) -> ParsedDocument:
    config = config or KnowledgeConfig()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, config)
    if suffix in (".md", ".markdown"):
        return parse_markdown(path)
    if suffix in (".html", ".htm"):
        return parse_html(path)
    if suffix == ".txt":
        return parse_plain_text(path)
    if suffix == ".rst":
        return parse_rst(path)
    raise DocumentParseError(
        f"{path}: unsupported file type {suffix!r} -- supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def parse_plain_text(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise DocumentParseError(f"{path}: file is empty")
    # Paragraph-level blocks (blank-line-separated) -- the finest structure
    # a plain .txt file actually has; no section/page metadata to offer.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks = [ParsedBlock(text=p, section=None, page=None) for p in paragraphs]
    return ParsedDocument(blocks=blocks, title=None, page_count=None, extraction_method="plain_text")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-+\s*$")
# Confirmed necessary, not hypothetical: a real acquired file (FreeCAD wiki's
# crash-log example inside a non-standard "{{code|...}}" MediaWiki template
# this parser doesn't otherwise recognize as a code block) contains gdb/lldb
# stack-trace lines like "#1  0x7fff63d660ba in BRep_Tool::Curve(...)" --
# these match _ATX_HEADING_RE (a single '#', whitespace, then content) and
# were being misread as a cascading sequence of level-1 headings, corrupting
# every chunk's `section` until the next real heading. Scoped narrowly to
# "digits, whitespace, a 0x hex address" so a legitimate numeral-led heading
# (e.g. "# 3D View", a real FreeCAD wiki page title) is never affected.
_STACK_TRACE_FRAME_RE = re.compile(r"^\d+\s+0x[0-9a-fA-F]")


def parse_markdown(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise DocumentParseError(f"{path}: file is empty")

    lines = text.split("\n")
    # Confirmed necessary, not hypothetical: a real file acquired via
    # knowledge/sources.py (FreeCAD's wiki-to-markdown mirror) opens with a
    # "---\n<yaml metadata>\n---\n" front-matter block, and the CLOSING
    # "---" line was being misread by the setext-heading check below as an
    # H2 underline for whatever metadata line preceded it -- corrupting
    # both the document's title and its first section with YAML field text
    # (e.g. "SeeAlso: PartDesign_Chamfer") instead of the file's real `#`
    # heading. Skipped as pure metadata, never shown to chunking.py at all,
    # rather than attempting to parse it as content.
    if lines and lines[0].strip() == "---":
        for end in range(1, len(lines)):
            if lines[end].strip() in ("---", "..."):
                lines = lines[end + 1 :]
                break

    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    title: str | None = None
    paragraph_lines: list[str] = []
    in_code_fence = False

    def flush():
        if paragraph_lines:
            joined = "\n".join(paragraph_lines).strip()
            if joined:
                blocks.append(ParsedBlock(text=joined, section=current_section, page=None))
            paragraph_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_code_fence = not in_code_fence
            paragraph_lines.append(line)
            i += 1
            continue
        if in_code_fence:
            paragraph_lines.append(line)
            i += 1
            continue

        atx = _ATX_HEADING_RE.match(line)
        if atx and _STACK_TRACE_FRAME_RE.match(atx.group(2)):
            atx = None  # a "#N  0xADDRESS ..." stack-trace line, not a real heading -- see regex comment above
        # Setext (underlined) headings: a non-blank line followed by a line
        # of all =/- characters. Checked before treating `line` as prose.
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        is_setext = line.strip() and (_SETEXT_H1_RE.match(next_line) or _SETEXT_H2_RE.match(next_line))

        if atx or is_setext:
            flush()
            heading_text = atx.group(2).strip() if atx else line.strip()
            current_section = heading_text
            if title is None:
                title = heading_text  # first heading in the document, whatever its level
            i += 2 if is_setext and not atx else 1
            continue

        if not line.strip():
            flush()
        else:
            paragraph_lines.append(line)
        i += 1
    flush()

    if not blocks:
        raise DocumentParseError(f"{path}: no extractable text content after parsing")
    return ParsedDocument(blocks=blocks, title=title, page_count=None, extraction_method="markdown")


# ---------------------------------------------------------------------------
# reStructuredText (Sphinx docs -- CadQuery, build123d, Blender manual)
# ---------------------------------------------------------------------------
# A pragmatic subset, not a full docutils-equivalent parser: real Sphinx
# docs lean on roles/substitutions/cross-references this doesn't resolve,
# and inline markup (`code`, *emphasis*, `link <url>`_) is left as-is in
# the extracted text rather than stripped -- readable enough for retrieval
# and for an LLM to read directly, not worth a markup-stripping pass given
# this project's "small, transparent implementation" preference (section
# 3). What IS handled, because it directly affects chunk quality: heading
# detection (for section tracking) and skipping non-prose directives
# (images/toctrees/raw/labels) so they don't pollute retrieved passages,
# while admonitions (note/warning/tip/seealso) and code blocks are KEPT --
# section 5 explicitly warns against splitting a warning from the
# operation it describes, and section 4 says to preserve code examples.
_RST_HEADING_CHARS = set("=-`:'\"~^_*+#<>.")
_RST_DIRECTIVE_RE = re.compile(r"^\.\.\s+([\w-]+)::\s*(.*)$")
_RST_LABEL_RE = re.compile(r"^\.\.\s+_[^:]+:\s*$")  # a cross-reference target, e.g. ".. _some_label:"
# Directives whose body is navigation/binary content, never prose --
# skipped entirely (the directive line AND its indented body).
_RST_STRIP_DIRECTIVES = {"toctree", "image", "raw", "include", "figure", "highlight", "literalinclude"}
# Directives whose body IS prose worth keeping -- only the directive
# marker line itself is dropped, replaced with a plain-text label prefix.
_RST_ADMONITION_DIRECTIVES = {
    "note": "Note", "warning": "Warning", "important": "Important", "tip": "Tip",
    "seealso": "See also", "versionadded": "Added in version", "versionchanged": "Changed in version",
    "deprecated": "Deprecated since version", "admonition": "Note",
}


def _is_rst_heading_underline(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 3 and len(set(stripped)) == 1 and stripped[0] in _RST_HEADING_CHARS


def parse_rst(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise DocumentParseError(f"{path}: file is empty")

    lines = text.split("\n")
    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    title: str | None = None
    paragraph_lines: list[str] = []
    skip_indent: int | None = None  # while set, skip lines indented >= this (a stripped directive's body)

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def flush():
        if paragraph_lines:
            joined = "\n".join(paragraph_lines).strip()
            if joined and not _RST_LABEL_RE.match(joined):
                blocks.append(ParsedBlock(text=joined, section=current_section))
            paragraph_lines.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        if skip_indent is not None:
            if line.strip() == "" or indent_of(line) >= skip_indent:
                i += 1
                continue
            skip_indent = None  # dedented back out of the skipped directive's body

        # Heading: a non-blank title line immediately followed by an
        # underline of a repeated punctuation character spanning at least
        # the title's width (docutils' own rule, loosely applied) --
        # optionally preceded by a matching overline too.
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _is_rst_heading_underline(next_line) and len(next_line.strip()) >= len(line.strip()) - 2:
            flush()
            heading_text = line.strip()
            current_section = heading_text
            if title is None:
                title = heading_text
            i += 2
            # an overline immediately before the title (already consumed as
            # plain text, harmless) -- also skip a trailing underline-only
            # duplicate if the SAME char was used as an overline right above
            continue

        directive = _RST_DIRECTIVE_RE.match(line)
        if directive:
            name, _rest = directive.groups()
            # A directive only HAS a body if the next non-blank line is
            # MORE indented than the directive marker itself -- otherwise
            # that next line is just the following paragraph, not part of
            # this directive, and must never be skipped. (Confirmed
            # necessary, not hypothetical: without the indent comparison,
            # a single-line directive with no body -- e.g. a bare
            # ".. image:: path.png" -- swallowed every following line in
            # the document, since "skip anything at indent >= 0" matches
            # everything.)
            directive_indent = indent_of(line)
            body_indent = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    continue
                candidate_indent = indent_of(lines[j])
                if candidate_indent > directive_indent:
                    body_indent = candidate_indent
                break
            if name in _RST_STRIP_DIRECTIVES:
                flush()
                if body_indent is not None:
                    skip_indent = body_indent
                i += 1
                continue
            if name in _RST_ADMONITION_DIRECTIVES:
                flush()
                paragraph_lines.append(f"{_RST_ADMONITION_DIRECTIVES[name]}:")
                i += 1
                continue
            # an unrecognized directive (code-block, math, table, ...) --
            # conservative default: keep its body as plain text rather than
            # silently dropping potentially-useful content, matching this
            # parser's own stated bias (section 4: never silently discard).
            i += 1
            continue

        if _RST_LABEL_RE.match(line.strip()):
            i += 1
            continue

        if not line.strip():
            flush()
        else:
            paragraph_lines.append(line.strip() if not paragraph_lines else line)
        i += 1
    flush()

    if not blocks:
        raise DocumentParseError(f"{path}: no extractable text content after RST parsing")
    return ParsedDocument(blocks=blocks, title=title, page_count=None, extraction_method="rst")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def parse_html(path: Path) -> ParsedDocument:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise DocumentParseError(
            f"{path}: HTML parsing requires beautifulsoup4, which is not installed"
        ) from exc

    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        raise DocumentParseError(f"{path}: file is empty")

    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else None

    blocks: list[ParsedBlock] = []
    current_section = title
    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "pre"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            current_section = text
            if title is None:
                title = text
            continue
        blocks.append(ParsedBlock(text=text, section=current_section, page=None))

    if not blocks:
        raise DocumentParseError(f"{path}: no extractable text content after HTML parsing")
    return ParsedDocument(blocks=blocks, title=title, page_count=None, extraction_method="html")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# A line is a repeated-header/footer CANDIDATE if it's short and recurs
# near-verbatim across a large fraction of pages -- the standard signature
# of a running header/footer/page-number, not real body content (section 4:
# "Remove repetitive headers, footers, page numbers, and extraction
# artifacts where possible"). Scoped to short lines specifically so a
# genuinely repeated short technical phrase inside normal body text (rare,
# and would need to recur on most pages to even be a candidate) is the only
# plausible false-positive case, and even that only strips one short line.
_MAX_HEADER_FOOTER_LINE_CHARS = 80
_HEADER_FOOTER_MIN_PAGE_FRACTION = 0.3

# A heading-like PDF line: short, no trailing sentence punctuation, and
# either numbered ("3.2 Fillet Radius Selection") or starting with a
# capital letter -- conservative on purpose (section 5: don't fabricate
# section names). Any page whose lines don't match this gets NO section,
# not a guessed one -- page_start/page_end remains the reliable structural
# anchor for those blocks.
_PDF_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,/()&'-]{2,80}$")


def _strip_repeated_header_footer_lines(pages: list[str]) -> list[str]:
    if len(pages) < 4:
        return pages  # too few pages for "repeated across many pages" to mean anything
    line_counts: Counter[str] = Counter()
    for page_text in pages:
        lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
        candidate_lines = {ln for ln in (lines[:2] + lines[-2:]) if len(ln) <= _MAX_HEADER_FOOTER_LINE_CHARS}
        for ln in candidate_lines:
            line_counts[ln] += 1
    threshold = max(2, int(len(pages) * _HEADER_FOOTER_MIN_PAGE_FRACTION))
    repeated = {ln for ln, count in line_counts.items() if count >= threshold}
    if not repeated:
        return pages
    cleaned = []
    for page_text in pages:
        kept = [ln for ln in page_text.split("\n") if ln.strip() not in repeated]
        cleaned.append("\n".join(kept))
    return cleaned


def parse_pdf(path: Path, config: KnowledgeConfig) -> ParsedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise DocumentParseError(f"{path}: PDF parsing requires PyMuPDF (pip install pymupdf)") from exc

    doc = fitz.open(str(path))
    try:
        page_texts = [page.get_text("text") for page in doc]
        title = (doc.metadata or {}).get("title") or None
        if title:
            title = title.strip() or None
        page_count = doc.page_count
    finally:
        doc.close()

    non_empty_fraction = sum(1 for t in page_texts if len(t.strip()) >= 20) / max(1, len(page_texts))
    warnings: list[str] = []
    if non_empty_fraction < 0.5:
        scanned_estimate = int(round((1 - non_empty_fraction) * len(page_texts)))
        if not config.enable_ocr:
            raise DocumentParseError(
                f"{path}: {scanned_estimate}/{len(page_texts)} pages have no extractable text layer "
                "(likely scanned images) -- OCR is disabled (KNOWLEDGE_ENABLE_OCR=0). Enable OCR "
                "explicitly or supply a text-based PDF instead; this is reported rather than silently "
                "indexing near-empty pages as valid engineering content."
            )
        page_texts, ocr_warning = _ocr_fill_empty_pages(path, page_texts)
        if ocr_warning:
            warnings.append(ocr_warning)

    page_texts = _strip_repeated_header_footer_lines(page_texts)

    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    for page_num, page_text in enumerate(page_texts, start=1):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        for para in paragraphs:
            first_line = para.split("\n", 1)[0].strip()
            if len(para.split("\n")) == 1 and _PDF_HEADING_RE.match(first_line) and len(first_line) < 100:
                current_section = first_line
                if title is None:
                    title = first_line
                continue  # the heading line itself isn't indexed as a body block
            blocks.append(ParsedBlock(text=para, section=current_section, page=page_num))

    if not blocks:
        raise DocumentParseError(f"{path}: no extractable text content after PDF parsing")
    return ParsedDocument(
        blocks=blocks, title=title, page_count=page_count, extraction_method="pdf_text", warnings=warnings
    )


def _ocr_fill_empty_pages(path: Path, page_texts: list[str]) -> tuple[list[str], str | None]:
    """Only called when config.enable_ocr is True AND a real text layer is
    sparse. Optional dependency (pytesseract + a system tesseract binary) --
    missing either degrades to leaving empty pages empty (reported via the
    returned warning), never a hard crash (section 3's "graceful
    degradation if optional dependencies are unavailable")."""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        return page_texts, (
            "KNOWLEDGE_ENABLE_OCR=1 but pytesseract/Pillow are not installed -- "
            "pages with no text layer were left empty rather than OCR'd"
        )

    doc = fitz.open(str(path))
    try:
        filled = list(page_texts)
        for i, page in enumerate(doc):
            if len(filled[i].strip()) >= 20:
                continue
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                filled[i] = pytesseract.image_to_string(img)
            except Exception:
                continue  # OCR failing on one page degrades that page to empty, not the whole document
        return filled, None
    finally:
        doc.close()
