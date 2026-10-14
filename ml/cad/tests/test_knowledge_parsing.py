"""Tests for knowledge/document_parser.py -- format-specific extraction.
PDF tests build a tiny synthetic PDF in-memory via PyMuPDF itself (fitz can
both write and read PDFs), so there's no binary fixture file to maintain."""

from pathlib import Path

import pytest

from knowledge.config import KnowledgeConfig
from knowledge.document_parser import (
    DocumentParseError,
    _strip_repeated_header_footer_lines,
    parse_html,
    parse_markdown,
    parse_pdf,
    parse_plain_text,
    parse_rst,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "knowledge_docs"


def test_parse_plain_text_splits_on_blank_lines(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("First paragraph.\n\nSecond paragraph, two lines\nstill the same paragraph.\n")
    doc = parse_plain_text(path)
    assert doc.extraction_method == "plain_text"
    assert len(doc.blocks) == 2
    assert doc.blocks[0].text == "First paragraph."
    assert "still the same paragraph" in doc.blocks[1].text


def test_parse_plain_text_raises_on_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    with pytest.raises(DocumentParseError):
        parse_plain_text(path)


def test_parse_markdown_tracks_headings_as_sections(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(
        "# Doc Title\n\nIntro paragraph.\n\n## Section A\n\nParagraph under A.\n\n## Section B\n\nParagraph under B.\n"
    )
    doc = parse_markdown(path)
    assert doc.title == "Doc Title"
    sections = [b.section for b in doc.blocks]
    assert sections == ["Doc Title", "Section A", "Section B"]
    assert doc.blocks[1].text == "Paragraph under A."


def test_parse_markdown_does_not_split_inside_a_code_fence(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# T\n\n```\nnot a # heading\nstill code\n```\n")
    doc = parse_markdown(path)
    assert len(doc.blocks) == 1
    assert "not a # heading" in doc.blocks[0].text
    assert "still code" in doc.blocks[0].text


def test_parse_markdown_raises_on_empty_file(tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("   \n\n  ")
    with pytest.raises(DocumentParseError):
        parse_markdown(path)


def test_parse_markdown_skips_yaml_front_matter(tmp_path):
    """Regression test for a real, confirmed failure against an actually
    acquired document (FreeCAD-documentation's wiki mirror): a leading
    '---\\n<yaml>\\n---\\n' front-matter block's CLOSING '---' was being
    misread as a setext H2 underline, corrupting the title/section with
    YAML field text instead of the file's real '#' heading."""
    path = tmp_path / "doc.md"
    path.write_text(
        "---\n GuiCommand:\n   Name: PartDesign Fillet\n   SeeAlso: PartDesign_Chamfer\n---\n\n"
        "# PartDesign Fillet\n\nReal body text.\n"
    )
    doc = parse_markdown(path)
    assert doc.title == "PartDesign Fillet"
    assert doc.blocks[0].text == "Real body text."
    assert doc.blocks[0].section == "PartDesign Fillet"


def test_parse_markdown_does_not_misread_a_stack_trace_frame_as_a_heading(tmp_path):
    """Regression test for a real, confirmed failure: a gdb/lldb-style
    crash-log line ("#1  0xADDRESS in function(...)") matches the ATX
    heading regex (a single '#', whitespace, content) and was being read
    as a new level-1 heading, corrupting every subsequent chunk's section
    until the next real heading."""
    path = tmp_path / "doc.md"
    path.write_text(
        "# Known issues\n\nFreeCAD may output a log like this:\n\n"
        "#1  0x7fff63d660ba in BRep_Tool::Curve(TopoDS_Edge const&) from libTKBRep.so\n"
        "#2  0x7fff63d69546 in BRep_Tool::Curve(TopoDS_Edge const&) from libTKBRep.so\n\n"
        "## Topological naming\n\nMore real content.\n"
    )
    doc = parse_markdown(path)
    sections = [b.section for b in doc.blocks]
    assert "Known issues" in sections
    assert "Topological naming" in sections
    # the stack-trace content must never have become its own section name
    assert not any(s and s.startswith("1  0x") for s in sections)


def test_parse_markdown_real_numeral_led_heading_still_works(tmp_path):
    """The stack-trace guard above is scoped to "digits, whitespace, 0x
    hex" specifically so a legitimate numeral-led heading (e.g. a real
    FreeCAD wiki page title like "3D View") is never affected."""
    path = tmp_path / "doc.md"
    path.write_text("# 3D View\n\nBody text about the 3D view.\n")
    doc = parse_markdown(path)
    assert doc.title == "3D View"
    assert doc.blocks[0].section == "3D View"


def test_parse_html_extracts_headings_and_title(tmp_path):
    path = tmp_path / "doc.html"
    path.write_text(
        "<html><head><title>My Doc</title></head><body>"
        "<h1>Intro</h1><p>Intro text.</p>"
        "<h2>Details</h2><p>Detail text.</p>"
        "<script>ignored()</script>"
        "</body></html>"
    )
    doc = parse_html(path)
    assert doc.title == "My Doc"
    assert [b.text for b in doc.blocks] == ["Intro text.", "Detail text."]
    assert doc.blocks[0].section == "Intro"
    assert doc.blocks[1].section == "Details"
    assert all("ignored" not in b.text for b in doc.blocks)


def test_parse_html_raises_when_no_content(tmp_path):
    path = tmp_path / "empty.html"
    path.write_text("<html><body></body></html>")
    with pytest.raises(DocumentParseError):
        parse_html(path)


def test_strip_repeated_header_footer_lines_removes_running_header():
    pages = [
        "COMPANY CONFIDENTIAL\nPage body text one.\nmore text",
        "COMPANY CONFIDENTIAL\nPage body text two.\nmore text",
        "COMPANY CONFIDENTIAL\nPage body text three.\nmore text",
        "COMPANY CONFIDENTIAL\nPage body text four.\nmore text",
    ]
    cleaned = _strip_repeated_header_footer_lines(pages)
    assert all("COMPANY CONFIDENTIAL" not in p for p in cleaned)
    assert "Page body text one." in cleaned[0]


def test_strip_repeated_header_footer_lines_is_a_noop_for_few_pages():
    pages = ["A header\nunique one", "A header\nunique two"]
    assert _strip_repeated_header_footer_lines(pages) == pages


def _build_synthetic_pdf(path, page_texts: list[str]) -> None:
    import fitz

    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        y = 72
        for line in text.split("\n"):
            page.insert_text((72, y), line, fontsize=11)
            y += 16
    doc.save(str(path))
    doc.close()


def test_parse_pdf_extracts_text_and_pages(tmp_path):
    path = tmp_path / "doc.pdf"
    _build_synthetic_pdf(path, [
        "Fillets And Chamfers\nA fillet rounds a sharp edge.",
        "More detail on the second page about chamfers.",
    ])
    config = KnowledgeConfig()
    doc = parse_pdf(path, config)
    assert doc.extraction_method == "pdf_text"
    assert doc.page_count == 2
    assert any(b.page == 1 for b in doc.blocks)
    assert any(b.page == 2 for b in doc.blocks)
    assert any("fillet" in b.text.lower() for b in doc.blocks)


def test_parse_pdf_raises_when_no_text_layer_and_ocr_disabled(tmp_path):
    import fitz

    path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()  # pages with zero text content -- simulates a scanned PDF
    doc.save(str(path))
    doc.close()

    config = KnowledgeConfig()  # enable_ocr defaults to False
    with pytest.raises(DocumentParseError, match="OCR"):
        parse_pdf(path, config)


def test_fixture_documents_parse_successfully():
    """Smoke test against the real test fixtures used by the other
    knowledge/ test modules -- confirms they're well-formed."""
    md = parse_markdown(FIXTURES_DIR / "solid_modeling_basics.md")
    assert md.title == "Solid Modeling Basics"
    assert len(md.blocks) >= 5

    txt = parse_plain_text(FIXTURES_DIR / "bracket_mounting.txt")
    assert len(txt.blocks) >= 2


def test_parse_rst_tracks_underlined_headings_as_sections(tmp_path):
    path = tmp_path / "doc.rst"
    path.write_text(
        "*************\nIntroduction\n*************\n\nIntro text.\n\n"
        "Workplanes\n==========\n\nWorkplane text.\n"
    )
    doc = parse_rst(path)
    assert doc.title == "Introduction"
    sections = [b.section for b in doc.blocks]
    assert "Workplanes" in sections
    assert any("Workplane text." in b.text for b in doc.blocks)


def test_parse_rst_strips_labels_and_image_directives(tmp_path):
    path = tmp_path / "doc.rst"
    path.write_text(
        ".. _some_label:\n\nTitle\n=====\n\nSome body text.\n\n"
        ".. image:: _static/thing.png\n\nMore body text after the image.\n"
    )
    doc = parse_rst(path)
    joined = "\n".join(b.text for b in doc.blocks)
    assert "_static/thing.png" not in joined
    assert "some_label" not in joined
    assert "Some body text." in joined
    assert "More body text after the image." in joined


def test_parse_rst_keeps_admonition_body_with_a_plain_text_label(tmp_path):
    """Section 5's "do not split a warning from the operation it
    describes" -- an admonition's body must survive, tagged with a plain
    label, not be dropped like a navigational directive."""
    path = tmp_path / "doc.rst"
    path.write_text(
        "Title\n=====\n\n.. warning::\n   Do not exceed the rated load.\n\nNormal text after.\n"
    )
    doc = parse_rst(path)
    joined = "\n".join(b.text for b in doc.blocks)
    assert "Warning:" in joined
    assert "Do not exceed the rated load." in joined


def test_parse_rst_raises_on_empty_file(tmp_path):
    path = tmp_path / "empty.rst"
    path.write_text("")
    with pytest.raises(DocumentParseError):
        parse_rst(path)


def test_parse_rst_real_acquired_cadquery_file_has_a_clean_title_and_sections():
    """Smoke test against a real file synced via knowledge/sources.py in
    this session (not a hand-authored fixture) -- skipped if that scratch
    sync directory isn't present (e.g. a fresh checkout/CI run that never
    performed the real-source smoke test)."""
    real_file = Path("/tmp/sources_smoke_test/sources/raw/cadquery_docs/doc/intro.rst")
    if not real_file.exists():
        pytest.skip("real acquired-source smoke test directory not present")
    doc = parse_rst(real_file)
    assert doc.title == "Introduction"
    assert any(b.section == "What is CadQuery" for b in doc.blocks)
