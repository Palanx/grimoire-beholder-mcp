"""Markdown and plain-text -> Book/Chapter/Section hierarchy."""

from __future__ import annotations

from pathlib import Path

from book_rag.sources.plaintext import MarkdownParser, PlaintextParser


def test_markdown_top_level_headings_become_chapters(tmp_path: Path) -> None:
    path = tmp_path / "book.md"
    path.write_text(
        "# Chapter One\n\n"
        "First paragraph of chapter one.\n\n"
        "Second paragraph of chapter one.\n\n"
        "# Chapter Two\n\n"
        "Only paragraph of chapter two.\n",
        encoding="utf-8",
    )

    extracted = MarkdownParser().extract(path)

    assert extracted.source_type == "markdown"
    assert [c.title for c in extracted.chapters] == ["Chapter One", "Chapter Two"]
    assert "First paragraph of chapter one." in extracted.chapters[0].sections[0].text
    assert "Only paragraph of chapter two." in extracted.chapters[1].sections[0].text


def test_markdown_text_before_first_heading_becomes_untitled_chapter(tmp_path: Path) -> None:
    path = tmp_path / "book.md"
    path.write_text(
        "Some preamble text with no heading.\n\n# Chapter One\n\nChapter one body text.\n",
        encoding="utf-8",
    )

    extracted = MarkdownParser().extract(path)

    assert extracted.chapters[0].title == "Untitled"
    assert "Some preamble text" in extracted.chapters[0].sections[0].text
    assert extracted.chapters[1].title == "Chapter One"


def test_markdown_page_start_is_strictly_increasing(tmp_path: Path) -> None:
    path = tmp_path / "book.md"
    path.write_text(
        "# Chapter One\n\nPara A.\n\nPara B.\n\n# Chapter Two\n\nPara C.\n",
        encoding="utf-8",
    )

    extracted = MarkdownParser().extract(path)

    page_starts = [s.page_start for c in extracted.chapters for s in c.sections]
    assert page_starts == sorted(page_starts)


def test_plaintext_has_no_heading_syntax_and_is_one_chapter(tmp_path: Path) -> None:
    path = tmp_path / "book.txt"
    path.write_text(
        "Paragraph one of a plain text file.\n\n"
        "# This looks like a heading but isn't special here.\n\n"
        "Paragraph three.\n",
        encoding="utf-8",
    )

    extracted = PlaintextParser().extract(path)

    assert extracted.source_type == "text"
    assert len(extracted.chapters) == 1
    assert extracted.chapters[0].title == "Full Document"
    text = extracted.chapters[0].sections[0].text
    assert "Paragraph one" in text
    assert "looks like a heading" in text


def test_plaintext_and_markdown_have_no_title_or_author_metadata(tmp_path: Path) -> None:
    path = tmp_path / "book.txt"
    path.write_text("Just some text.\n", encoding="utf-8")

    extracted = PlaintextParser().extract(path)

    assert extracted.title is None
    assert extracted.author is None


def test_markdown_and_plaintext_can_parse_claim_their_own_extensions(tmp_path: Path) -> None:
    md = MarkdownParser()
    txt = PlaintextParser()

    assert md.can_parse(tmp_path / "book.md") is True
    assert md.can_parse(tmp_path / "book.markdown") is True
    assert md.can_parse(tmp_path / "book.txt") is False
    assert txt.can_parse(tmp_path / "book.txt") is True
    assert txt.can_parse(tmp_path / "book.md") is False
