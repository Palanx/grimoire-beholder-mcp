"""Section derivation: the three-branch priority (TOC sub-entries > auto-split > whole chapter)."""

from __future__ import annotations

from pathlib import Path

from book_rag.sources import pdf


def test_toc_subentries_become_sections(toc_pdf_path: Path) -> None:
    book = pdf.extract_book(str(toc_pdf_path))

    assert len(book.chapters) == 2
    ch1, ch2 = book.chapters
    assert ch1.title == "Chapter 1: Origins"
    assert [s.title for s in ch1.sections] == [None, "1.1 The Beginning", "1.2 The Middle"]
    assert [s.section_index for s in ch1.sections] == [0, 1, 2]
    assert ch2.title == "Chapter 2: Consequences"
    assert [s.title for s in ch2.sections] == [None, "2.1 Immediate Effects", "2.2 Long Term Effects"]


def test_long_flat_chapter_auto_splits_on_paragraph_boundaries(flat_pdf_path: Path) -> None:
    book = pdf.extract_book(str(flat_pdf_path), section_split_tokens=150)

    assert len(book.chapters) == 1
    sections = book.chapters[0].sections
    assert len(sections) > 1
    assert all(s.title is None for s in sections)
    max_chars = 150 * 4
    # Auto-split only overshoots the cap when a single paragraph alone exceeds it.
    assert all(len(s.text) <= max_chars * 2 for s in sections)
    page_starts = [s.page_start for s in sections]
    assert page_starts == sorted(page_starts)


def test_short_chapter_becomes_one_whole_section(flat_pdf_path: Path) -> None:
    book = pdf.extract_book(str(flat_pdf_path), section_split_tokens=3000)

    assert len(book.chapters) == 1
    sections = book.chapters[0].sections
    assert len(sections) == 1
    assert sections[0].title is None
    for i in range(6):
        assert f"Paragraph block {i}." in sections[0].text


def test_content_hash_is_stable_across_calls(toc_pdf_path: Path) -> None:
    first = pdf.extract_book(str(toc_pdf_path))
    second = pdf.extract_book(str(toc_pdf_path))

    assert first.content_hash == second.content_hash
