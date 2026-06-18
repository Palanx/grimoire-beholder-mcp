"""EPUB -> Book/Chapter/Section hierarchy, against a tiny synthetic EPUB built with ebooklib."""

from __future__ import annotations

from pathlib import Path

import pytest
from ebooklib import epub

from grimoire_beholder.sources.epub import EpubParser


@pytest.fixture
def sample_epub_path(tmp_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("test-book-id")
    book.set_title("A Test Book")
    book.set_language("en")
    book.add_author("Jane Author")

    c1 = epub.EpubHtml(uid="chap1", title="Chapter One", file_name="chap1.xhtml", lang="en")
    c1.content = (
        "<html><body><h1>Chapter One</h1>"
        "<p>First paragraph of chapter one.</p>"
        "<p>Second paragraph of chapter one.</p></body></html>"
    )
    book.add_item(c1)

    c2 = epub.EpubHtml(uid="chap2", title="Chapter Two", file_name="chap2.xhtml", lang="en")
    c2.content = "<html><body><h1>Chapter Two</h1><p>Only paragraph of chapter two.</p></body></html>"
    book.add_item(c2)

    book.toc = (
        epub.Link("chap1.xhtml", "Chapter One", "chap1"),
        epub.Link("chap2.xhtml", "Chapter Two", "chap2"),
    )
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]

    path = tmp_path / "sample.epub"
    epub.write_epub(str(path), book)
    return path


def test_epub_extracts_title_and_author(sample_epub_path: Path) -> None:
    extracted = EpubParser().extract(sample_epub_path)

    assert extracted.title == "A Test Book"
    assert extracted.author == "Jane Author"
    assert extracted.source_type == "epub"


def test_epub_one_chapter_per_spine_document_with_toc_titles(sample_epub_path: Path) -> None:
    extracted = EpubParser().extract(sample_epub_path)

    assert len(extracted.chapters) == 2
    assert [c.title for c in extracted.chapters] == ["Chapter One", "Chapter Two"]
    assert "First paragraph of chapter one." in extracted.chapters[0].sections[0].text
    assert "Only paragraph of chapter two." in extracted.chapters[1].sections[0].text


def test_epub_page_start_is_strictly_increasing_across_chapters(sample_epub_path: Path) -> None:
    extracted = EpubParser().extract(sample_epub_path)

    page_starts = [s.page_start for c in extracted.chapters for s in c.sections]
    assert page_starts == sorted(page_starts)
    assert len(set(page_starts)) == len(page_starts)


def test_epub_content_hash_stable_across_calls(sample_epub_path: Path) -> None:
    first = EpubParser().extract(sample_epub_path)
    second = EpubParser().extract(sample_epub_path)

    assert first.content_hash == second.content_hash


def test_epub_can_parse_claims_only_epub_extension(sample_epub_path: Path, tmp_path: Path) -> None:
    parser = EpubParser()

    assert parser.can_parse(sample_epub_path) is True
    assert parser.can_parse(tmp_path / "book.pdf") is False
