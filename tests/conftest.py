"""Shared fixtures: a temp SQLite connection and synthetic test PDFs.

The PDFs are built with PyMuPDF directly in-fixture (no fixture files
committed to the repo) so the test suite is fully self-contained.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from book_rag import db as db_mod

_SENT = "The quick brown fox jumps over the lazy dog and contemplates existence. "


@pytest.fixture
def conn(tmp_path: Path):
    connection = db_mod.connect(str(tmp_path / "test.db"))
    yield connection
    connection.close()


def _make_page(doc: pymupdf.Document, lines: list[str]) -> None:
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 750), "\n".join(lines), fontsize=11)


@pytest.fixture
def toc_pdf_path(tmp_path: Path) -> Path:
    """A 6-page PDF with a TOC: 2 chapters, each with 2 level-2 sub-entries."""
    doc = pymupdf.open()
    _make_page(doc, ["Chapter 1: Origins", "", "Intro paragraph for chapter 1." + _SENT * 3])
    _make_page(doc, ["1.1 The Beginning", "", _SENT * 4])
    _make_page(doc, ["1.2 The Middle", "", _SENT * 4])
    _make_page(doc, ["Chapter 2: Consequences", "", "Intro paragraph for chapter 2." + _SENT * 3])
    _make_page(doc, ["2.1 Immediate Effects", "", _SENT * 4])
    _make_page(doc, ["2.2 Long Term Effects", "", _SENT * 4])
    doc.set_toc(
        [
            [1, "Chapter 1: Origins", 1],
            [2, "1.1 The Beginning", 2],
            [2, "1.2 The Middle", 3],
            [1, "Chapter 2: Consequences", 4],
            [2, "2.1 Immediate Effects", 5],
            [2, "2.2 Long Term Effects", 6],
        ]
    )
    path = tmp_path / "toc_book.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def flat_pdf_path(tmp_path: Path) -> Path:
    """A 6-page PDF with no TOC and no headings: one flat, undivided chapter."""
    doc = pymupdf.open()
    for i in range(6):
        _make_page(doc, [f"Paragraph block {i}.", _SENT * 8])
    path = tmp_path / "flat_book.pdf"
    doc.save(str(path))
    doc.close()
    return path
