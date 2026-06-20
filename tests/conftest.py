"""Shared fixtures: a temp SQLite connection and synthetic test PDFs.

The PDFs are built with PyMuPDF directly in-fixture (no fixture files
committed to the repo) so the test suite is fully self-contained.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from grimoire_beholder import db as db_mod

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


@pytest.fixture
def llm_toc_pdf_path(tmp_path: Path) -> Path:
    """A 9-page PDF with no embedded outline, reproducing the Gregoire TOC-extraction bug.

    - Pages 2-3: a printed TOC spanning *multiple* pages, with declared page
      numbers (1, 6, 10) that do not match the physical PDF pages the
      chapters actually start on (5, 7, 8) -- a non-constant offset, since
      front matter/preface pages push real content later than the TOC's own
      page numbers suggest.
    - Page 4: a Preface page that looks like body text, not a TOC line, so
      TOC-region detection has a real stopping point to find.
    - Page 9: opens with the literal line "Chapter 4: Counting Mistakes" --
      a sub-heading inside Appendix A's body that the regex-only heading
      fallback misreads as its own chapter (the exact failure mode that
      motivated the LLM-TOC fallback). The LLM-TOC path must not promote it.
    """
    doc = pymupdf.open()
    _make_page(doc, ["Professional Test Book", "by Test Author"])
    _make_page(doc, ["Table of Contents", "Chapter 1: Origins .......... 1"])
    _make_page(
        doc,
        [
            "Chapter 2: Consequences .......... 6",
            "Appendix A: Common Mistakes .......... 10",
        ],
    )
    _make_page(
        doc,
        [
            "Preface",
            "This book assumes no prior knowledge of C++ and starts from the basics "
            "of programming.",
        ],
    )
    _make_page(
        doc,
        ["Chapter 1: Origins", "", "This chapter covers the origins of the language." + _SENT * 3],
    )
    _make_page(
        doc,
        ["More discussion of structured and object oriented programming." + _SENT * 4],
    )
    _make_page(
        doc,
        [
            "Chapter 2: Consequences",
            "",
            "This chapter discusses the consequences of early design decisions." + _SENT * 3,
        ],
    )
    _make_page(
        doc,
        [
            "Appendix A: Common Mistakes",
            "",
            "This appendix lists common mistakes beginners make." + _SENT * 3,
        ],
    )
    _make_page(
        doc,
        [
            "Chapter 4: Counting Mistakes",
            "",
            "Many beginners write off-by-one errors when counting loop iterations, "
            "confusing zero based and one based indices." + _SENT * 3,
        ],
    )
    path = tmp_path / "llm_toc_book.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def sparse_density_toc_pdf_path(tmp_path: Path) -> Path:
    """A PDF whose multi-page TOC prints a page number only on top-level entries.

    Reproduces the Gregoire "Professional C++" TOC-region-detection bug:
    each TOC page after the first is almost entirely indented sub-entries
    with no trailing page number, so line-density alone collapses toward 0
    and would truncate the captured region to a single page. The repeating
    "Contents" header is the only signal that survives across the whole
    3-page TOC (pages 2-4); page 5 (Preface) has neither signal and is the
    real stopping point.
    """
    doc = pymupdf.open()
    _make_page(doc, ["Professional Test Book", "by Test Author"])
    _make_page(
        doc,
        ["Contents", "Chapter 1: Origins .......... 5", "1.1 The Beginning", "1.2 The Middle"],
    )
    _make_page(
        doc,
        ["Contents", "1.3 The End", "1.4 Looking Back", "1.5 Looking Forward"],
    )
    _make_page(
        doc,
        ["Contents", "Chapter 2: Consequences .......... 9", "2.1 Immediate Effects", "2.2 Long Term Effects"],
    )
    _make_page(
        doc,
        [
            "Preface",
            "This book assumes no prior knowledge of C++ and starts from the basics "
            "of programming." + _SENT * 2,
        ],
    )
    _make_page(doc, ["Chapter 1: Origins", "", "This chapter covers the origins." + _SENT * 3])
    _make_page(doc, ["More discussion of the topic." + _SENT * 4])
    _make_page(
        doc, ["Chapter 2: Consequences", "", "This chapter discusses consequences." + _SENT * 3]
    )
    path = tmp_path / "sparse_density_toc_book.pdf"
    doc.save(str(path))
    doc.close()
    return path
