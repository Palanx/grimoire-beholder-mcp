"""SourceParser registry: extension-based dispatch -- the seam for adding a new source type."""

from __future__ import annotations

from pathlib import Path

import pytest

from grimoire_beholder.sources import get_parser
from grimoire_beholder.sources.epub import EpubParser
from grimoire_beholder.sources.pdf import PdfParser
from grimoire_beholder.sources.plaintext import MarkdownParser, PlaintextParser


@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [
        ("book.pdf", PdfParser),
        ("book.PDF", PdfParser),
        ("book.epub", EpubParser),
        ("book.md", MarkdownParser),
        ("book.markdown", MarkdownParser),
        ("book.txt", PlaintextParser),
    ],
)
def test_get_parser_dispatches_by_extension(filename: str, expected_type: type) -> None:
    assert isinstance(get_parser(Path(filename)), expected_type)


def test_get_parser_raises_on_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="No source parser registered"):
        get_parser(Path("book.docx"))
