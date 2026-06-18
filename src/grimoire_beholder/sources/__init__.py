"""Source parsing layer: turns a file on disk into a Book/Chapter/Section hierarchy.

This is the only layer that knows file formats exist. Everything
downstream (chunking, contextualization, embedding, retrieval) consumes
plain `ExtractedBook`/`Chapter`/`Section` dataclasses and has no idea
whether they came from a PDF, an EPUB, or a markdown file.

To add a new source type: write a class implementing `SourceParser`
(`source_type`, `extensions`, `can_parse`, `extract`) in its own module
here, and append an instance to `_PARSERS` below. Nothing else changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .common import Chapter, ExtractedBook, Section
from .epub import EpubParser
from .pdf import PdfParser
from .plaintext import MarkdownParser, PlaintextParser


class SourceParser(Protocol):
    source_type: str
    extensions: tuple[str, ...]

    def can_parse(self, path: Path) -> bool: ...

    def extract(self, path: Path, section_split_tokens: int = 3000) -> ExtractedBook: ...


_PARSERS: list[SourceParser] = [PdfParser(), EpubParser(), MarkdownParser(), PlaintextParser()]


def get_parser(path: Path) -> SourceParser:
    """Return the registered parser that claims this file, by extension."""
    for parser in _PARSERS:
        if parser.can_parse(path):
            return parser
    supported = sorted({ext for p in _PARSERS for ext in p.extensions})
    raise ValueError(
        f"No source parser registered for '{path.suffix}' files ({path}). "
        f"Supported extensions: {', '.join(supported)}."
    )


__all__ = ["Chapter", "ExtractedBook", "Section", "SourceParser", "get_parser"]
