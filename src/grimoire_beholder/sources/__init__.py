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

import sqlite3
from pathlib import Path
from typing import Protocol

from ..ollama_client import OllamaClient
from .common import Chapter, ExtractedBook, Section
from .epub import EpubParser
from .pdf import PdfParser
from .plaintext import MarkdownParser, PlaintextParser


class SourceParser(Protocol):
    source_type: str
    extensions: tuple[str, ...]

    def can_parse(self, path: Path) -> bool: ...

    def extract(
        self,
        path: Path,
        section_split_tokens: int = 3000,
        conn: sqlite3.Connection | None = None,
        llm_client: OllamaClient | None = None,
        llm_model: str | None = None,
    ) -> ExtractedBook:
        """Parse `path` into a Book/Chapter/Section hierarchy.

        `conn`/`llm_client`/`llm_model` are only meaningful to `PdfParser`,
        which uses them to extract and cache a TOC via the LLM when a PDF
        has no embedded outline. Every other parser accepts and ignores
        them so `ingest.py` can call `extract()` uniformly without knowing
        which concrete parser it got.
        """
        ...


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
