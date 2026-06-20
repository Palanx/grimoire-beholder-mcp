"""Markdown and plain-text -> Book/Chapter/Section hierarchy.

Neither format carries a table of contents, so structure is inferred from
the text itself:

- Markdown: a top-level (`# `) heading -- on its own paragraph block, i.e.
  followed by a blank line -- starts a new chapter; text before the first
  one becomes an "Untitled" leading chapter. Within each chapter,
  paragraphs are packed into sections by the same token-budget rule every
  other source type uses (see `common.auto_split_paragraphs`).
- Plain text: no heading syntax is recognized, so the whole file is one
  chapter, split into sections the same way.

Neither format has page numbers, so `page_start` here is a 1-based
paragraph ordinal within the file. Title/author metadata isn't attempted
for these formats (there's nowhere reliable to read it from); the book's
display name falls back to the filename, same as it always has.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from ..ollama_client import OllamaClient
from .common import Chapter, ExtractedBook, auto_split_paragraphs, content_hash_of

_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")


class MarkdownParser:
    source_type = "markdown"
    extensions = (".md", ".markdown")

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def extract(
        self,
        path: Path,
        section_split_tokens: int = 3000,
        conn: sqlite3.Connection | None = None,
        llm_client: OllamaClient | None = None,
        llm_model: str | None = None,
    ) -> ExtractedBook:
        """Unused `conn`/`llm_client`/`llm_model`: only PdfParser's LLM-TOC fallback needs them."""
        return _parse_text_document(path, section_split_tokens, self.source_type, _MD_HEADING_RE)


class PlaintextParser:
    source_type = "text"
    extensions = (".txt",)

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def extract(
        self,
        path: Path,
        section_split_tokens: int = 3000,
        conn: sqlite3.Connection | None = None,
        llm_client: OllamaClient | None = None,
        llm_model: str | None = None,
    ) -> ExtractedBook:
        """Unused `conn`/`llm_client`/`llm_model`: only PdfParser's LLM-TOC fallback needs them."""
        return _parse_text_document(path, section_split_tokens, self.source_type, None)


def _parse_text_document(
    path: Path,
    section_split_tokens: int,
    source_type: str,
    heading_re: re.Pattern | None,
) -> ExtractedBook:
    content_hash = content_hash_of(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]

    chapters = []
    for title, body_start, body_end in _chapter_bounds(paragraphs, heading_re):
        located = [(i + 1, paragraphs[i]) for i in range(body_start, body_end)]
        sections = auto_split_paragraphs(
            located, section_split_tokens, fallback_location=body_start + 1
        )
        chapters.append(Chapter(len(chapters), title, body_start + 1, sections))

    return ExtractedBook(
        content_hash=content_hash,
        page_count=len(paragraphs),
        chapters=chapters,
        title=None,
        author=None,
        source_type=source_type,
    )


def _chapter_bounds(
    paragraphs: list[str], heading_re: re.Pattern | None
) -> list[tuple[str, int, int]]:
    if not paragraphs:
        return []
    if heading_re is None:
        return [("Full Document", 0, len(paragraphs))]

    headings: list[tuple[str, int]] = []
    for i, para in enumerate(paragraphs):
        match = heading_re.match(para.splitlines()[0])
        if match:
            headings.append((match.group(1), i))

    if not headings:
        return [("Full Document", 0, len(paragraphs))]

    bounds: list[tuple[str, int, int]] = []
    if headings[0][1] > 0:
        bounds.append(("Untitled", 0, headings[0][1]))
    for i, (title, heading_pos) in enumerate(headings):
        body_start = heading_pos + 1
        body_end = headings[i + 1][1] if i + 1 < len(headings) else len(paragraphs)
        bounds.append((title, body_start, body_end))
    return bounds
