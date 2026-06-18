"""Dataclasses and helpers shared by every SourceParser.

`page_start` doubles as two different things depending on the source: a
real 1-indexed PDF page number, or -- for sources with no fixed pagination
(EPUB, plaintext, markdown) -- a synthetic, strictly increasing logical
location ordinal. Either way it sorts correctly and citates the same way.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

_CHARS_PER_TOKEN = 4


@dataclass
class Section:
    section_index: int
    title: str | None
    text: str
    page_start: int


@dataclass
class Chapter:
    chapter_index: int
    title: str
    page_start: int
    sections: list[Section] = field(default_factory=list)


@dataclass
class ExtractedBook:
    content_hash: str
    page_count: int
    chapters: list[Chapter]
    title: str | None = None
    author: str | None = None
    source_type: str = "pdf"


def content_hash_of(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def auto_split_paragraphs(
    paragraphs: list[tuple[int, str]], section_split_tokens: int, fallback_location: int = 0
) -> list[Section]:
    """Greedily pack (location, paragraph_text) pairs into ~section_split_tokens sections.

    Breaks always land on a paragraph boundary, except when a single
    paragraph alone exceeds the target size, in which case it simply forms
    its own oversized section. When everything fits in one budget, this
    naturally collapses to a single section -- callers don't need a
    separate "is this chapter short" branch.

    `fallback_location` is used only when `paragraphs` is empty (e.g. a
    markdown chapter with no body between two adjacent headings) -- pass
    the chapter's own starting location so the resulting empty section
    still sorts at or after its chapter instead of always sorting first.
    """
    max_chars = section_split_tokens * _CHARS_PER_TOKEN
    if not paragraphs:
        return [Section(0, None, "", fallback_location)]

    sections: list[Section] = []
    current_parts: list[str] = []
    current_len = 0
    current_loc = paragraphs[0][0]
    for loc, para in paragraphs:
        added_len = len(para) + 2
        if current_parts and current_len + added_len > max_chars:
            sections.append(Section(len(sections), None, "\n\n".join(current_parts), current_loc))
            current_parts = []
            current_len = 0
            current_loc = loc
        current_parts.append(para)
        current_len += added_len
    if current_parts:
        sections.append(Section(len(sections), None, "\n\n".join(current_parts), current_loc))
    return sections
