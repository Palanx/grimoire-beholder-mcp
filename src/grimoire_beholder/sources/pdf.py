"""PDF -> Book/Chapter/Section hierarchy, using the embedded table of contents.

Chapters come from level-1 TOC entries, falling back to a regex heading
scan only when the PDF has no TOC. Each chapter is then split into
sections, in strict priority order:

1. TOC sub-entries (level >= 2) that fall inside the chapter's page range.
2. If there are none and the chapter is longer than `section_split_tokens`,
   auto-split into consecutive ~`section_split_tokens`-token sections,
   breaking on paragraph boundaries where possible.
3. Otherwise, the whole chapter is a single section.

A chunk is later built from one section's text only, so it can never cross
a section boundary -- the hierarchy (chapter -> section -> chunk) always
exists, even for a flat chapter with no sub-headings at all.

Cleans repeated running headers/footers and stray page numbers before any
of the above, so page-range slicing operates on clean text.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

from .common import Chapter, ExtractedBook, Section, content_hash_of

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?[ivxlcdm\d]+\s*$", re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_HEADING_RE = re.compile(r"^\s*(?:chapter|part)\s+[ivxlcdm\d]+\b", re.IGNORECASE)
_PARA_MARKER = "\x00"
_CHARS_PER_TOKEN = 4


class PdfParser:
    source_type = "pdf"
    extensions = (".pdf",)

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def extract(self, path: Path, section_split_tokens: int = 3000) -> ExtractedBook:
        return extract_book(str(path), section_split_tokens)


def extract_book(pdf_path: str, section_split_tokens: int = 3000) -> ExtractedBook:
    """Extract the full chapter/section hierarchy, content hash, and metadata from a PDF."""
    content_hash = content_hash_of(Path(pdf_path))

    doc = pymupdf.open(pdf_path)
    try:
        page_count = doc.page_count
        pages_text = [page.get_text("text") for page in doc]
        pages_text = _strip_running_headers_footers(pages_text)
        pages_text = [_clean_text(t) for t in pages_text]

        toc = doc.get_toc(simple=True)
        chapter_bounds = _chapter_bounds_from_toc(toc, page_count)
        if chapter_bounds is None:
            chapter_bounds = _chapter_bounds_from_headings(pages_text)

        chapters = []
        for chapter_index, (title, page_start, page_end) in enumerate(chapter_bounds):
            sub_entries = _sub_entries_for_chapter(toc, page_start, page_end)
            sections = _derive_sections(
                sub_entries, pages_text, page_start, page_end, section_split_tokens
            )
            chapters.append(Chapter(chapter_index, title, page_start, sections))

        metadata = doc.metadata or {}
        title = (metadata.get("title") or "").strip() or None
        author = (metadata.get("author") or "").strip() or None
        return ExtractedBook(
            content_hash=content_hash,
            page_count=page_count,
            chapters=chapters,
            title=title,
            author=author,
            source_type="pdf",
        )
    finally:
        doc.close()


# -- chapter boundary detection -----------------------------------------------


def _chapter_bounds_from_toc(toc: list, page_count: int) -> list[tuple[str, int, int]] | None:
    top = [(title.strip(), page) for level, title, page in toc if level == 1]
    if not top:
        return None
    bounds = []
    for i, (title, page) in enumerate(top):
        page_start = max(1, page)
        page_end = top[i + 1][1] - 1 if i + 1 < len(top) else page_count
        page_end = max(page_start, min(page_end, page_count))
        bounds.append((title, page_start, page_end))
    return bounds


def _chapter_bounds_from_headings(pages_text: list[str]) -> list[tuple[str, int, int]]:
    """Heuristic fallback: treat lines matching 'Chapter N' / 'Part N' as boundaries."""
    n_pages = len(pages_text)
    boundaries: list[tuple[str, int]] = []
    for page_idx, text in enumerate(pages_text, start=1):
        for line in text.split("\n")[:3]:
            if _HEADING_RE.match(line.strip()):
                boundaries.append((line.strip(), page_idx))
                break

    if not boundaries:
        return [("Full Document", 1, n_pages)]

    bounds = []
    for i, (title, page_start) in enumerate(boundaries):
        page_end = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else n_pages
        page_end = max(page_start, page_end)
        bounds.append((title, page_start, page_end))
    return bounds


# -- section derivation --------------------------------------------------------


def _sub_entries_for_chapter(toc: list, page_start: int, page_end: int) -> list[tuple[str, int]]:
    seen_pages: set[int] = set()
    entries = []
    for level, title, page in toc:
        if level >= 2 and page_start <= page <= page_end and page not in seen_pages:
            entries.append((title.strip(), page))
            seen_pages.add(page)
    return sorted(entries, key=lambda e: e[1])


def _derive_sections(
    sub_entries: list[tuple[str, int]],
    pages_text: list[str],
    page_start: int,
    page_end: int,
    section_split_tokens: int,
) -> list[Section]:
    if sub_entries:
        return _sections_from_subentries(sub_entries, pages_text, page_start, page_end)

    chapter_text = _join_pages(pages_text, page_start, page_end)
    if len(chapter_text) > section_split_tokens * _CHARS_PER_TOKEN:
        return _auto_split_chapter(pages_text, page_start, page_end, section_split_tokens)
    return [Section(0, None, chapter_text, page_start)]


def _sections_from_subentries(
    sub_entries: list[tuple[str, int]], pages_text: list[str], page_start: int, page_end: int
) -> list[Section]:
    boundaries: list[tuple[str | None, int]] = []
    if sub_entries[0][1] > page_start:
        boundaries.append((None, page_start))
    boundaries.extend(sub_entries)

    sections = []
    for i, (title, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else page_end
        end = max(end, start)
        text = _join_pages(pages_text, start, end)
        sections.append(Section(i, title, text, start))
    return sections


def _auto_split_chapter(
    pages_text: list[str], page_start: int, page_end: int, section_split_tokens: int
) -> list[Section]:
    """Greedily pack paragraphs into ~section_split_tokens-sized sections.

    Breaks always land on a paragraph boundary, except when a single
    paragraph alone exceeds the target size, in which case it simply
    forms its own oversized section.
    """
    max_chars = section_split_tokens * _CHARS_PER_TOKEN
    paragraphs: list[tuple[int, str]] = []
    for offset, page_text in enumerate(pages_text[page_start - 1 : page_end]):
        page_num = page_start + offset
        for para in page_text.split("\n\n"):
            para = para.strip()
            if para:
                paragraphs.append((page_num, para))

    if not paragraphs:
        return [Section(0, None, "", page_start)]

    sections: list[Section] = []
    current_parts: list[str] = []
    current_len = 0
    current_page = paragraphs[0][0]
    for page_num, para in paragraphs:
        added_len = len(para) + 2
        if current_parts and current_len + added_len > max_chars:
            sections.append(
                Section(len(sections), None, "\n\n".join(current_parts), current_page)
            )
            current_parts = []
            current_len = 0
            current_page = page_num
        current_parts.append(para)
        current_len += added_len
    if current_parts:
        sections.append(Section(len(sections), None, "\n\n".join(current_parts), current_page))
    return sections


def _join_pages(pages_text: list[str], page_start: int, page_end: int) -> str:
    return "\n\n".join(p for p in pages_text[page_start - 1 : page_end] if p)


# -- text cleaning ----------------------------------------------------------


def _strip_running_headers_footers(pages_text: list[str]) -> list[str]:
    """Drop the first/last line of each page if it repeats on most pages."""
    if len(pages_text) < 3:
        return pages_text
    split_pages = [text.split("\n") for text in pages_text]
    first_lines = Counter(lines[0].strip() for lines in split_pages if lines)
    last_lines = Counter(lines[-1].strip() for lines in split_pages if lines)
    threshold = max(3, len(pages_text) // 2)
    repeated_first = {line for line, n in first_lines.items() if line and n >= threshold}
    repeated_last = {line for line, n in last_lines.items() if line and n >= threshold}

    cleaned = []
    for lines in split_pages:
        if lines and lines[0].strip() in repeated_first:
            lines = lines[1:]
        if lines and lines[-1].strip() in repeated_last:
            lines = lines[:-1]
        lines = [line for line in lines if not _PAGE_NUMBER_RE.match(line.strip())]
        cleaned.append("\n".join(lines))
    return cleaned


def _clean_text(raw: str) -> str:
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", raw)
    text = re.sub(r"\n{2,}", _PARA_MARKER, text)
    text = text.replace("\n", " ")
    text = text.replace(_PARA_MARKER, "\n\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
