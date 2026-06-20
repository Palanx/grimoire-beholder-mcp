"""EPUB -> Book/Chapter/Section hierarchy, via ebooklib.

Each spine document (in reading order, navigation documents excluded)
becomes one chapter; chapter titles come from the EPUB's nav/TOC where it
maps to that document, falling back to the document's first heading tag
and then to a generic "Chapter N". EPUBs are reflowable text with no fixed
pagination, so `page_start` here is a synthetic, strictly increasing
location ordinal (spine position * a stride, plus paragraph offset within
that chapter) rather than a real page number -- it still sorts and citates
correctly, it's just not a number a reader would recognize.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import ebooklib
from ebooklib import epub
from lxml import html as lxml_html

from ..ollama_client import OllamaClient
from .common import Chapter, ExtractedBook, auto_split_paragraphs, content_hash_of

_LOCATION_STRIDE = 100_000
_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}
_HEADING_TAGS = ("h1", "h2", "h3", "h4")


class EpubParser:
    source_type = "epub"
    extensions = (".epub",)

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
        content_hash = content_hash_of(path)
        book = epub.read_epub(str(path))

        title = (book.title or "").strip() or None
        creators = book.get_metadata("DC", "creator")
        author = creators[0][0].strip() if creators and creators[0][0] else None

        toc_titles = _flatten_toc_titles(book.toc)

        chapters: list[Chapter] = []
        for idref, _linear in book.spine:
            item = book.get_item_with_id(idref)
            if item is None or isinstance(item, epub.EpubNav):
                continue
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue

            content = item.get_content()
            paragraphs = _extract_paragraphs(content)
            if not paragraphs:
                continue

            spine_ordinal = len(chapters) + 1
            chapter_title = (
                toc_titles.get(item.get_name())
                or _first_heading(content)
                or f"Chapter {spine_ordinal}"
            )
            located = [
                (spine_ordinal * _LOCATION_STRIDE + i, para) for i, para in enumerate(paragraphs)
            ]
            sections = auto_split_paragraphs(located, section_split_tokens)
            chapters.append(Chapter(len(chapters), chapter_title, located[0][0], sections))

        return ExtractedBook(
            content_hash=content_hash,
            page_count=len(chapters),
            chapters=chapters,
            title=title,
            author=author,
            source_type=self.source_type,
        )


def _flatten_toc_titles(toc) -> dict[str, str]:
    """Map href (fragment stripped) -> nav title, walking the possibly-nested TOC."""
    titles: dict[str, str] = {}

    def visit(node) -> None:
        if isinstance(node, epub.Link):
            href = node.href.split("#")[0]
            titles.setdefault(href, node.title.strip())
        elif isinstance(node, (tuple, list)):
            for child in node:
                visit(child)

    visit(toc)
    return titles


def _extract_paragraphs(html_bytes: bytes) -> list[str]:
    tree = lxml_html.fromstring(html_bytes)
    paragraphs = []
    for el in tree.iter():
        if el.tag in _BLOCK_TAGS:
            text = " ".join(el.text_content().split())
            if text:
                paragraphs.append(text)
    return paragraphs


def _first_heading(html_bytes: bytes) -> str | None:
    tree = lxml_html.fromstring(html_bytes)
    for tag in _HEADING_TAGS:
        el = tree.find(f".//{tag}")
        if el is not None:
            text = " ".join(el.text_content().split())
            if text:
                return text
    return None
