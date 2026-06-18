"""Read-only MCP server exposing the book-rag library to Claude.

Exactly five tools, all read-only: nothing here can ingest or delete a
book, and no cloud LLM is ever called -- the only model invoked is the
local embedding model, to embed a search question. `ingest` and `delete`
remain CLI-only by design and are never wired into this server.

Runs over stdio, the standard MCP transport for a locally-spawned tool
process (e.g. from claude_desktop_config.json).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import db, ollama_client
from . import search as search_mod
from .config import load_config

mcp = FastMCP("book-rag")


@mcp.tool()
def list_books() -> list[dict]:
    """List every book in the library, with its id, slug, name, author, type, and page count."""
    config = load_config()
    conn = db.connect(config.db_path)
    try:
        return [
            {
                "book_id": b.id,
                "slug": b.slug,
                "name": b.name,
                "author": b.author,
                "source_type": b.source_type,
                "page_count": b.page_count,
            }
            for b in db.list_books(conn)
        ]
    finally:
        conn.close()


@mcp.tool()
def get_book_outline(book_id: int) -> dict:
    """Return one book's chapter/section outline -- the map for get_section.

    Lists every chapter (chapter_index, title, page_start) and, nested
    under each, its sections (section_index, title, page_start,
    approx_tokens). Auto-split sections have no native title; one is
    synthesized from a text snippet so it's still identifiable. Returns
    no section text -- that's get_section's job; this is just the map.

    Typical flow: list_books -> get_book_outline(book_id) -> get_section(
    book_id, chapter_index, section_index) for targeted reading. Use
    search_book instead when you don't already know which section to read.
    """
    config = load_config()
    conn = db.connect(config.db_path)
    try:
        book = db.get_book_by_id(conn, book_id)
        if book is None:
            raise ValueError(f"No book with book_id={book_id}.")
        sections_by_chapter: dict[int, list[db.Section]] = {}
        for section in db.list_sections(conn, book_id):
            sections_by_chapter.setdefault(section.chapter_index, []).append(section)
        return {
            "book_id": book.id,
            "slug": book.slug,
            "name": book.name,
            "chapters": [
                {
                    "chapter_index": chapter.chapter_index,
                    "title": chapter.title,
                    "page_start": chapter.page_start,
                    "sections": [
                        {
                            "section_index": section.section_index,
                            "title": _readable_section_title(section),
                            "page_start": section.page_start,
                            "approx_tokens": _approx_tokens(section.text),
                        }
                        for section in sections_by_chapter.get(chapter.chapter_index, [])
                    ],
                }
                for chapter in db.list_chapters(conn, book_id)
            ],
        }
    finally:
        conn.close()


def _readable_section_title(section: db.Section) -> str:
    """The section's title, or a synthesized label+snippet for auto-split sections with none."""
    if section.title:
        return section.title
    snippet = " ".join(section.text.split())[:60].rstrip()
    label = f"Section {section.section_index + 1}"
    return f"{label} -- {snippet}..." if snippet else label


def _approx_tokens(text: str) -> int:
    """Cheap size hint: tokens approximated as chars/4, matching chunk.py's convention."""
    return max(1, len(text) // 4)


@mcp.tool()
def search_book(
    question: str,
    book_id: int | None = None,
    top_k: int | None = None,
    author: str | None = None,
    source_type: str | None = None,
) -> list[dict]:
    """Search the library and return cited, scored excerpts.

    Combines semantic and keyword search by default. Optionally scope to
    one book (book_id), or filter by exact author or source_type (one of
    "pdf", "epub", "markdown", "text").
    """
    config = load_config()
    ollama_client.ensure_models_available([config.embedding_model])
    client = ollama_client.RealOllamaClient()
    conn = db.connect(config.db_path)
    try:
        results = search_mod.search(
            conn,
            client,
            config.embedding_model,
            question,
            book_id=book_id,
            top_k=top_k or config.top_k,
            expand=False,
            author=author,
            source_type=source_type,
            mode=config.retrieval_mode,
            candidate_pool_size=config.candidate_pool_size,
            rrf_k=config.rrf_k,
        )
        return [
            {
                "book_id": r.book_id,
                "book_slug": r.book_slug,
                "book_name": r.book_name,
                "chapter_index": r.chapter_index,
                "chapter_title": r.chapter_title,
                "section_index": r.section_index,
                "section_title": r.section_title,
                "page_start": r.page_start,
                "score": r.score,
                "text": r.raw_text,
            }
            for r in results
        ]
    finally:
        conn.close()


@mcp.tool()
def get_section(book_id: int, chapter_index: int, section_index: int) -> dict:
    """Fetch one section's full text and summary.

    The parent of a search_book hit, or a section located via
    get_book_outline's chapter_index/section_index map.
    """
    config = load_config()
    conn = db.connect(config.db_path)
    try:
        section = db.get_section(conn, book_id, chapter_index, section_index)
        if section is None:
            raise ValueError(
                f"No section book_id={book_id} chapter_index={chapter_index} "
                f"section_index={section_index}."
            )
        return {
            "book_id": section.book_id,
            "chapter_index": section.chapter_index,
            "section_index": section.section_index,
            "title": section.title,
            "summary": section.summary,
            "text": section.text,
            "page_start": section.page_start,
        }
    finally:
        conn.close()


@mcp.tool()
def book_status() -> list[dict]:
    """Report ingest status (chapter/section counts, chunk counts per stage) for every book."""
    config = load_config()
    conn = db.connect(config.db_path)
    try:
        result = []
        for book in db.list_books(conn):
            result.append(
                {
                    "book_id": book.id,
                    "slug": book.slug,
                    "name": book.name,
                    "chapters": db.chapter_count(conn, book.id),
                    "sections": db.section_count(conn, book.id),
                    "chunks": db.counts_by_status(conn, book.id),
                }
            )
        return result
    finally:
        conn.close()


def run() -> None:
    """Start the server over stdio."""
    mcp.run(transport="stdio")
