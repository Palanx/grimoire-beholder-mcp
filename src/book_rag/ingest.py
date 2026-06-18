"""Orchestrates one book's ingest: extract -> load -> summarize -> contextualize -> embed.

Content-hash idempotency: re-ingesting the same source file under the same
slug re-extracts and re-loads (both idempotent -- existing rows are left
untouched) and simply resumes contextualization/embedding wherever they
left off. A slug collision with *different* content is refused unless
`force=True`, which deletes the old book first.

The source file's format only matters to `sources.get_parser` -- everything
below that call (chunking, contextualization, embedding) is identical
regardless of whether the book came from a PDF, an EPUB, or plaintext.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from . import chunk as chunk_mod
from . import contextualize, db, embed, ollama_client, sources
from .config import Config


def slugify(name: str) -> str:
    """Turn a display name into a database-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "book"


def run_ingest(
    conn: sqlite3.Connection,
    client: ollama_client.OllamaClient,
    config: Config,
    source_path: Path,
    name: str | None = None,
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> int:
    """Ingest one source file into the library, fully resumable. Returns the book_id."""
    db.ensure_embedding_model(conn, config.embedding_model)

    progress(f"Extracting hierarchy from {source_path} ...")
    parser = sources.get_parser(source_path)
    extracted = parser.extract(source_path, config.section_split_tokens)

    display_name = name or extracted.title or source_path.stem
    slug = slugify(display_name)

    existing = db.get_book_by_slug(conn, slug)
    if existing is None:
        book_id = db.insert_book(
            conn,
            slug,
            display_name,
            extracted.content_hash,
            extracted.page_count,
            author=extracted.author,
            source_type=extracted.source_type,
        )
    elif existing.content_hash == extracted.content_hash:
        book_id = existing.id
        progress(f"'{slug}' already ingested with identical content -- resuming.")
    elif force:
        progress(f"'{slug}' exists with different content; --force given, replacing it.")
        db.delete_book(conn, existing.id)
        book_id = db.insert_book(
            conn,
            slug,
            display_name,
            extracted.content_hash,
            extracted.page_count,
            author=extracted.author,
            source_type=extracted.source_type,
        )
    else:
        raise RuntimeError(
            f"A book with slug '{slug}' already exists but its content differs "
            f"from {source_path}. Pass --force to replace it, or pass a different "
            "--name to give this source its own slug."
        )

    progress(f"Found {len(extracted.chapters)} chapter(s).")
    total_sections = 0
    for chapter in extracted.chapters:
        db.upsert_chapter(conn, book_id, chapter.chapter_index, chapter.title, chapter.page_start)
        for section in chapter.sections:
            db.upsert_section(
                conn,
                book_id,
                chapter.chapter_index,
                section.section_index,
                section.title,
                section.text,
                section.page_start,
            )
            total_sections += 1
            texts = chunk_mod.chunk_section(section.text, config.chunk_size, config.chunk_overlap)
            for chunk_index, text in enumerate(texts):
                db.insert_chunk(
                    conn,
                    book_id,
                    chapter.chapter_index,
                    section.section_index,
                    chunk_index,
                    text,
                    section.page_start,
                )
    conn.commit()
    progress(f"Loaded {total_sections} section(s).")

    contextualize.summarize_sections(conn, client, config.llm_model, book_id)
    contextualize.contextualize_pending(conn, client, config.llm_model, book_id)
    embed.embed_pending(conn, client, config.embedding_model, config.embed_batch_size, book_id)

    return book_id
