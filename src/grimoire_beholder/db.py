"""SQLite-backed multi-book library: schema, CRUD, and status queries.

The database file is the entire checkpoint for every book in the library.
Each chunk has a `status` column (`pending -> contextualized -> embedded`);
every ingest stage resumes by querying that column, so a crash mid-run
loses at most one in-flight chunk (contextualization, committed per row)
or one in-flight batch (embedding, committed per batch) -- never a whole
book, let alone the whole library.

The `meta` table holds index-wide settings, in particular the embedding
model the index was built with -- see `ensure_embedding_model`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    author TEXT,
    source_type TEXT NOT NULL DEFAULT 'pdf',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapters (
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    PRIMARY KEY (book_id, chapter_index)
);

CREATE TABLE IF NOT EXISTS sections (
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_index INTEGER NOT NULL,
    section_index INTEGER NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    summary TEXT,
    page_start INTEGER NOT NULL,
    PRIMARY KEY (book_id, chapter_index, section_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_index INTEGER NOT NULL,
    section_index INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    context TEXT,
    embedding BLOB,
    page_start INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    PRIMARY KEY (book_id, chapter_index, section_index, chunk_index)
);

CREATE TABLE IF NOT EXISTS pdf_toc_cache (
    content_hash TEXT PRIMARY KEY,
    chapters_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chapters_book ON chapters(book_id);
CREATE INDEX IF NOT EXISTS idx_sections_book ON sections(book_id, chapter_index);
CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_chunks_book_status ON chunks(book_id, status);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    raw_text,
    context,
    book_id UNINDEXED,
    chapter_index UNINDEXED,
    section_index UNINDEXED,
    chunk_index UNINDEXED
);
"""

_BOOKS_MIGRATION_COLUMNS = {
    "author": "ALTER TABLE books ADD COLUMN author TEXT",
    "source_type": "ALTER TABLE books ADD COLUMN source_type TEXT NOT NULL DEFAULT 'pdf'",
}


class FTS5UnavailableError(RuntimeError):
    """Raised when the running SQLite build lacks the FTS5 extension."""


@dataclass
class Book:
    id: int
    slug: str
    name: str
    content_hash: str
    page_count: int
    author: str | None = None
    source_type: str = "pdf"


@dataclass
class Chapter:
    book_id: int
    chapter_index: int
    title: str
    page_start: int


@dataclass
class Section:
    book_id: int
    chapter_index: int
    section_index: int
    title: str | None
    text: str
    summary: str | None
    page_start: int


@dataclass
class Chunk:
    book_id: int
    chapter_index: int
    section_index: int
    chunk_index: int
    raw_text: str
    context: str | None
    embedding: bytes | None
    page_start: int
    status: str


@dataclass
class SearchRow:
    """One embedded chunk joined with the book/chapter/section it belongs to."""

    book_id: int
    book_slug: str
    book_name: str
    chapter_index: int
    chapter_title: str
    section_index: int
    section_title: str | None
    chunk_index: int
    raw_text: str
    context: str | None
    embedding: bytes
    page_start: int


def connect(db_path: str) -> sqlite3.Connection:
    """Open the database, enabling WAL mode and foreign keys, ensuring the schema exists.

    Hybrid search depends on FTS5, which is a compile-time SQLite option.
    We probe for it explicitly and fail loudly here rather than letting a
    raw `OperationalError` surface later from deep inside a search call.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _check_fts5_available(conn)
    conn.executescript(_SCHEMA)
    _migrate_books_table(conn)
    conn.commit()
    return conn


def _check_fts5_available(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_probe")
    except sqlite3.OperationalError as exc:
        raise FTS5UnavailableError(
            "This SQLite build does not have the FTS5 extension compiled in, but "
            "grimoire-beholder-mcp's hybrid search requires it. Check `PRAGMA compile_options` "
            "for ENABLE_FTS5, or use a Python/SQLite build that includes it -- "
            "the official python.org installers and Homebrew's sqlite3 both do."
        ) from exc


def _migrate_books_table(conn: sqlite3.Connection) -> None:
    """Add columns to `books` that didn't exist in earlier versions of this schema."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(books)").fetchall()}
    for column, ddl in _BOOKS_MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(ddl)


# -- embedding-model coherence guard -----------------------------------------


def ensure_embedding_model(conn: sqlite3.Connection, embedding_model: str) -> None:
    """Fail loudly if `embedding_model` differs from the one this index was built with.

    The first ingest into a fresh database stamps its embedding model into
    `meta`. Every later ingest -- of any book -- must match it, or the new
    vectors would silently land in a different vector space than the
    existing ones and similarity search would become meaningless.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('embedding_model', ?)", (embedding_model,)
        )
        conn.commit()
        return
    stored = row["value"]
    if stored != embedding_model:
        raise RuntimeError(
            f"Embedding model mismatch: this index was built with '{stored}', "
            f"but config.toml currently specifies '{embedding_model}'. Mixing "
            "embedding models in one index would silently corrupt similarity "
            f"search across the whole library. Either set embedding_model back "
            f"to '{stored}' in config.toml, or point db_path at a fresh "
            "database to start a new index with the new model."
        )


# -- books ---------------------------------------------------------------


def get_book_by_slug(conn: sqlite3.Connection, slug: str) -> Book | None:
    row = conn.execute("SELECT * FROM books WHERE slug = ?", (slug,)).fetchone()
    return _row_to_book(row) if row else None


def get_book_by_id(conn: sqlite3.Connection, book_id: int) -> Book | None:
    row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
    return _row_to_book(row) if row else None


def list_books(conn: sqlite3.Connection) -> list[Book]:
    rows = conn.execute("SELECT * FROM books ORDER BY id").fetchall()
    return [_row_to_book(r) for r in rows]


def insert_book(
    conn: sqlite3.Connection,
    slug: str,
    name: str,
    content_hash: str,
    page_count: int,
    author: str | None = None,
    source_type: str = "pdf",
) -> int:
    cur = conn.execute(
        "INSERT INTO books (slug, name, content_hash, page_count, author, source_type) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (slug, name, content_hash, page_count, author, source_type),
    )
    conn.commit()
    return cur.lastrowid


def delete_book(conn: sqlite3.Connection, book_id: int) -> None:
    """Delete a book and every chapter/section/chunk/FTS row under it, transactionally."""
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM chunks_fts WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM sections WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _row_to_book(row: sqlite3.Row) -> Book:
    return Book(
        row["id"],
        row["slug"],
        row["name"],
        row["content_hash"],
        row["page_count"],
        row["author"],
        row["source_type"],
    )


# -- chapters --------------------------------------------------------------


def upsert_chapter(
    conn: sqlite3.Connection, book_id: int, chapter_index: int, title: str, page_start: int
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO chapters (book_id, chapter_index, title, page_start) "
        "VALUES (?, ?, ?, ?)",
        (book_id, chapter_index, title, page_start),
    )


def list_chapters(conn: sqlite3.Connection, book_id: int) -> list[Chapter]:
    rows = conn.execute(
        "SELECT * FROM chapters WHERE book_id = ? ORDER BY chapter_index", (book_id,)
    ).fetchall()
    return [_row_to_chapter(r) for r in rows]


def chapter_count(conn: sqlite3.Connection, book_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM chapters WHERE book_id = ?", (book_id,)
    ).fetchone()["n"]


def _row_to_chapter(row: sqlite3.Row) -> Chapter:
    return Chapter(row["book_id"], row["chapter_index"], row["title"], row["page_start"])


# -- PDF TOC cache (LLM-extracted, offset-resolved chapter bounds) ----------


def get_cached_pdf_toc(conn: sqlite3.Connection, content_hash: str) -> list[tuple[str, int, int]] | None:
    """Return a previously validated (title, page_start, page_end) chapter list, if cached.

    Keyed by content_hash rather than book_id: extraction runs before the
    book row exists, and this also makes re-ingesting the same file under a
    different slug reuse the cache for free.
    """
    row = conn.execute(
        "SELECT chapters_json FROM pdf_toc_cache WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if row is None:
        return None
    return [tuple(entry) for entry in json.loads(row["chapters_json"])]


def set_cached_pdf_toc(
    conn: sqlite3.Connection, content_hash: str, chapter_bounds: list[tuple[str, int, int]]
) -> None:
    """Persist a validated, offset-resolved chapter list so re-ingestion never re-calls the LLM."""
    conn.execute(
        "INSERT OR REPLACE INTO pdf_toc_cache (content_hash, chapters_json) VALUES (?, ?)",
        (content_hash, json.dumps([list(entry) for entry in chapter_bounds])),
    )
    conn.commit()


def section_count(conn: sqlite3.Connection, book_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM sections WHERE book_id = ?", (book_id,)
    ).fetchone()["n"]


# -- sections ----------------------------------------------------------------


def upsert_section(
    conn: sqlite3.Connection,
    book_id: int,
    chapter_index: int,
    section_index: int,
    title: str | None,
    text: str,
    page_start: int,
) -> None:
    """Insert or replace a section, preserving any summary already generated for it.

    Re-ingesting the same PDF reproduces identical section text (content
    hash matched), so carrying the existing summary forward avoids
    redoing LLM work on an unchanged re-run.
    """
    conn.execute(
        """INSERT INTO sections (book_id, chapter_index, section_index, title, text, summary, page_start)
           VALUES (?, ?, ?, ?, ?,
               (SELECT summary FROM sections
                WHERE book_id = ? AND chapter_index = ? AND section_index = ?),
               ?)
           ON CONFLICT(book_id, chapter_index, section_index) DO UPDATE SET
               title = excluded.title, text = excluded.text, page_start = excluded.page_start""",
        (
            book_id,
            chapter_index,
            section_index,
            title,
            text,
            book_id,
            chapter_index,
            section_index,
            page_start,
        ),
    )


def get_section(
    conn: sqlite3.Connection, book_id: int, chapter_index: int, section_index: int
) -> Section | None:
    row = conn.execute(
        "SELECT * FROM sections WHERE book_id = ? AND chapter_index = ? AND section_index = ?",
        (book_id, chapter_index, section_index),
    ).fetchone()
    return _row_to_section(row) if row else None


def list_sections(conn: sqlite3.Connection, book_id: int) -> list[Section]:
    """Every section in a book, ordered by chapter/section index -- the outline's leaves."""
    rows = conn.execute(
        "SELECT * FROM sections WHERE book_id = ? ORDER BY chapter_index, section_index",
        (book_id,),
    ).fetchall()
    return [_row_to_section(r) for r in rows]


def get_sections_needing_summary(
    conn: sqlite3.Connection, book_id: int | None = None
) -> list[Section]:
    if book_id is None:
        rows = conn.execute(
            "SELECT * FROM sections WHERE summary IS NULL "
            "ORDER BY book_id, chapter_index, section_index"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sections WHERE book_id = ? AND summary IS NULL "
            "ORDER BY chapter_index, section_index",
            (book_id,),
        ).fetchall()
    return [_row_to_section(r) for r in rows]


def set_section_summary(
    conn: sqlite3.Connection, book_id: int, chapter_index: int, section_index: int, summary: str
) -> None:
    conn.execute(
        "UPDATE sections SET summary = ? "
        "WHERE book_id = ? AND chapter_index = ? AND section_index = ?",
        (summary, book_id, chapter_index, section_index),
    )
    conn.commit()


def _row_to_section(row: sqlite3.Row) -> Section:
    return Section(
        row["book_id"],
        row["chapter_index"],
        row["section_index"],
        row["title"],
        row["text"],
        row["summary"],
        row["page_start"],
    )


# -- chunks --------------------------------------------------------------


def insert_chunk(
    conn: sqlite3.Connection,
    book_id: int,
    chapter_index: int,
    section_index: int,
    chunk_index: int,
    raw_text: str,
    page_start: int,
) -> None:
    """Insert a chunk as 'pending'; never overwrites an existing row (idempotent re-ingest)."""
    conn.execute(
        """INSERT INTO chunks
           (book_id, chapter_index, section_index, chunk_index, raw_text, page_start, status)
           VALUES (?, ?, ?, ?, ?, ?, 'pending')
           ON CONFLICT(book_id, chapter_index, section_index, chunk_index) DO NOTHING""",
        (book_id, chapter_index, section_index, chunk_index, raw_text, page_start),
    )


def get_chunks_by_status(
    conn: sqlite3.Connection, status: str, book_id: int | None = None, limit: int | None = None
) -> list[Chunk]:
    sql = "SELECT * FROM chunks WHERE status = ?"
    params: list = [status]
    if book_id is not None:
        sql += " AND book_id = ?"
        params.append(book_id)
    sql += " ORDER BY book_id, chapter_index, section_index, chunk_index"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_chunk(r) for r in rows]


def set_chunk_context(
    conn: sqlite3.Connection,
    book_id: int,
    chapter_index: int,
    section_index: int,
    chunk_index: int,
    context: str,
) -> None:
    """Store generated context and advance status; commits immediately (per-row checkpoint)."""
    conn.execute(
        """UPDATE chunks SET context = ?, status = 'contextualized'
           WHERE book_id = ? AND chapter_index = ? AND section_index = ? AND chunk_index = ?""",
        (context, book_id, chapter_index, section_index, chunk_index),
    )
    conn.commit()


def set_chunk_embedding(
    conn: sqlite3.Connection,
    book_id: int,
    chapter_index: int,
    section_index: int,
    chunk_index: int,
    embedding: bytes,
) -> None:
    """Store an embedding and advance status. Caller commits once per batch."""
    conn.execute(
        """UPDATE chunks SET embedding = ?, status = 'embedded'
           WHERE book_id = ? AND chapter_index = ? AND section_index = ? AND chunk_index = ?""",
        (embedding, book_id, chapter_index, section_index, chunk_index),
    )


def counts_by_status(conn: sqlite3.Connection, book_id: int | None = None) -> dict[str, int]:
    if book_id is None:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM chunks GROUP BY status").fetchall()
    else:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM chunks WHERE book_id = ? GROUP BY status",
            (book_id,),
        ).fetchall()
    counts = {"pending": 0, "contextualized": 0, "embedded": 0}
    counts.update({row["status"]: row["n"] for row in rows})
    return counts


def get_search_rows(
    conn: sqlite3.Connection,
    book_id: int | None = None,
    author: str | None = None,
    source_type: str | None = None,
) -> list[SearchRow]:
    """Fetch every embedded chunk joined with its book/chapter/section metadata, for ranking."""
    sql = """
        SELECT b.id AS book_id, b.slug AS book_slug, b.name AS book_name,
               c.chapter_index AS chapter_index, c.title AS chapter_title,
               s.section_index AS section_index, s.title AS section_title,
               ch.chunk_index AS chunk_index, ch.raw_text AS raw_text,
               ch.context AS context, ch.embedding AS embedding, ch.page_start AS page_start
        FROM chunks ch
        JOIN books b ON b.id = ch.book_id
        JOIN chapters c ON c.book_id = ch.book_id AND c.chapter_index = ch.chapter_index
        JOIN sections s ON s.book_id = ch.book_id AND s.chapter_index = ch.chapter_index
                        AND s.section_index = ch.section_index
        WHERE ch.status = 'embedded'
    """
    params: list = []
    if book_id is not None:
        sql += " AND ch.book_id = ?"
        params.append(book_id)
    if author is not None:
        sql += " AND b.author = ?"
        params.append(author)
    if source_type is not None:
        sql += " AND b.source_type = ?"
        params.append(source_type)
    rows = conn.execute(sql, params).fetchall()
    return [
        SearchRow(
            row["book_id"],
            row["book_slug"],
            row["book_name"],
            row["chapter_index"],
            row["chapter_title"],
            row["section_index"],
            row["section_title"],
            row["chunk_index"],
            row["raw_text"],
            row["context"],
            row["embedding"],
            row["page_start"],
        )
        for row in rows
    ]


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        row["book_id"],
        row["chapter_index"],
        row["section_index"],
        row["chunk_index"],
        row["raw_text"],
        row["context"],
        row["embedding"],
        row["page_start"],
        row["status"],
    )


def serialize_embedding(vec) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


# -- full-text index (FTS5) -------------------------------------------------


def index_chunk_fts(
    conn: sqlite3.Connection,
    book_id: int,
    chapter_index: int,
    section_index: int,
    chunk_index: int,
    raw_text: str,
    context: str | None,
) -> None:
    """Add one chunk to the full-text index. Caller commits (same batch as its embedding)."""
    conn.execute(
        "INSERT INTO chunks_fts (book_id, chapter_index, section_index, chunk_index, "
        "raw_text, context) VALUES (?, ?, ?, ?, ?, ?)",
        (book_id, chapter_index, section_index, chunk_index, raw_text, context or ""),
    )


def search_fts(
    conn: sqlite3.Connection,
    match_query: str,
    book_id: int | None = None,
    author: str | None = None,
    source_type: str | None = None,
    limit: int = 50,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """BM25 full-text search over indexed chunks, best match first.

    Returns `((book_id, chapter_index, section_index, chunk_index), score)`
    pairs. SQLite's bm25() is more-negative-is-better; this negates it so
    higher is always better here, matching the cosine convention used by
    vector search -- callers never need to know the underlying sign.
    """
    sql = """
        SELECT f.book_id AS book_id, f.chapter_index AS chapter_index,
               f.section_index AS section_index, f.chunk_index AS chunk_index,
               bm25(chunks_fts) AS rank_score
        FROM chunks_fts f
        JOIN books b ON b.id = f.book_id
        WHERE chunks_fts MATCH ?
    """
    params: list = [match_query]
    if book_id is not None:
        sql += " AND f.book_id = ?"
        params.append(book_id)
    if author is not None:
        sql += " AND b.author = ?"
        params.append(author)
    if source_type is not None:
        sql += " AND b.source_type = ?"
        params.append(source_type)
    sql += " ORDER BY rank_score ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        (
            (row["book_id"], row["chapter_index"], row["section_index"], row["chunk_index"]),
            -row["rank_score"],
        )
        for row in rows
    ]


def rebuild_fts_index(conn: sqlite3.Connection, book_id: int | None = None) -> int:
    """Drop and repopulate the FTS index from every currently-embedded chunk.

    The index is normally populated incrementally as chunks are embedded;
    this exists so it can be rebuilt from scratch on demand (e.g. to
    recover a hand-edited database, or after restoring an old backup).
    """
    select_sql = (
        "SELECT book_id, chapter_index, section_index, chunk_index, raw_text, context "
        "FROM chunks WHERE status = 'embedded'"
    )
    if book_id is None:
        conn.execute("DELETE FROM chunks_fts")
        rows = conn.execute(select_sql).fetchall()
    else:
        conn.execute("DELETE FROM chunks_fts WHERE book_id = ?", (book_id,))
        rows = conn.execute(select_sql + " AND book_id = ?", (book_id,)).fetchall()
    for row in rows:
        index_chunk_fts(
            conn,
            row["book_id"],
            row["chapter_index"],
            row["section_index"],
            row["chunk_index"],
            row["raw_text"],
            row["context"],
        )
    conn.commit()
    return len(rows)
