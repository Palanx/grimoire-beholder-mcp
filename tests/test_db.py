"""Schema-level invariants: idempotent natural keys, the embedding-model guard, and delete."""

from __future__ import annotations

import sqlite3

import pytest

from grimoire_beholder import db


def test_insert_chunk_idempotent_under_natural_key(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "slug", "Name", "hash1", 10)

    db.insert_chunk(conn, book_id, 0, 0, 0, "first text", 1)
    db.insert_chunk(conn, book_id, 0, 0, 0, "different text on re-run", 1)
    conn.commit()

    rows = db.get_chunks_by_status(conn, "pending", book_id)
    assert len(rows) == 1
    assert rows[0].raw_text == "first text"


def test_insert_chunk_unique_per_index_within_book(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "slug", "Name", "hash1", 10)

    db.insert_chunk(conn, book_id, 0, 0, 0, "a", 1)
    db.insert_chunk(conn, book_id, 0, 0, 1, "b", 1)
    db.insert_chunk(conn, book_id, 0, 1, 0, "c", 1)
    conn.commit()

    rows = db.get_chunks_by_status(conn, "pending", book_id)
    keys = {(r.chapter_index, r.section_index, r.chunk_index) for r in rows}
    assert keys == {(0, 0, 0), (0, 0, 1), (0, 1, 0)}


def test_ensure_embedding_model_stamps_on_first_call(conn: sqlite3.Connection) -> None:
    db.ensure_embedding_model(conn, "nomic-embed-text")

    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    assert row["value"] == "nomic-embed-text"


def test_ensure_embedding_model_matching_value_proceeds(conn: sqlite3.Connection) -> None:
    db.ensure_embedding_model(conn, "nomic-embed-text")

    db.ensure_embedding_model(conn, "nomic-embed-text")  # must not raise


def test_ensure_embedding_model_mismatch_raises(conn: sqlite3.Connection) -> None:
    db.ensure_embedding_model(conn, "nomic-embed-text")

    with pytest.raises(RuntimeError, match="Embedding model mismatch"):
        db.ensure_embedding_model(conn, "a-different-model")


def test_delete_book_removes_everything_and_leaves_other_books(conn: sqlite3.Connection) -> None:
    b1 = db.insert_book(conn, "book-one", "Book One", "h1", 5)
    db.upsert_chapter(conn, b1, 0, "Ch1", 1)
    db.upsert_section(conn, b1, 0, 0, None, "text1", 1)
    db.insert_chunk(conn, b1, 0, 0, 0, "chunk1", 1)

    b2 = db.insert_book(conn, "book-two", "Book Two", "h2", 5)
    db.upsert_chapter(conn, b2, 0, "Ch2", 1)
    db.upsert_section(conn, b2, 0, 0, None, "text2", 1)
    db.insert_chunk(conn, b2, 0, 0, 0, "chunk2", 1)
    conn.commit()

    db.delete_book(conn, b1)

    assert db.get_book_by_id(conn, b1) is None
    assert conn.execute("SELECT COUNT(*) AS n FROM chapters WHERE book_id = ?", (b1,)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM sections WHERE book_id = ?", (b1,)).fetchone()["n"] == 0
    assert conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE book_id = ?", (b1,)).fetchone()["n"] == 0

    assert db.get_book_by_id(conn, b2) is not None
    assert db.chapter_count(conn, b2) == 1
    assert db.section_count(conn, b2) == 1
    assert len(db.get_chunks_by_status(conn, "pending", b2)) == 1


def test_list_chapters_and_list_sections_are_ordered_by_index(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "slug", "Name", "hash1", 10)
    db.upsert_chapter(conn, book_id, 1, "Second Chapter", 20)
    db.upsert_chapter(conn, book_id, 0, "First Chapter", 1)
    db.upsert_section(conn, book_id, 0, 1, "Sec B", "text", 5)
    db.upsert_section(conn, book_id, 0, 0, "Sec A", "text", 1)
    db.upsert_section(conn, book_id, 1, 0, "Sec C", "text", 20)
    conn.commit()

    chapters = db.list_chapters(conn, book_id)
    assert [c.chapter_index for c in chapters] == [0, 1]
    assert [c.title for c in chapters] == ["First Chapter", "Second Chapter"]

    sections = db.list_sections(conn, book_id)
    assert [(s.chapter_index, s.section_index) for s in sections] == [(0, 0), (0, 1), (1, 0)]
    assert [s.title for s in sections] == ["Sec A", "Sec B", "Sec C"]


def test_upsert_section_preserves_existing_summary_on_reingest(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "slug", "Name", "hash1", 10)
    db.upsert_section(conn, book_id, 0, 0, "Title", "original text", 1)
    db.set_section_summary(conn, book_id, 0, 0, "a generated summary")

    db.upsert_section(conn, book_id, 0, 0, "Title", "original text", 1)

    section = db.get_section(conn, book_id, 0, 0)
    assert section.summary == "a generated summary"


# -- FTS5 keyword index ------------------------------------------------------


def test_check_fts5_available_wraps_operational_error_loudly() -> None:
    class _FailingConn:
        def execute(self, sql, *params):
            raise sqlite3.OperationalError("no such module: fts5")

    with pytest.raises(db.FTS5UnavailableError):
        db._check_fts5_available(_FailingConn())


def test_search_fts_finds_indexed_chunk_by_keyword(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.index_chunk_fts(conn, book_id, 0, 0, 0, "the quick brown fox jumps", "context")
    conn.commit()

    hits = db.search_fts(conn, '"fox"')

    assert len(hits) == 1
    key, score = hits[0]
    assert key == (book_id, 0, 0, 0)
    assert isinstance(score, float)


def test_search_fts_no_match_returns_empty(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.index_chunk_fts(conn, book_id, 0, 0, 0, "the quick brown fox jumps", "context")
    conn.commit()

    assert db.search_fts(conn, '"giraffe"') == []


def test_search_fts_filters_by_author_and_source_type(conn: sqlite3.Connection) -> None:
    b1 = db.insert_book(conn, "b1", "Book One", "h1", 1, author="Ada", source_type="epub")
    b2 = db.insert_book(conn, "b2", "Book Two", "h2", 1, author="Bob", source_type="pdf")
    db.index_chunk_fts(conn, b1, 0, 0, 0, "shared keyword text", "")
    db.index_chunk_fts(conn, b2, 0, 0, 0, "shared keyword text", "")
    conn.commit()

    by_author = db.search_fts(conn, '"keyword"', author="Ada")
    assert [key for key, _ in by_author] == [(b1, 0, 0, 0)]

    by_type = db.search_fts(conn, '"keyword"', source_type="pdf")
    assert [key for key, _ in by_type] == [(b2, 0, 0, 0)]


def test_rebuild_fts_index_repopulates_from_embedded_chunks(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, None, "text", 1)
    db.insert_chunk(conn, book_id, 0, 0, 0, "keyword chunk text", 1)
    db.set_chunk_context(conn, book_id, 0, 0, 0, "ctx")
    db.set_chunk_embedding(conn, book_id, 0, 0, 0, db.serialize_embedding([1.0, 0.0]))
    conn.commit()
    assert db.search_fts(conn, '"keyword"') == []  # not indexed yet -- embed_pending does that

    n = db.rebuild_fts_index(conn)

    assert n == 1
    hits = db.search_fts(conn, '"keyword"')
    assert [key for key, _ in hits] == [(book_id, 0, 0, 0)]

    n_again = db.rebuild_fts_index(conn)  # rebuild must not duplicate rows
    assert n_again == 1
    assert len(db.search_fts(conn, '"keyword"')) == 1


def test_delete_book_also_removes_its_fts_rows(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.index_chunk_fts(conn, book_id, 0, 0, 0, "keyword text", "")
    conn.commit()

    db.delete_book(conn, book_id)

    assert db.search_fts(conn, '"keyword"') == []
