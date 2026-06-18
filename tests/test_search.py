"""Cosine retrieval ranking, book scoping, and section expansion, with hand-built vectors."""

from __future__ import annotations

import sqlite3

import pytest

from book_rag import db, search
from fakes import FakeOllamaClient


def _seed_chunk(
    conn: sqlite3.Connection,
    book_id: int,
    chunk_index: int,
    text: str,
    vector: list[float],
) -> None:
    db.insert_chunk(conn, book_id, 0, 0, chunk_index, text, 1)
    db.set_chunk_context(conn, book_id, 0, 0, chunk_index, f"ctx-{chunk_index}")
    db.set_chunk_embedding(conn, book_id, 0, 0, chunk_index, db.serialize_embedding(vector))


def test_search_ranks_by_cosine_similarity_with_known_vectors(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, "Sec", "section text", 1)
    _seed_chunk(conn, book_id, 0, "chunk 0", [1.0, 0.0])
    _seed_chunk(conn, book_id, 1, "chunk 1", [0.0, 1.0])
    _seed_chunk(conn, book_id, 2, "chunk 2", [0.7071, 0.7071])
    conn.commit()
    client = FakeOllamaClient(vectors={"query": [1.0, 0.0]})

    results = search.search(conn, client, "embed-model", "query", top_k=3)

    assert [r.raw_text for r in results] == ["chunk 0", "chunk 2", "chunk 1"]
    assert results[0].score > results[1].score > results[2].score
    assert results[0].book_name == "Book"
    assert results[0].chapter_title == "Ch"
    assert results[0].section_index == 0
    assert results[0].page_start == 1


def test_search_book_id_scopes_results_to_one_book(conn: sqlite3.Connection) -> None:
    b1 = db.insert_book(conn, "b1", "Book One", "h1", 1)
    db.upsert_chapter(conn, b1, 0, "Ch", 1)
    db.upsert_section(conn, b1, 0, 0, None, "text", 1)
    _seed_chunk(conn, b1, 0, "b1 chunk", [1.0, 0.0])

    b2 = db.insert_book(conn, "b2", "Book Two", "h2", 1)
    db.upsert_chapter(conn, b2, 0, "Ch", 1)
    db.upsert_section(conn, b2, 0, 0, None, "text", 1)
    _seed_chunk(conn, b2, 0, "b2 chunk", [1.0, 0.0])
    conn.commit()
    client = FakeOllamaClient(vectors={"q": [1.0, 0.0]})

    all_results = search.search(conn, client, "m", "q", book_id=None, top_k=10)
    assert {r.book_id for r in all_results} == {b1, b2}

    scoped_results = search.search(conn, client, "m", "q", book_id=b1, top_k=10)
    assert {r.book_id for r in scoped_results} == {b1}


def test_search_expand_attaches_full_parent_section_text(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(
        conn, book_id, 0, 0, "Sec Title", "FULL SECTION TEXT spanning more than one chunk", 1
    )
    _seed_chunk(conn, book_id, 0, "partial chunk text", [1.0, 0.0])
    conn.commit()
    client = FakeOllamaClient(vectors={"q": [1.0, 0.0]})

    results = search.search(conn, client, "m", "q", expand=True)

    assert results[0].raw_text == "partial chunk text"
    assert results[0].section_text == "FULL SECTION TEXT spanning more than one chunk"


def test_search_without_expand_leaves_section_text_unset(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, "Sec Title", "full section text", 1)
    _seed_chunk(conn, book_id, 0, "partial chunk text", [1.0, 0.0])
    conn.commit()
    client = FakeOllamaClient(vectors={"q": [1.0, 0.0]})

    results = search.search(conn, client, "m", "q", expand=False)

    assert results[0].section_text is None


def test_search_with_no_embedded_chunks_returns_empty(conn: sqlite3.Connection) -> None:
    client = FakeOllamaClient()

    results = search.search(conn, client, "m", "anything")

    assert results == []


def test_search_filters_by_author(conn: sqlite3.Connection) -> None:
    b1 = db.insert_book(conn, "b1", "Book One", "h1", 1, author="Ada Lovelace")
    db.upsert_chapter(conn, b1, 0, "Ch", 1)
    db.upsert_section(conn, b1, 0, 0, None, "text", 1)
    _seed_chunk(conn, b1, 0, "ada chunk", [1.0, 0.0])

    b2 = db.insert_book(conn, "b2", "Book Two", "h2", 1, author="Bob")
    db.upsert_chapter(conn, b2, 0, "Ch", 1)
    db.upsert_section(conn, b2, 0, 0, None, "text", 1)
    _seed_chunk(conn, b2, 0, "bob chunk", [1.0, 0.0])
    conn.commit()
    client = FakeOllamaClient(vectors={"q": [1.0, 0.0]})

    results = search.search(conn, client, "m", "q", author="Ada Lovelace", top_k=10)

    assert {r.book_id for r in results} == {b1}


def test_search_filters_by_source_type(conn: sqlite3.Connection) -> None:
    b1 = db.insert_book(conn, "b1", "Book One", "h1", 1, source_type="epub")
    db.upsert_chapter(conn, b1, 0, "Ch", 1)
    db.upsert_section(conn, b1, 0, 0, None, "text", 1)
    _seed_chunk(conn, b1, 0, "epub chunk", [1.0, 0.0])

    b2 = db.insert_book(conn, "b2", "Book Two", "h2", 1, source_type="pdf")
    db.upsert_chapter(conn, b2, 0, "Ch", 1)
    db.upsert_section(conn, b2, 0, 0, None, "text", 1)
    _seed_chunk(conn, b2, 0, "pdf chunk", [1.0, 0.0])
    conn.commit()
    client = FakeOllamaClient(vectors={"q": [1.0, 0.0]})

    results = search.search(conn, client, "m", "q", source_type="epub", top_k=10)

    assert {r.book_id for r in results} == {b1}


def test_search_author_and_source_type_filters_compose(conn: sqlite3.Connection) -> None:
    b1 = db.insert_book(conn, "b1", "Book One", "h1", 1, author="Ada", source_type="epub")
    db.upsert_chapter(conn, b1, 0, "Ch", 1)
    db.upsert_section(conn, b1, 0, 0, None, "text", 1)
    _seed_chunk(conn, b1, 0, "ada epub chunk", [1.0, 0.0])

    b2 = db.insert_book(conn, "b2", "Book Two", "h2", 1, author="Ada", source_type="pdf")
    db.upsert_chapter(conn, b2, 0, "Ch", 1)
    db.upsert_section(conn, b2, 0, 0, None, "text", 1)
    _seed_chunk(conn, b2, 0, "ada pdf chunk", [1.0, 0.0])
    conn.commit()
    client = FakeOllamaClient(vectors={"q": [1.0, 0.0]})

    results = search.search(
        conn, client, "m", "q", author="Ada", source_type="epub", top_k=10
    )

    assert {r.book_id for r in results} == {b1}


def test_hybrid_mode_lets_a_strong_keyword_match_outrank_a_weaker_vector_match(
    conn: sqlite3.Connection,
) -> None:
    """Hybrid search must compose with, not replace, vector ranking.

    Chunk 0's embedding matches the query exactly; chunk 1's embedding is
    orthogonal to it but its text exactly matches the query keyword. Pure
    vector search must rank chunk 0 first; RRF-fused hybrid search must
    let chunk 1's unanimous keyword match pull it ahead.
    """
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, None, "text", 1)
    db.insert_chunk(conn, book_id, 0, 0, 0, "irrelevant filler text", 1)
    db.set_chunk_context(conn, book_id, 0, 0, 0, "ctx-0")
    db.set_chunk_embedding(conn, book_id, 0, 0, 0, db.serialize_embedding([1.0, 0.0]))
    db.index_chunk_fts(conn, book_id, 0, 0, 0, "irrelevant filler text", "ctx-0")
    db.insert_chunk(conn, book_id, 0, 0, 1, "wombat", 1)
    db.set_chunk_context(conn, book_id, 0, 0, 1, "ctx-1")
    db.set_chunk_embedding(conn, book_id, 0, 0, 1, db.serialize_embedding([0.0, 1.0]))
    db.index_chunk_fts(conn, book_id, 0, 0, 1, "wombat", "ctx-1")
    conn.commit()
    client = FakeOllamaClient(vectors={"wombat": [1.0, 0.0]})

    vector_only = search.search(conn, client, "m", "wombat", mode="vector", top_k=2)
    assert vector_only[0].raw_text == "irrelevant filler text"

    hybrid = search.search(conn, client, "m", "wombat", mode="hybrid", top_k=2)
    assert hybrid[0].raw_text == "wombat"


def test_search_rejects_unknown_mode(conn: sqlite3.Connection) -> None:
    client = FakeOllamaClient()

    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        search.search(conn, client, "m", "q", mode="bogus")
