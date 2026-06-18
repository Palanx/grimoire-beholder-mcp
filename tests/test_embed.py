"""Batched, sequential, per-batch-committed embedding -- and resumability."""

from __future__ import annotations

import sqlite3

from book_rag import db, embed
from fakes import FakeOllamaClient


class _AssertingEmbedClient:
    """Checks, on every embed() call, that the previous batch was already committed."""

    def __init__(self, conn: sqlite3.Connection, batch_size: int, book_id: int) -> None:
        self._conn = conn
        self._batch_size = batch_size
        self._book_id = book_id
        self._call_index = 0
        self.batch_texts: list[list[str]] = []

    def generate(self, model: str, system: str, prompt: str) -> str:
        raise NotImplementedError

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        already_embedded = db.counts_by_status(self._conn, self._book_id).get("embedded", 0)
        assert already_embedded == self._call_index * self._batch_size
        self.batch_texts.append(list(inputs))
        self._call_index += 1
        return [[0.0, 0.0] for _ in inputs]


def _seed_contextualized_chunks(conn: sqlite3.Connection, book_id: int, count: int) -> None:
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, None, "section text", 1)
    for i in range(count):
        db.insert_chunk(conn, book_id, 0, 0, i, f"chunk {i}", 1)
        db.set_chunk_context(conn, book_id, 0, 0, i, f"ctx {i}")
    conn.commit()


def test_embed_pending_processes_one_batch_at_a_time_committing_each(
    conn: sqlite3.Connection,
) -> None:
    book_id = db.insert_book(conn, "b", "B", "h", 1)
    _seed_contextualized_chunks(conn, book_id, 5)
    client = _AssertingEmbedClient(conn, batch_size=2, book_id=book_id)

    n = embed.embed_pending(conn, client, "model", batch_size=2, book_id=book_id)

    assert n == 5
    assert [len(b) for b in client.batch_texts] == [2, 2, 1]
    assert db.counts_by_status(conn, book_id) == {
        "pending": 0,
        "contextualized": 0,
        "embedded": 5,
    }


def test_embed_pending_leaves_pending_and_already_embedded_rows_alone(
    conn: sqlite3.Connection,
) -> None:
    book_id = db.insert_book(conn, "b", "B", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, None, "section text", 1)

    db.insert_chunk(conn, book_id, 0, 0, 0, "c0", 1)
    db.set_chunk_context(conn, book_id, 0, 0, 0, "ctx0")
    db.set_chunk_embedding(conn, book_id, 0, 0, 0, b"SENTINEL")
    db.insert_chunk(conn, book_id, 0, 0, 1, "c1", 1)
    db.set_chunk_context(conn, book_id, 0, 0, 1, "ctx1")
    db.insert_chunk(conn, book_id, 0, 0, 2, "c2", 1)
    conn.commit()
    client = FakeOllamaClient()

    n = embed.embed_pending(conn, client, "model", batch_size=16, book_id=book_id)

    assert n == 1
    embedded_by_index = {
        r.chunk_index: r for r in db.get_chunks_by_status(conn, "embedded", book_id)
    }
    assert embedded_by_index[0].embedding == b"SENTINEL"
    assert embedded_by_index[1].embedding != b"SENTINEL"
    pending = db.get_chunks_by_status(conn, "pending", book_id)
    assert [c.chunk_index for c in pending] == [2]
