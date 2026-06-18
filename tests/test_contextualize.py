"""Resumability of section summaries and chunk contextualization."""

from __future__ import annotations

import sqlite3

import pytest

from grimoire_beholder import contextualize, db
from fakes import FakeOllamaClient


def test_summarize_sections_only_fills_missing_summaries(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "B", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, None, "text a", 1)
    db.upsert_section(conn, book_id, 0, 1, None, "text b", 1)
    db.set_section_summary(conn, book_id, 0, 0, "existing summary")
    conn.commit()
    client = FakeOllamaClient()

    n = contextualize.summarize_sections(conn, client, "model", book_id)

    assert n == 1
    assert len(client.generate_calls) == 1
    assert db.get_section(conn, book_id, 0, 0).summary == "existing summary"
    assert db.get_section(conn, book_id, 0, 1).summary is not None


def test_contextualize_pending_moves_all_to_contextualized(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "B", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, None, "section text", 1)
    db.insert_chunk(conn, book_id, 0, 0, 0, "chunk 0", 1)
    db.insert_chunk(conn, book_id, 0, 0, 1, "chunk 1", 1)
    conn.commit()
    client = FakeOllamaClient()

    n = contextualize.contextualize_pending(conn, client, "model", book_id)

    assert n == 2
    assert db.counts_by_status(conn, book_id) == {
        "pending": 0,
        "contextualized": 2,
        "embedded": 0,
    }


class _CrashAfterN:
    """A client that fails after N successful generate() calls -- simulates a mid-run crash."""

    def __init__(self, n: int) -> None:
        self._n = n
        self.calls = 0

    def generate(self, model: str, system: str, prompt: str) -> str:
        self.calls += 1
        if self.calls > self._n:
            raise RuntimeError("simulated crash")
        return f"ctx-{self.calls}"

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError


def test_contextualize_pending_resumes_after_interrupt(conn: sqlite3.Connection) -> None:
    book_id = db.insert_book(conn, "b", "B", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    db.upsert_section(conn, book_id, 0, 0, None, "section text", 1)
    for i in range(5):
        db.insert_chunk(conn, book_id, 0, 0, i, f"chunk {i}", 1)
    conn.commit()
    crashy = _CrashAfterN(2)

    with pytest.raises(RuntimeError, match="simulated crash"):
        contextualize.contextualize_pending(conn, crashy, "model", book_id)

    counts = db.counts_by_status(conn, book_id)
    assert counts == {"pending": 3, "contextualized": 2, "embedded": 0}

    fresh = FakeOllamaClient()
    n = contextualize.contextualize_pending(conn, fresh, "model", book_id)

    assert n == 3
    assert db.counts_by_status(conn, book_id) == {
        "pending": 0,
        "contextualized": 5,
        "embedded": 0,
    }
