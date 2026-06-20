"""Full ingest orchestration: idempotency, force-replace, stage ordering, section boundaries."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from grimoire_beholder import db, ingest
from grimoire_beholder.config import Config
from grimoire_beholder.sources import pdf
from fakes import FakeOllamaClient

_CONFIG = Config(
    llm_model="fake-llm",
    embedding_model="fake-embed",
    chunk_size=10,
    chunk_overlap=2,
    section_split_tokens=150,
    embed_batch_size=4,
    top_k=5,
    db_path=":memory:",
    retrieval_mode="hybrid",
    candidate_pool_size=50,
    rrf_k=60,
    num_ctx=4096,
)


def test_run_ingest_reaches_fully_embedded_status(
    conn: sqlite3.Connection, toc_pdf_path: Path
) -> None:
    client = FakeOllamaClient()

    book_id = ingest.run_ingest(conn, client, _CONFIG, toc_pdf_path)

    counts = db.counts_by_status(conn, book_id)
    total = sum(counts.values())
    assert total > 0
    assert counts["embedded"] == total


@pytest.mark.parametrize("pdf_fixture", ["toc_pdf_path", "flat_pdf_path"])
def test_chunks_never_cross_section_boundaries(
    conn: sqlite3.Connection, pdf_fixture: str, request: pytest.FixtureRequest
) -> None:
    pdf_path = request.getfixturevalue(pdf_fixture)
    client = FakeOllamaClient()

    book_id = ingest.run_ingest(conn, client, _CONFIG, pdf_path)

    chunks = db.get_chunks_by_status(conn, "embedded", book_id)
    assert chunks
    for chunk in chunks:
        section = db.get_section(conn, book_id, chunk.chapter_index, chunk.section_index)
        assert section is not None
        assert chunk.raw_text.strip() in section.text


def test_ingest_runs_generate_stages_fully_before_any_embed(
    conn: sqlite3.Connection, toc_pdf_path: Path
) -> None:
    class _OrderTrackingClient(FakeOllamaClient):
        def __init__(self) -> None:
            super().__init__()
            self.call_order: list[str] = []

        def generate(self, model: str, system: str, prompt: str) -> str:
            self.call_order.append("generate")
            return super().generate(model, system, prompt)

        def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
            self.call_order.append("embed")
            return super().embed(model, inputs)

    client = _OrderTrackingClient()

    ingest.run_ingest(conn, client, _CONFIG, toc_pdf_path)

    last_generate = max(i for i, c in enumerate(client.call_order) if c == "generate")
    first_embed = min(i for i, c in enumerate(client.call_order) if c == "embed")
    assert last_generate < first_embed


def test_reingesting_same_pdf_is_idempotent(conn: sqlite3.Connection, toc_pdf_path: Path) -> None:
    client = FakeOllamaClient()
    book_id_1 = ingest.run_ingest(conn, client, _CONFIG, toc_pdf_path)
    counts_1 = db.counts_by_status(conn, book_id_1)

    book_id_2 = ingest.run_ingest(conn, client, _CONFIG, toc_pdf_path)

    assert book_id_2 == book_id_1
    assert len(db.list_books(conn)) == 1
    assert db.counts_by_status(conn, book_id_1) == counts_1


def test_same_slug_different_content_requires_force(
    conn: sqlite3.Connection, toc_pdf_path: Path, flat_pdf_path: Path
) -> None:
    client = FakeOllamaClient()
    ingest.run_ingest(conn, client, _CONFIG, toc_pdf_path, name="mybook")

    with pytest.raises(RuntimeError, match="already exists"):
        ingest.run_ingest(conn, client, _CONFIG, flat_pdf_path, name="mybook")

    ingest.run_ingest(conn, client, _CONFIG, flat_pdf_path, name="mybook", force=True)

    books = db.list_books(conn)
    assert len(books) == 1
    expected_hash = pdf.extract_book(str(flat_pdf_path)).content_hash
    assert db.get_book_by_slug(conn, "mybook").content_hash == expected_hash


def test_run_ingest_blocks_when_embedding_model_changed(
    conn: sqlite3.Connection, toc_pdf_path: Path, flat_pdf_path: Path
) -> None:
    client = FakeOllamaClient()
    ingest.run_ingest(conn, client, _CONFIG, toc_pdf_path)

    mismatched_config = replace(_CONFIG, embedding_model="a-different-embedding-model")
    with pytest.raises(RuntimeError, match="Embedding model mismatch"):
        ingest.run_ingest(conn, client, mismatched_config, flat_pdf_path, name="another")
