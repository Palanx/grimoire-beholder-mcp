"""MCP server: exactly five read-only tools, and search_book/get_section composition."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from grimoire_beholder import db, mcp_server
from grimoire_beholder.config import Config
from fakes import FakeOllamaClient

_CONFIG_KWARGS = dict(
    llm_model="m",
    embedding_model="e",
    chunk_size=600,
    chunk_overlap=80,
    section_split_tokens=3000,
    embed_batch_size=16,
    top_k=5,
    retrieval_mode="hybrid",
    candidate_pool_size=50,
    rrf_k=60,
    num_ctx=4096,
)


def test_exactly_five_read_only_tools_are_registered() -> None:
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = sorted(t.name for t in tools)

    assert names == [
        "book_status",
        "get_book_outline",
        "get_section",
        "list_books",
        "search_book",
    ]
    mutating_keywords = ("ingest", "delete", "remove", "write")
    assert not any(kw in name for name in names for kw in mutating_keywords)


def test_search_book_and_get_section_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "mcp_test.db"
    conn = db.connect(str(db_path))
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Ch", 1)
    section_text = "Full section text, including the chunk's excerpt right here, and more besides."
    db.upsert_section(conn, book_id, 0, 0, "Sec Title", section_text, 1)
    db.insert_chunk(conn, book_id, 0, 0, 0, "the chunk's excerpt", 1)
    db.set_chunk_context(conn, book_id, 0, 0, 0, "ctx")
    db.set_chunk_embedding(conn, book_id, 0, 0, 0, db.serialize_embedding([1.0, 0.0]))
    conn.commit()
    conn.close()

    config = Config(db_path=str(db_path), **_CONFIG_KWARGS)
    fake_client = FakeOllamaClient(vectors={"the beginning": [1.0, 0.0]})
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server.ollama_client, "ensure_models_available", lambda required: None)
    monkeypatch.setattr(mcp_server.ollama_client, "RealOllamaClient", lambda **kwargs: fake_client)

    hits = mcp_server.search_book("the beginning", top_k=1)
    assert len(hits) == 1
    top = hits[0]
    assert top["text"] == "the chunk's excerpt"

    section = mcp_server.get_section(top["book_id"], top["chapter_index"], top["section_index"])

    assert top["text"] in section["text"]
    assert section["text"] == section_text


def test_list_books_and_book_status_report_every_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "mcp_test.db"
    conn = db.connect(str(db_path))
    for slug in ("book-a", "book-b", "book-c"):
        db.insert_book(conn, slug, slug.title(), f"hash-{slug}", 1)
    conn.commit()
    conn.close()

    config = Config(db_path=str(db_path), **_CONFIG_KWARGS)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)

    books = mcp_server.list_books()
    statuses = mcp_server.book_status()

    assert {b["slug"] for b in books} == {"book-a", "book-b", "book-c"}
    assert {s["slug"] for s in statuses} == {"book-a", "book-b", "book-c"}


def test_get_book_outline_returns_chapter_section_tree_without_full_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "mcp_test.db"
    conn = db.connect(str(db_path))
    book_id = db.insert_book(conn, "b", "Book", "h", 1)
    db.upsert_chapter(conn, book_id, 0, "Chapter One", 1)
    db.upsert_chapter(conn, book_id, 1, "Chapter Two", 50)
    db.upsert_section(conn, book_id, 0, 0, "Intro", "Short intro text.", 1)
    long_text = "auto split section body text that keeps going. " * 10
    db.upsert_section(conn, book_id, 0, 1, None, long_text, 10)
    db.upsert_section(conn, book_id, 1, 0, "Finale", "The end.", 50)
    conn.commit()
    conn.close()

    config = Config(db_path=str(db_path), **_CONFIG_KWARGS)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)

    outline = mcp_server.get_book_outline(book_id)

    assert outline["book_id"] == book_id
    assert outline["slug"] == "b"

    chapters = outline["chapters"]
    assert [c["chapter_index"] for c in chapters] == [0, 1]
    assert [c["title"] for c in chapters] == ["Chapter One", "Chapter Two"]
    assert [c["page_start"] for c in chapters] == [1, 50]

    ch0_sections = chapters[0]["sections"]
    assert [s["section_index"] for s in ch0_sections] == [0, 1]
    assert ch0_sections[0]["title"] == "Intro"
    assert ch0_sections[0]["page_start"] == 1
    assert ch0_sections[0]["approx_tokens"] == len("Short intro text.") // 4

    untitled = ch0_sections[1]
    assert untitled["title"] is not None
    assert "Section 2" in untitled["title"]
    assert "auto split section body text" in untitled["title"]
    assert untitled["approx_tokens"] == len(long_text) // 4

    ch1_sections = chapters[1]["sections"]
    assert ch1_sections[0]["title"] == "Finale"

    dumped = str(outline)
    assert long_text not in dumped
    assert "The end." not in dumped


def test_get_book_outline_raises_clearly_for_missing_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "mcp_test.db"
    db.connect(str(db_path)).close()

    config = Config(db_path=str(db_path), **_CONFIG_KWARGS)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)

    with pytest.raises(ValueError, match="book_id=999"):
        mcp_server.get_book_outline(999)
