"""Typer CLI: slug resolution / exit codes, confirm-skip logic, option forwarding.

cli.py is "thin" by design (ARCHITECTURE.md) -- no retrieval or ingestion logic
-- but thin isn't risk-free: it still resolves a --book/slug argument to a
book_id with a real exit-code branch, skips a confirmation prompt under
--yes, and forwards an optional --mode override into search.search. None of
that was covered by any test before this file existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from grimoire_beholder import cli, db
from grimoire_beholder.config import Config

runner = CliRunner()

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
)


def _config(db_path: Path) -> Config:
    return Config(db_path=str(db_path), **_CONFIG_KWARGS)


def test_delete_unknown_slug_exits_with_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "lib.db"
    db.connect(str(db_path)).close()
    monkeypatch.setattr(cli, "load_config", lambda: _config(db_path))

    result = runner.invoke(cli.app, ["delete", "missing-slug", "--yes"])

    assert result.exit_code == 1
    assert "No book with slug 'missing-slug'" in result.output


def test_delete_with_yes_skips_confirmation_and_removes_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "lib.db"
    conn = db.connect(str(db_path))
    db.insert_book(conn, "b", "Book", "h", 1)
    conn.commit()
    conn.close()
    monkeypatch.setattr(cli, "load_config", lambda: _config(db_path))

    result = runner.invoke(cli.app, ["delete", "b", "--yes"])

    assert result.exit_code == 0
    conn = db.connect(str(db_path))
    assert db.get_book_by_slug(conn, "b") is None
    conn.close()


def test_delete_without_confirmation_aborts_and_keeps_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "lib.db"
    conn = db.connect(str(db_path))
    db.insert_book(conn, "b", "Book", "h", 1)
    conn.commit()
    conn.close()
    monkeypatch.setattr(cli, "load_config", lambda: _config(db_path))

    result = runner.invoke(cli.app, ["delete", "b"], input="n\n")

    assert result.exit_code == 0
    assert "Aborted" in result.output
    conn = db.connect(str(db_path))
    assert db.get_book_by_slug(conn, "b") is not None
    conn.close()


def test_query_unknown_book_slug_exits_with_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "lib.db"
    db.connect(str(db_path)).close()
    monkeypatch.setattr(cli, "load_config", lambda: _config(db_path))
    monkeypatch.setattr(cli.ollama_client, "ensure_models_available", lambda required: None)

    result = runner.invoke(cli.app, ["query", "a question", "--book", "missing-slug"])

    assert result.exit_code == 1
    assert "No book with slug 'missing-slug'" in result.output


def test_query_forwards_mode_override_to_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "lib.db"
    db.connect(str(db_path)).close()
    monkeypatch.setattr(cli, "load_config", lambda: _config(db_path))
    monkeypatch.setattr(cli.ollama_client, "ensure_models_available", lambda required: None)

    captured: dict = {}

    def fake_search(conn, client, embedding_model, question, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli.search_mod, "search", fake_search)

    result = runner.invoke(cli.app, ["query", "a question", "--mode", "vector"])

    assert result.exit_code == 0
    assert captured["mode"] == "vector"
