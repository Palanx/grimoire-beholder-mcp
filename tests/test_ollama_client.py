"""<think>-block stripping, both as a pure function and inside RealOllamaClient.generate()."""

from __future__ import annotations

import pytest

from book_rag import ollama_client


def test_strip_think_removes_block() -> None:
    raw = "<think>internal reasoning that should never be stored</think>The answer is 42."

    assert ollama_client._strip_think(raw) == "The answer is 42."


def test_strip_think_is_a_noop_when_absent() -> None:
    raw = "Just a plain response."

    assert ollama_client._strip_think(raw) == "Just a plain response."


class _FakeResponse:
    def __init__(self, response: str) -> None:
        self.response = response


def test_real_client_generate_strips_think_before_returning(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_response = "<think>scratch work the model should not leak</think>Final summary text."

    def fake_generate(model: str, system: str, prompt: str, think: bool) -> _FakeResponse:
        return _FakeResponse(raw_response)

    monkeypatch.setattr(ollama_client.ollama, "generate", fake_generate)
    client = ollama_client.RealOllamaClient()

    result = client.generate("cogito:8b", "system prompt", "user prompt")

    assert result == "Final summary text."
    assert "<think>" not in result
