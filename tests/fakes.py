"""Deterministic test doubles for grimoire_beholder.ollama_client.OllamaClient.

Used everywhere in the suite so tests run with no Ollama daemon and no
models pulled, while still exercising the real pipeline code.
"""

from __future__ import annotations

import hashlib


def hash_vector(text: str, dim: int = 8) -> list[float]:
    """A stable, content-derived vector: the same text always yields the same vector."""
    digest = hashlib.sha256(text.encode()).digest()
    return [(b / 255.0) * 2 - 1 for b in digest[:dim]]


class FakeOllamaClient:
    """Canned generate(), hash-based embed() -- implements OllamaClient with no network calls."""

    def __init__(self, vectors: dict[str, list[float]] | None = None, dim: int = 8) -> None:
        self._vectors = dict(vectors or {})
        self._dim = dim
        self.generate_calls: list[tuple[str, str, str]] = []
        self.embed_batches: list[list[str]] = []

    def generate(self, model: str, system: str, prompt: str) -> str:
        self.generate_calls.append((model, system, prompt))
        return f"context::{prompt[:40]}"

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        self.embed_batches.append(list(inputs))
        return [self._vectors.get(text, hash_vector(text, self._dim)) for text in inputs]
