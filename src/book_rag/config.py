"""Configuration loading for book-rag, backed by a single config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULTS = {
    "llm_model": "cogito:8b",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 600,
    "chunk_overlap": 80,
    "section_split_tokens": 3000,
    "embed_batch_size": 16,
    "top_k": 5,
    "db_path": "book.db",
    "retrieval_mode": "hybrid",
    "candidate_pool_size": 50,
    "rrf_k": 60,
}


@dataclass(frozen=True)
class Config:
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    section_split_tokens: int
    embed_batch_size: int
    top_k: int
    db_path: str
    retrieval_mode: str
    candidate_pool_size: int
    rrf_k: int


def load_config(path: Path = Path("config.toml")) -> Config:
    """Load config.toml, falling back to built-in defaults for missing keys.

    Missing file or missing keys are not errors: every field has a sensible
    default so the pipeline runs with zero edits.
    """
    values = dict(_DEFAULTS)
    if path.exists():
        with path.open("rb") as f:
            values.update(tomllib.load(f))
    return Config(**{key: values[key] for key in _DEFAULTS})
