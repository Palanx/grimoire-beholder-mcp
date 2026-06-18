"""The RetrievalStrategy extension point and the value type it returns.

To add a new retrieval strategy: write a class with a `name` and a `run`
method matching `RetrievalStrategy` below, in its own module here. It
receives both the raw question and its embedded query vector so it can use
whichever (or both) it needs without changing the interface. Wire it into
`book_rag.search` if it should participate in fusion.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol

import numpy as np

ChunkKey = tuple[int, int, int, int]  # (book_id, chapter_index, section_index, chunk_index)


@dataclass
class RankedHit:
    key: ChunkKey
    score: float


class RetrievalStrategy(Protocol):
    name: str

    def run(
        self,
        conn: sqlite3.Connection,
        question: str,
        query_vector: np.ndarray,
        book_id: int | None,
        author: str | None,
        source_type: str | None,
        pool_size: int,
    ) -> list[RankedHit]: ...
