"""Vector retrieval strategy: brute-force cosine similarity over embedded chunks."""

from __future__ import annotations

import sqlite3

import numpy as np

from .. import db
from .strategies import RankedHit


class VectorStrategy:
    """Cosine similarity over embedded chunks already fetched by the caller.

    Takes `rows` at construction, rather than querying for them itself, so
    `search.search()` can fetch the filtered corpus once and reuse it both
    here and for the final result rows -- avoiding a second identical
    `db.get_search_rows` call on every search.
    """

    name = "vector"

    def __init__(self, rows: list[db.SearchRow]) -> None:
        self._rows = rows

    def run(
        self,
        conn: sqlite3.Connection,
        question: str,
        query_vector: np.ndarray,
        book_id: int | None,
        author: str | None,
        source_type: str | None,
        pool_size: int,
    ) -> list[RankedHit]:
        rows = self._rows
        if not rows:
            return []

        matrix = np.stack([db.deserialize_embedding(row.embedding) for row in rows])
        query_norm = query_vector / np.linalg.norm(query_vector)
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        scores = matrix_norm @ query_norm

        top_indices = np.argsort(-scores)[:pool_size]
        return [
            RankedHit(
                key=(rows[i].book_id, rows[i].chapter_index, rows[i].section_index, rows[i].chunk_index),
                score=float(scores[i]),
            )
            for i in top_indices
        ]
