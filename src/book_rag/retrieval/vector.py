"""Vector retrieval strategy: brute-force cosine similarity over embedded chunks."""

from __future__ import annotations

import sqlite3

import numpy as np

from .. import db
from .strategies import RankedHit


class VectorStrategy:
    name = "vector"

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
        rows = db.get_search_rows(conn, book_id=book_id, author=author, source_type=source_type)
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
