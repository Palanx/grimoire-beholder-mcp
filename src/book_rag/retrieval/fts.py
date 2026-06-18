"""Keyword retrieval strategy: SQLite FTS5 + BM25 over indexed chunks.

The question is tokenized and every term quoted and OR'd together, e.g.
`machine learning models` -> `"machine" OR "learning" OR "models"`. Quoting
each term neutralizes FTS5's own query syntax (`-`, `*`, `:`, parens, ...)
so arbitrary user text can never produce a MATCH syntax error; OR-ing them
favors recall, which is what a fusion-with-vector-search setup wants --
RRF naturally rewards chunks that also rank well on the vector side.
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np

from .. import db
from .strategies import RankedHit

_TERM_RE = re.compile(r"\w+")


class FtsStrategy:
    name = "fts"

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
        match_query = _sanitize_query(question)
        if not match_query:
            return []
        hits = db.search_fts(
            conn,
            match_query,
            book_id=book_id,
            author=author,
            source_type=source_type,
            limit=pool_size,
        )
        return [RankedHit(key=key, score=score) for key, score in hits]


def _sanitize_query(question: str) -> str | None:
    terms = _TERM_RE.findall(question)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)
