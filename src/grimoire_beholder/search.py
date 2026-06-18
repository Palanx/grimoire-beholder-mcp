"""Shared retrieval: one `search()` used by both the CLI and the MCP server.

Embeds the question (the only Ollama call here -- no LLM, no cloud calls),
then runs one or two RetrievalStrategy implementations over the library's
contextualized, embedded chunks:

- `mode="vector"`: cosine similarity only, exactly as before hybrid search
  existed.
- `mode="hybrid"` (the default): cosine similarity AND FTS5/BM25 keyword
  search, each over the same contextualized chunks and the same filters,
  fused with Reciprocal Rank Fusion. Hybrid search augments contextual
  retrieval, it doesn't replace it -- both arms rank the exact same
  chunk corpus the section-based contextualization pipeline produced.

Optionally expands each hit with its full parent section's text so a
caller gets enough surrounding context without a second lookup.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from . import db, ollama_client
from .retrieval import FtsStrategy, RankedHit, VectorStrategy, reciprocal_rank_fusion

_MODES = ("hybrid", "vector")


@dataclass
class Result:
    book_id: int
    book_slug: str
    book_name: str
    chapter_title: str
    chapter_index: int
    section_title: str | None
    section_index: int
    page_start: int
    raw_text: str
    score: float
    section_text: str | None = None


def search(
    conn: sqlite3.Connection,
    client: ollama_client.OllamaClient,
    embedding_model: str,
    question: str,
    book_id: int | None = None,
    top_k: int = 5,
    expand: bool = False,
    author: str | None = None,
    source_type: str | None = None,
    mode: str = "hybrid",
    candidate_pool_size: int = 50,
    rrf_k: int = 60,
) -> list[Result]:
    """Return the top_k most relevant embedded chunks to `question`.

    Scoped to one book if `book_id` is given, further filtered by `author`
    and/or `source_type` if given, otherwise ranked across the whole
    library. If `expand` is set, each result also carries its parent
    section's full text. `mode="vector"` disables the FTS5 arm and falls
    back to pure cosine ranking.
    """
    if mode not in _MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {_MODES}.")

    db.ensure_embedding_model(conn, embedding_model)
    query_vector = _embed_query(client, embedding_model, question)
    pool_size = max(top_k, candidate_pool_size)

    # Fetched once and reused for both vector ranking and result-row lookup
    # below -- VectorStrategy used to re-fetch this same filtered set itself.
    rows = db.get_search_rows(conn, book_id=book_id, author=author, source_type=source_type)

    vector_hits = VectorStrategy(rows).run(
        conn, question, query_vector, book_id, author, source_type, pool_size
    )
    if mode == "vector":
        fused = [(hit.key, hit.score) for hit in vector_hits[:top_k]]
    else:
        fts_hits = FtsStrategy().run(
            conn, question, query_vector, book_id, author, source_type, pool_size
        )
        fused = reciprocal_rank_fusion([vector_hits, fts_hits], k=rrf_k)[:top_k]

    if not fused:
        return []

    row_by_key = {(r.book_id, r.chapter_index, r.section_index, r.chunk_index): r for r in rows}

    results = []
    for key, score in fused:
        row = row_by_key.get(key)
        if row is None:
            continue
        section_text = None
        if expand:
            section = db.get_section(conn, row.book_id, row.chapter_index, row.section_index)
            section_text = section.text if section else None
        results.append(
            Result(
                book_id=row.book_id,
                book_slug=row.book_slug,
                book_name=row.book_name,
                chapter_title=row.chapter_title,
                chapter_index=row.chapter_index,
                section_title=row.section_title,
                section_index=row.section_index,
                page_start=row.page_start,
                raw_text=row.raw_text,
                score=score,
                section_text=section_text,
            )
        )
    return results


def _embed_query(client: ollama_client.OllamaClient, model: str, question: str) -> np.ndarray:
    vectors = client.embed(model, [question])
    return np.asarray(vectors[0], dtype="<f4")


def format_results(results: list[Result]) -> str:
    """Render results as citation blocks ready to paste into a conversation."""
    if not results:
        return "No embedded chunks found. Run `ingest` first."
    blocks = []
    for r in results:
        section = f", {r.section_title}" if r.section_title else ""
        header = f"[{r.book_name} -- {r.chapter_title}{section}, p.{r.page_start}] (score={r.score:.3f})"
        body = r.section_text if r.section_text is not None else r.raw_text
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)
