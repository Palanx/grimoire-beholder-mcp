"""Batched, synchronous embedding of contextualized chunks.

One batch at a time, no concurrency: select up to `batch_size` chunks with
status='contextualized', send them as a single list to Ollama's batch
embed endpoint, wait for the response, store every embedding, index the
chunk into the FTS5 keyword index, set status='embedded', commit the
whole batch, then move to the next batch. A crash loses at most one
in-flight batch -- the next run re-embeds and re-indexes it, both
idempotent.
"""

from __future__ import annotations

import sqlite3

from tqdm import tqdm

from . import db, ollama_client


def embed_pending(
    conn: sqlite3.Connection,
    client: ollama_client.OllamaClient,
    model: str,
    batch_size: int = 16,
    book_id: int | None = None,
) -> int:
    """Embed every chunk with status='contextualized', batch by batch."""
    total = db.counts_by_status(conn, book_id).get("contextualized", 0)
    processed = 0
    with tqdm(total=total, desc="Embedding chunks") as progress:
        while True:
            batch = db.get_chunks_by_status(conn, "contextualized", book_id, limit=batch_size)
            if not batch:
                break
            inputs = [f"{chunk.context}\n\n{chunk.raw_text}" for chunk in batch]
            embeddings = client.embed(model, inputs)
            for chunk, vector in zip(batch, embeddings):
                db.set_chunk_embedding(
                    conn,
                    chunk.book_id,
                    chunk.chapter_index,
                    chunk.section_index,
                    chunk.chunk_index,
                    db.serialize_embedding(vector),
                )
                db.index_chunk_fts(
                    conn,
                    chunk.book_id,
                    chunk.chapter_index,
                    chunk.section_index,
                    chunk.chunk_index,
                    chunk.raw_text,
                    chunk.context,
                )
            conn.commit()
            processed += len(batch)
            progress.update(len(batch))
    return processed
