"""Section summaries and per-chunk context generation via Ollama.

Sections, not chapters, are summarized: a long, dense chapter (e.g. in a
philosophy or psychology book) is split into several sections precisely so
each one gets its own focused summary instead of one over-generic
chapter-level summary that would wash out the chapter's internal nuance.
Every chunk's context is then generated from its own section's summary.

Resumable by construction: section summaries are only generated where
`summary IS NULL`, and each chunk's context is committed immediately after
generation, so a crash loses at most one in-flight chunk.
"""

from __future__ import annotations

import sqlite3

from tqdm import tqdm

from . import db, ollama_client

_SUMMARY_SYSTEM_PROMPT = (
    "You write concise summaries of a section of a book for a search index. "
    "Respond with ONLY a 2-3 sentence summary, no preamble."
)

_CONTEXT_SYSTEM_PROMPT = (
    "You write short context notes that situate an excerpt within its "
    "section, to help a search system match it to relevant queries. "
    "Respond with ONLY a 1-2 sentence context note, no preamble."
)


def summarize_sections(
    conn: sqlite3.Connection,
    client: ollama_client.OllamaClient,
    model: str,
    book_id: int | None = None,
) -> int:
    """Generate a one-time summary per section, skipping ones already set."""
    sections = db.get_sections_needing_summary(conn, book_id)
    for section in tqdm(sections, desc="Summarizing sections"):
        title = section.title or f"(untitled section {section.section_index})"
        prompt = f"Section title: {title}\n\nSection text:\n{section.text}"
        summary = client.generate(model, _SUMMARY_SYSTEM_PROMPT, prompt)
        db.set_section_summary(
            conn, section.book_id, section.chapter_index, section.section_index, summary
        )
    return len(sections)


def contextualize_pending(
    conn: sqlite3.Connection,
    client: ollama_client.OllamaClient,
    model: str,
    book_id: int | None = None,
) -> int:
    """Generate and store context for every chunk with status='pending'."""
    chunks = db.get_chunks_by_status(conn, "pending", book_id)
    for chunk in tqdm(chunks, desc="Contextualizing chunks"):
        section = db.get_section(conn, chunk.book_id, chunk.chapter_index, chunk.section_index)
        summary = section.summary if section and section.summary else "(no summary available)"
        prompt = (
            f"Section summary: {summary}\n\n"
            f"Excerpt:\n{chunk.raw_text}\n\n"
            "Write the short context note for this excerpt."
        )
        context = client.generate(model, _CONTEXT_SYSTEM_PROMPT, prompt)
        db.set_chunk_context(
            conn,
            chunk.book_id,
            chunk.chapter_index,
            chunk.section_index,
            chunk.chunk_index,
            context,
        )
    return len(chunks)
