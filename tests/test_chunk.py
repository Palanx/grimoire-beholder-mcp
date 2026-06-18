"""Deterministic invariants of the in-house text splitter."""

from __future__ import annotations

from grimoire_beholder.chunk import chunk_section


def test_chunk_size_approximately_respected() -> None:
    words = [f"tok{i:03d}" for i in range(300)]
    text = " ".join(words)

    chunks = chunk_section(text, chunk_size=10, overlap=2)

    max_chars = 10 * 4
    assert all(len(c) <= max_chars for c in chunks)


def test_overlap_present_between_consecutive_chunks() -> None:
    words = [f"w{i:03d}" for i in range(300)]
    text = " ".join(words)

    chunks = chunk_section(text, chunk_size=10, overlap=3)

    assert len(chunks) > 2
    for i in range(len(chunks) - 1):
        tail_words = set(chunks[i].split()[-2:])
        head_words = set(chunks[i + 1].split()[:4])
        assert tail_words & head_words, (tail_words, head_words)


def test_no_content_dropped_across_chunks() -> None:
    words = [f"tok{i:03d}" for i in range(250)]
    text = " ".join(words)

    chunks = chunk_section(text, chunk_size=12, overlap=2)

    seen: set[str] = set()
    for chunk in chunks:
        seen.update(chunk.split())
    assert seen == set(words)


def test_single_short_section_produces_one_chunk() -> None:
    text = "Just a short section, well under the size limit."

    chunks = chunk_section(text, chunk_size=600, overlap=80)

    assert chunks == [text]
