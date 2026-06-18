"""In-house recursive text splitter (no langchain dependency).

Tokens are approximated as chars/4. Splits on the first separator that
actually breaks the text up, recursing through progressively finer
separators, then re-merges pieces up to chunk_size with overlap carried
from the tail of the previous chunk.

Always called with one section's text at a time, so a chunk can never
cross a section boundary -- the caller is responsible for slicing text by
section before it ever reaches this module.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_section(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """Split one section's text into overlapping ~chunk_size-token pieces."""
    return split_text(text, chunk_size, overlap)


def split_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """Split text into ~chunk_size-token pieces with ~overlap-token overlap."""
    max_chars = chunk_size * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN
    pieces = _split_recursive(text.strip(), max_chars, _SEPARATORS)
    return _merge_with_overlap(pieces, max_chars, overlap_chars)


def _split_recursive(text: str, max_chars: int, separators: list[str]) -> list[str]:
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    sep, rest = separators[0], separators[1:]
    raw = text.split(sep)
    if len(raw) == 1:
        return _split_recursive(text, max_chars, rest)
    # Reattach the separator to the end of every part but the last, so
    # joining the results reproduces `text` exactly -- no punctuation or
    # whitespace is ever discarded, however dense or run-on the prose is.
    parts = [p + sep for p in raw[:-1]] + [raw[-1]]
    result = []
    for part in parts:
        if part:
            result.extend(_split_recursive(part, max_chars, rest))
    return result


def _merge_with_overlap(pieces: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    merged: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            merged.append(current.strip())
        tail = current[-overlap_chars:] if overlap_chars else ""
        carried = tail + piece
        current = carried if len(carried) <= max_chars else piece
    if current:
        merged.append(current.strip())
    return merged
