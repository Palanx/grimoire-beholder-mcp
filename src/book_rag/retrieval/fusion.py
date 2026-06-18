"""Reciprocal Rank Fusion: combine several ranked hit lists into one.

score(chunk) = sum over every ranking it appears in of 1 / (k + rank)

`rank` is the 1-based position within that ranking, not the raw score --
this is precisely what makes RRF usable to combine scores that live on
incomparable scales (cosine similarity in [-1, 1] vs. an unbounded BM25
weight): only relative order within each list matters.
"""

from __future__ import annotations

from .strategies import ChunkKey, RankedHit


def reciprocal_rank_fusion(
    rankings: list[list[RankedHit]], k: int = 60
) -> list[tuple[ChunkKey, float]]:
    """Fuse multiple best-first rankings, returning (key, fused_score) best-first."""
    scores: dict[ChunkKey, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.key] = scores.get(hit.key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
