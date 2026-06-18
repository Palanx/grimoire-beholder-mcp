"""Retrieval strategies, RRF fusion, and the RetrievalStrategy extension point.

`grimoire_beholder.search` is the composition root that calls into this package; it
decides which strategies to run and how to fuse them. Nothing in here
knows about the CLI or the MCP server.
"""

from __future__ import annotations

from .fts import FtsStrategy
from .fusion import reciprocal_rank_fusion
from .strategies import ChunkKey, RankedHit, RetrievalStrategy
from .vector import VectorStrategy

__all__ = [
    "ChunkKey",
    "FtsStrategy",
    "RankedHit",
    "RetrievalStrategy",
    "VectorStrategy",
    "reciprocal_rank_fusion",
]
