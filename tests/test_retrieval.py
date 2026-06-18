"""Reciprocal Rank Fusion: known-input ordering and the documented k=60 formula."""

from __future__ import annotations

import pytest

from grimoire_beholder.retrieval import RankedHit, reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_well_in_both_lists() -> None:
    vector_ranking = [RankedHit(("a",), 0.9), RankedHit(("b",), 0.8), RankedHit(("c",), 0.7)]
    fts_ranking = [RankedHit(("c",), 5.0), RankedHit(("a",), 4.0), RankedHit(("d",), 3.0)]

    fused = reciprocal_rank_fusion([vector_ranking, fts_ranking], k=60)
    fused_keys = [key for key, _ in fused]

    # "a" is rank 1 in both lists -- the only candidate that can win.
    assert fused_keys[0] == ("a",)
    # "d" appears in only one list, at rank 3 -- the weakest candidate.
    assert fused_keys[-1] == ("d",)
    assert all(fused[i][1] >= fused[i + 1][1] for i in range(len(fused) - 1))


def test_rrf_score_matches_the_documented_formula() -> None:
    ranking = [RankedHit(("x",), 999.0), RankedHit(("y",), -999.0)]

    fused = reciprocal_rank_fusion([ranking], k=60)

    assert fused == [(("x",), pytest.approx(1 / 61)), (("y",), pytest.approx(1 / 62))]


def test_rrf_is_blind_to_raw_score_scale_only_rank_matters() -> None:
    cosine_like = [RankedHit(("p",), 0.31), RankedHit(("q",), 0.30)]
    bm25_like = [RankedHit(("p",), -12.4), RankedHit(("q",), -50.1)]

    fused = reciprocal_rank_fusion([cosine_like, bm25_like], k=60)

    assert [key for key, _ in fused] == [("p",), ("q",)]


def test_rrf_of_empty_rankings_is_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_ranking_preserves_order() -> None:
    ranking = [RankedHit(("a",), 1.0), RankedHit(("b",), 0.5), RankedHit(("c",), 0.1)]

    fused = reciprocal_rank_fusion([ranking], k=60)

    assert [key for key, _ in fused] == [("a",), ("b",), ("c",)]
