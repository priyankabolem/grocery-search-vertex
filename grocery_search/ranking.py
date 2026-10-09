"""Blend retrieval relevance with model scores to produce the final ranking.

Kept free of cloud dependencies so it can be unit tested in isolation.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

# Scores are rounded to this many decimals in the response. Rounding is applied only to the
# emitted values; ordering uses the unrounded blend so that near-ties are not reshuffled.
SCORE_DECIMALS = 3


def position_score(rank: int) -> float:
    """Reciprocal-rank relevance for a 0-based position: 1.0, 0.5, 0.33, ..."""
    if rank < 0:
        raise ValueError("rank must be non-negative")
    return 1.0 / (rank + 1)


def rerank(
    docs: Sequence[dict[str, Any]],
    model_scores: Sequence[float] | None,
    search_weight: float = 0.7,
    model_weight: float = 0.3,
) -> list[dict[str, Any]]:
    """Return a new, re-ranked list of documents; the input list is not modified.

    Each output document gets ``search_score`` and, when model scores are available,
    ``model_score`` and ``final_score``, each rounded to ``SCORE_DECIMALS`` decimals.
    Without model scores (or on a length mismatch) the original retrieval order is kept,
    so the service degrades gracefully.
    Ties on ``final_score`` keep retrieval order (Python's sort is stable).
    """
    ranked = deepcopy(list(docs))
    search_scores = [position_score(i) for i in range(len(ranked))]
    for doc, search_score in zip(ranked, search_scores, strict=True):
        doc["search_score"] = round(search_score, SCORE_DECIMALS)

    if not model_scores or len(model_scores) != len(ranked):
        return ranked

    blended = [
        search_weight * search_score + model_weight * float(model_score)
        for search_score, model_score in zip(search_scores, model_scores, strict=True)
    ]
    for doc, model_score, final_score in zip(ranked, model_scores, blended, strict=True):
        doc["model_score"] = round(float(model_score), SCORE_DECIMALS)
        doc["final_score"] = round(final_score, SCORE_DECIMALS)

    # Sort on the unrounded blend; sorted() is stable, so ties keep retrieval order.
    order = sorted(range(len(ranked)), key=lambda i: blended[i], reverse=True)
    return [ranked[i] for i in order]
