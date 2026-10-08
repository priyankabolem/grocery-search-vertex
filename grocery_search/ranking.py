"""Blend retrieval relevance with model scores to produce the final ranking.

Kept free of cloud dependencies so it can be unit tested in isolation.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any


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
    ``model_score`` and ``final_score``. Without model scores (or on a length mismatch)
    the original retrieval order is kept, so the service degrades gracefully.
    Ties on ``final_score`` keep retrieval order (Python's sort is stable).
    """
    ranked = deepcopy(list(docs))
    for i, doc in enumerate(ranked):
        doc["search_score"] = position_score(i)

    if not model_scores or len(model_scores) != len(ranked):
        return ranked

    for doc, score in zip(ranked, model_scores, strict=True):
        doc["model_score"] = float(score)
        doc["final_score"] = search_weight * doc["search_score"] + model_weight * doc["model_score"]

    ranked.sort(key=lambda d: d["final_score"], reverse=True)
    return ranked
