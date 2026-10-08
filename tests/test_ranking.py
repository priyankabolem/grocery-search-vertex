import pytest

from grocery_search.ranking import position_score, rerank


def _docs(*ids):
    return [{"product_id": i} for i in ids]


def test_position_score_is_reciprocal_rank():
    assert position_score(0) == 1.0
    assert position_score(1) == 0.5
    assert position_score(3) == 0.25


def test_position_score_rejects_negative_rank():
    with pytest.raises(ValueError):
        position_score(-1)


def test_rerank_blends_scores_and_sorts():
    ranked = rerank(_docs("A", "B", "C"), [0.40, 0.85, 0.60])
    assert [d["product_id"] for d in ranked] == ["A", "B", "C"]
    assert ranked[0]["final_score"] == pytest.approx(0.82)
    assert ranked[1]["final_score"] == pytest.approx(0.605)


def test_model_score_can_promote_a_lower_ranked_item():
    ranked = rerank(_docs("A", "B"), [0.0, 1.0], search_weight=0.5, model_weight=0.5)
    assert [d["product_id"] for d in ranked] == ["B", "A"]


def test_without_model_scores_keeps_retrieval_order():
    ranked = rerank(_docs("A", "B", "C"), None)
    assert [d["product_id"] for d in ranked] == ["A", "B", "C"]
    assert "final_score" not in ranked[0]
    assert ranked[2]["search_score"] == pytest.approx(1 / 3)


def test_length_mismatch_falls_back_to_retrieval_order():
    ranked = rerank(_docs("A", "B"), [0.9])
    assert [d["product_id"] for d in ranked] == ["A", "B"]
    assert "model_score" not in ranked[0]


def test_empty_input_returns_empty_list():
    assert rerank([], None) == []
    assert rerank([], []) == []


def test_ties_keep_retrieval_order():
    ranked = rerank(_docs("A", "B"), [0.5, 0.5], search_weight=0.0, model_weight=1.0)
    assert [d["product_id"] for d in ranked] == ["A", "B"]


def test_input_is_not_mutated():
    docs = _docs("A", "B")
    rerank(docs, [0.1, 0.9])
    assert docs == [{"product_id": "A"}, {"product_id": "B"}]
