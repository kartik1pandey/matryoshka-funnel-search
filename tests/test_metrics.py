import numpy as np
import pytest

from matryoshka_search.eval.metrics import ndcg_at_k, recall_at_k, retention


def test_recall_at_k_perfect_match():
    # 3 queries, 3 items; similarity is an identity matrix -> query i's best match is item i.
    similarity = np.eye(3) * 10
    relevant = np.array([0, 1, 2])
    assert recall_at_k(similarity, relevant, k=1) == 1.0


def test_recall_at_k_worst_case():
    similarity = np.array([[10.0, 5.0, 1.0]])  # query 0 ranks item 0 first
    relevant = np.array([2])  # but the true match is item 2 (ranked last)
    assert recall_at_k(similarity, relevant, k=1) == 0.0
    assert recall_at_k(similarity, relevant, k=3) == 1.0


def test_recall_at_k_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        recall_at_k(np.eye(2), np.array([0, 1]), k=0)


def test_ndcg_at_k_rank_one_is_perfect():
    similarity = np.array([[10.0, 5.0, 1.0]])
    relevant = np.array([0])
    assert ndcg_at_k(similarity, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_lower_rank_scores_less_than_one():
    similarity = np.array([[10.0, 5.0, 1.0]])
    relevant = np.array([1])  # true match is ranked 2nd
    score = ndcg_at_k(similarity, relevant, k=3)
    assert 0.0 < score < 1.0
    assert score == pytest.approx(1.0 / np.log2(3))


def test_ndcg_at_k_zero_when_outside_k():
    similarity = np.array([[10.0, 5.0, 1.0]])
    relevant = np.array([2])
    assert ndcg_at_k(similarity, relevant, k=1) == 0.0


def test_retention_full_dim_is_one():
    metric_at_dim = {512: 0.9, 256: 0.85, 64: 0.7}
    result = retention(metric_at_dim, full_dim=512)
    assert result[512] == pytest.approx(1.0)
    assert result[256] == pytest.approx(0.85 / 0.9)
    assert result[64] == pytest.approx(0.7 / 0.9)


def test_retention_rejects_zero_reference():
    with pytest.raises(ValueError):
        retention({512: 0.0, 64: 0.1}, full_dim=512)
