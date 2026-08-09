"""Retrieval-quality metrics: Recall@K, nDCG@K, and retention.

Kept framework-agnostic (plain NumPy in, plain NumPy out) so it can be used
against embeddings from either the JAX-trained Matryoshka head or the
PyTorch-side baseline without either framework being an import-time
dependency of this module.

See docs/04_methodology.md#evaluation-protocol for the definitions this
implements.
"""

from __future__ import annotations

import numpy as np


def _ranked_indices(similarity: np.ndarray) -> np.ndarray:
    """Return indices sorted by descending similarity, per query row."""
    return np.argsort(-similarity, axis=1, kind="stable")


def recall_at_k(similarity: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Fraction of queries whose true match appears in the top-k results.

    Args:
        similarity: (n_queries, n_items) similarity scores.
        relevant: (n_queries,) index of the single true match per query.
        k: cutoff rank.

    Returns:
        Mean recall@k across all queries, in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = _ranked_indices(similarity)[:, :k]
    hits = (top_k == relevant[:, None]).any(axis=1)
    return float(hits.mean())


def ndcg_at_k(similarity: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, for a single relevant item per query.

    With exactly one relevant item per query, the ideal DCG is always 1
    (the relevant item at rank 1), so nDCG@k reduces to
    1 / log2(rank + 2) if the true match is within the top-k, else 0,
    averaged across queries.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = _ranked_indices(similarity)[:, :k]
    n_queries = similarity.shape[0]
    scores = np.zeros(n_queries)
    for i in range(n_queries):
        matches = np.flatnonzero(ranked[i] == relevant[i])
        if matches.size:
            rank = int(matches[0])
            scores[i] = 1.0 / np.log2(rank + 2)
    return float(scores.mean())


def retention(metric_at_dim: dict[int, float], full_dim: int) -> dict[int, float]:
    """Express each dimension's metric as a fraction of the full-dimension metric.

    Args:
        metric_at_dim: mapping of dimension -> metric value (e.g. Recall@10).
        full_dim: the dimension to treat as the 100% reference point.

    Returns:
        Mapping of dimension -> retention fraction (metric_at_dim[d] / metric_at_dim[full_dim]).
    """
    reference = metric_at_dim[full_dim]
    if reference == 0:
        raise ValueError("full-dimension metric is zero; retention is undefined")
    return {dim: value / reference for dim, value in metric_at_dim.items()}
