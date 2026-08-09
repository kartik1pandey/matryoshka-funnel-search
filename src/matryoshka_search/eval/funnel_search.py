"""Two-stage "funnel search": cheap low-dimension candidate generation over
the full catalog, followed by accurate full-dimension reranking of the
shortlist only.

This module *is* the named production pattern documented in
docs/01_research_background.md and docs/02_architecture.md, implemented as
retrieval code rather than only described in prose. See
docs/adr/0002-brute-force-vs-ann-index.md for why this is brute-force at
this project's scale rather than an ANN index.
"""

from __future__ import annotations

import numpy as np


def brute_force_search(query_emb: np.ndarray, catalog_emb: np.ndarray, k: int) -> np.ndarray:
    """Rank every catalog item against a query by cosine similarity.

    Args:
        query_emb: (dim,) or (n_queries, dim) query embedding(s).
        catalog_emb: (n_items, dim) catalog embeddings.
        k: number of top results to return.

    Returns:
        (n_queries, k) array of catalog indices, ranked best-first. If
        `query_emb` was 1-D, returns a (k,) array instead.
    """
    single_query = query_emb.ndim == 1
    q = query_emb[None, :] if single_query else query_emb

    q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
    c_norm = catalog_emb / np.linalg.norm(catalog_emb, axis=1, keepdims=True)
    similarity = q_norm @ c_norm.T  # (n_queries, n_items)

    k = min(k, catalog_emb.shape[0])
    top_k = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    return top_k[0] if single_query else top_k


def funnel_search(
    query_emb: np.ndarray,
    catalog_emb: np.ndarray,
    low_dim: int,
    shortlist_size: int,
    final_k: int,
) -> np.ndarray:
    """Stage 1 (low-dim, full catalog) + Stage 2 (full-dim, shortlist only).

    Args:
        query_emb: (dim,) query embedding at full dimension.
        catalog_emb: (n_items, dim) catalog embeddings at full dimension.
        low_dim: truncation length used for the cheap Stage 1 pass.
        shortlist_size: number of candidates Stage 1 hands to Stage 2 (K).
        final_k: number of final results to return after Stage 2 reranking.

    Returns:
        (final_k,) array of catalog indices, ranked best-first.
    """
    if low_dim > query_emb.shape[-1]:
        raise ValueError("low_dim cannot exceed the embedding's full dimension")

    shortlist = brute_force_search(query_emb[:low_dim], catalog_emb[:, :low_dim], k=shortlist_size)
    reranked_within_shortlist = brute_force_search(
        query_emb, catalog_emb[shortlist], k=min(final_k, shortlist.shape[0])
    )
    return shortlist[reranked_within_shortlist]
