import numpy as np
from matryoshka_search.eval.funnel_search import brute_force_search, funnel_search


def _orthogonal_catalog(n_items: int, dim: int) -> np.ndarray:
    """Each item has a small baseline in every dimension (so no truncated
    prefix is ever the zero vector -> no div-by-zero in L2 normalization)
    plus a distinct spike, making it unambiguously most similar to a query
    pointing at it whenever the spike's dimension is within the truncation."""
    catalog = np.full((n_items, dim), 0.01)
    for i in range(n_items):
        catalog[i, i % dim] += 1.0
    return catalog


def test_brute_force_search_single_query_returns_1d():
    catalog = _orthogonal_catalog(5, 8)
    query = catalog[2].copy()
    result = brute_force_search(query, catalog, k=1)
    assert result.shape == (1,)
    assert result[0] == 2


def test_brute_force_search_batch_returns_2d():
    catalog = _orthogonal_catalog(5, 8)
    queries = catalog[[0, 3]].copy()
    result = brute_force_search(queries, catalog, k=1)
    assert result.shape == (2, 1)
    assert result[0, 0] == 0
    assert result[1, 0] == 3


def test_brute_force_search_k_larger_than_catalog_is_clamped():
    catalog = _orthogonal_catalog(3, 8)
    result = brute_force_search(catalog[0], catalog, k=100)
    assert result.shape == (3,)


def test_funnel_search_recovers_true_match():
    catalog = _orthogonal_catalog(20, 32)
    query = catalog[5].copy()
    result = funnel_search(query, catalog, low_dim=8, shortlist_size=10, final_k=3)
    assert 5 in result


def test_funnel_search_rejects_low_dim_exceeding_full_dim():
    catalog = _orthogonal_catalog(5, 8)
    query = catalog[0].copy()
    try:
        funnel_search(query, catalog, low_dim=16, shortlist_size=3, final_k=1)
        raised = False
    except ValueError:
        raised = True
    assert raised
