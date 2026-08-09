import json

import numpy as np

from matryoshka_search.demo.cli import format_results, load_catalog, run_search


def _orthogonal_catalog(n_items: int, dim: int) -> np.ndarray:
    """Same fixture shape as tests/test_funnel_search.py: a small nonzero
    baseline in every dimension plus a distinct per-item spike, so no
    truncated prefix is ever the zero vector."""
    catalog = np.full((n_items, dim), 0.01)
    for i in range(n_items):
        catalog[i, i % dim] += 1.0
    return catalog


def test_run_search_returns_expected_index_and_timing_keys():
    dim = 8
    catalog = _orthogonal_catalog(5, dim)
    target_item = catalog[3].copy()

    def encode_text(text: str) -> np.ndarray:
        return target_item

    def project(x: np.ndarray) -> np.ndarray:
        return x  # identity: skip an actual trained head for this test

    indices, timings = run_search(
        "anything", encode_text, project, catalog, low_dim=4, shortlist_size=5, k=2
    )

    assert 3 in indices
    assert set(timings) == {"encode_seconds", "project_seconds", "search_seconds", "total_seconds"}
    assert all(v >= 0 for v in timings.values())


def test_format_results_includes_item_id_title_and_path():
    indices = np.array([1, 0])
    item_ids = ["B001", "B002"]
    titles = {"B001": "First product", "B002": "Second product"}
    image_paths = {"B001": "images/B001.jpg", "B002": "images/B002.jpg"}

    lines = format_results(indices, item_ids, titles, image_paths)

    assert len(lines) == 2
    assert "B002" in lines[0] and "Second product" in lines[0] and "images/B002.jpg" in lines[0]
    assert "B001" in lines[1] and "First product" in lines[1]


def test_format_results_handles_unknown_item_id_gracefully():
    lines = format_results(np.array([0]), ["B999"], titles={}, image_paths={})
    assert "<unknown title>" in lines[0]
    assert "<unknown path>" in lines[0]


def test_load_catalog_looks_up_by_item_id_not_position(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "n_products": 2,
                "seed": 0,
                "products": [
                    {"item_id": "B001", "title": "First", "image_path": "images/B001.jpg"},
                    {"item_id": "B002", "title": "Second", "image_path": "images/B002.jpg"},
                ],
            }
        )
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # item_ids.json order is intentionally reversed vs. the manifest, to
    # confirm lookups go through item_id and not row position.
    (cache_dir / "item_ids.json").write_text(json.dumps(["B002", "B001"]))
    np.save(cache_dir / "image_embeddings.npy", np.zeros((2, 4)))

    item_ids, titles, image_paths, image_embeddings = load_catalog(manifest_path, cache_dir)

    assert item_ids == ["B002", "B001"]
    assert titles["B001"] == "First"
    assert image_paths["B002"] == "images/B002.jpg"
    assert image_embeddings.shape == (2, 4)
