import json
from pathlib import Path

import numpy as np

from matryoshka_search.model.backbone import (
    precompute_and_cache_embeddings,
    precompute_from_manifest,
)


class FakeEncoder:
    """Deterministic stand-in for OpenClipBackbone — lets these tests verify
    caching/orchestration logic without loading torch or a real model."""

    def __init__(self):
        self.encode_images_calls = 0
        self.encode_texts_calls = 0

    def encode_images(self, image_paths, batch_size=64):
        self.encode_images_calls += 1
        return np.array([[float(i), float(i) + 0.5] for i in range(len(image_paths))])

    def encode_texts(self, texts, batch_size=256):
        self.encode_texts_calls += 1
        return np.array([[float(len(t)), 0.0] for t in texts])


def test_precompute_and_cache_embeddings_computes_once(tmp_path):
    encoder = FakeEncoder()
    image_paths = [Path("a.jpg"), Path("b.jpg")]
    texts = ["hello", "world!"]

    image_emb, text_emb = precompute_and_cache_embeddings(encoder, image_paths, texts, tmp_path)

    assert image_emb.shape == (2, 2)
    assert text_emb.shape == (2, 2)
    assert encoder.encode_images_calls == 1
    assert encoder.encode_texts_calls == 1
    assert (tmp_path / "image_embeddings.npy").exists()
    assert (tmp_path / "text_embeddings.npy").exists()


def test_precompute_and_cache_embeddings_skips_recompute_when_cached(tmp_path):
    encoder = FakeEncoder()
    image_paths = [Path("a.jpg")]
    texts = ["hello"]

    first_image_emb, first_text_emb = precompute_and_cache_embeddings(
        encoder, image_paths, texts, tmp_path
    )
    second_image_emb, second_text_emb = precompute_and_cache_embeddings(
        encoder, image_paths, texts, tmp_path
    )

    assert encoder.encode_images_calls == 1
    assert encoder.encode_texts_calls == 1
    np.testing.assert_array_equal(first_image_emb, second_image_emb)
    np.testing.assert_array_equal(first_text_emb, second_text_emb)


def _write_manifest(path, products):
    path.write_text(json.dumps({"n_products": len(products), "seed": 0, "products": products}))


def test_precompute_from_manifest_preserves_order_and_writes_item_ids(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"item_id": "B001", "title": "First product", "image_path": "images/B001.jpg"},
            {"item_id": "B002", "title": "Second product", "image_path": "images/B002.jpg"},
        ],
    )
    cache_dir = tmp_path / "cache"
    encoder = FakeEncoder()

    image_emb, text_emb, item_ids = precompute_from_manifest(
        manifest_path, cache_dir, backbone=encoder
    )

    assert item_ids == ["B001", "B002"]
    assert image_emb.shape == (2, 2)
    assert text_emb.shape == (2, 2)

    item_ids_cache = cache_dir / "item_ids.json"
    assert item_ids_cache.exists()
    assert json.loads(item_ids_cache.read_text()) == ["B001", "B002"]
