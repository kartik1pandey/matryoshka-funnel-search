import gzip
import json

import pytest

from matryoshka_search.data.abo_dataset import (
    Product,
    _english_title,
    _iter_listings,
    _load_image_index,
    select_subset,
)


def test_english_title_prefers_en_us():
    listing = {
        "item_name": [
            {"language_tag": "nl_NL", "value": "Dutch title"},
            {"language_tag": "en_GB", "value": "British title"},
            {"language_tag": "en_US", "value": "American title"},
        ]
    }
    assert _english_title(listing) == "American title"


def test_english_title_falls_back_to_any_en_tag():
    listing = {"item_name": [{"language_tag": "en_IN", "value": "Indian English title"}]}
    assert _english_title(listing) == "Indian English title"


def test_english_title_none_when_no_english():
    listing = {"item_name": [{"language_tag": "es_MX", "value": "Titulo en espanol"}]}
    assert _english_title(listing) is None


def test_english_title_none_when_missing_item_name():
    assert _english_title({}) is None


def test_load_image_index(tmp_path):
    csv_path = tmp_path / "images.csv.gz"
    with gzip.open(csv_path, "wt") as f:
        f.write("image_id,height,width,path\n")
        f.write("abc123,100,100,ab/abc123.jpg\n")
        f.write("def456,200,200,de/def456.jpg\n")

    index = _load_image_index(csv_path)
    assert index == {"abc123": "ab/abc123.jpg", "def456": "de/def456.jpg"}


def _write_listing_shard(path, listings):
    with gzip.open(path, "wt") as f:
        for listing in listings:
            f.write(json.dumps(listing) + "\n")


def test_iter_listings_reads_across_shards(tmp_path):
    shard_0 = tmp_path / "listings_0.json.gz"
    shard_1 = tmp_path / "listings_1.json.gz"
    _write_listing_shard(shard_0, [{"item_id": "A"}, {"item_id": "B"}])
    _write_listing_shard(shard_1, [{"item_id": "C"}])

    item_ids = [listing["item_id"] for listing in _iter_listings([shard_0, shard_1])]
    assert item_ids == ["A", "B", "C"]


def _sample_listing(item_id: str, image_id: str | None, english: bool = True) -> dict:
    item_name = (
        [{"language_tag": "en_US", "value": f"Title for {item_id}"}]
        if english
        else [{"language_tag": "fr_FR", "value": f"Titre pour {item_id}"}]
    )
    listing = {"item_id": item_id, "item_name": item_name}
    if image_id is not None:
        listing["main_image_id"] = image_id
    return listing


def test_select_subset_filters_and_samples_deterministically(tmp_path):
    listings = [
        _sample_listing("A", "img-a"),
        _sample_listing("B", "img-b"),
        _sample_listing("C", None),  # no image -> excluded
        _sample_listing("D", "img-d-missing"),  # image not in index -> excluded
        _sample_listing("E", "img-e", english=False),  # no English title -> excluded
        _sample_listing("F", "img-f"),
    ]
    shard = tmp_path / "listings_0.json.gz"
    _write_listing_shard(shard, listings)

    images_csv = tmp_path / "images.csv.gz"
    with gzip.open(images_csv, "wt") as f:
        f.write("image_id,height,width,path\n")
        for image_id in ["img-a", "img-b", "img-f"]:
            f.write(f"{image_id},10,10,{image_id[:2]}/{image_id}.jpg\n")

    subset = select_subset([shard], images_csv, n_products=2, seed=0)

    assert len(subset) == 2
    assert all(isinstance(p, Product) for p in subset)
    valid_ids = {"A", "B", "F"}
    assert {p.item_id for p in subset} <= valid_ids


def test_select_subset_dedupes_repeated_item_id_across_marketplaces(tmp_path):
    # Same item_id (ASIN) listed twice, as if sold in two marketplaces -
    # must not be counted as two independent candidates.
    listings = [
        _sample_listing("A", "img-a"),
        _sample_listing("A", "img-a"),
        _sample_listing("B", "img-b"),
    ]
    shard = tmp_path / "listings_0.json.gz"
    _write_listing_shard(shard, listings)

    images_csv = tmp_path / "images.csv.gz"
    with gzip.open(images_csv, "wt") as f:
        f.write("image_id,height,width,path\n")
        for image_id in ["img-a", "img-b"]:
            f.write(f"{image_id},10,10,{image_id[:2]}/{image_id}.jpg\n")

    with pytest.raises(ValueError):
        # only 2 unique item_ids qualify despite 3 listing entries
        select_subset([shard], images_csv, n_products=3, seed=0)

    subset = select_subset([shard], images_csv, n_products=2, seed=0)
    assert sorted(p.item_id for p in subset) == ["A", "B"]


def test_select_subset_is_deterministic_across_calls(tmp_path):
    listings = [_sample_listing(chr(ord("A") + i), f"img-{i}") for i in range(10)]
    shard = tmp_path / "listings_0.json.gz"
    _write_listing_shard(shard, listings)

    images_csv = tmp_path / "images.csv.gz"
    with gzip.open(images_csv, "wt") as f:
        f.write("image_id,height,width,path\n")
        for i in range(10):
            f.write(f"img-{i},10,10,x/img-{i}.jpg\n")

    first = select_subset([shard], images_csv, n_products=4, seed=42)
    second = select_subset([shard], images_csv, n_products=4, seed=42)
    assert [p.item_id for p in first] == [p.item_id for p in second]


def test_select_subset_rejects_oversized_request(tmp_path):
    listings = [_sample_listing("A", "img-a")]
    shard = tmp_path / "listings_0.json.gz"
    _write_listing_shard(shard, listings)

    images_csv = tmp_path / "images.csv.gz"
    with gzip.open(images_csv, "wt") as f:
        f.write("image_id,height,width,path\nimg-a,10,10,i/img-a.jpg\n")

    with pytest.raises(ValueError):
        select_subset([shard], images_csv, n_products=5, seed=0)
