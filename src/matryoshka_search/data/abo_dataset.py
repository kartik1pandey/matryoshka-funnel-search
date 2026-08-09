"""Amazon Berkeley Objects (ABO) subset download and deterministic selection.

Source: https://registry.opendata.aws/amazon-berkeley-objects/ (public S3
bucket, plain HTTPS, no credentials needed). License: the bucket's own
`README.md` states CC BY 4.0 as the current license (confirmed by fetching
it directly) — a leftover `LICENSE-CC-BY-NC-4.0.txt` file from 2021 is
superseded by the CC-BY-4.0 license file and README added in 2023.

Design decision: rather than downloading the full `abo-images-small.tar`
(3.25GB, all 398k images), this module downloads the cheap metadata first
(16 listing shards, ~87MB total; the image_id -> path index, ~6.4MB), filters
to English-titled listings with a resolvable image, deterministically samples
the requested subset size, and downloads *only those* images individually.
For a 10k-20k product subset this is on the order of a few hundred MB instead
of 3.25GB.

English-only filtering: ABO's `item_name` field is multilingual per listing
(one language per Amazon marketplace). OpenCLIP's text tower is trained
predominantly on English text, so a non-English title would produce a
degraded text embedding independent of anything this project's Matryoshka
training does — filtering to English titles isolates the variable this
project is actually about.
"""

from __future__ import annotations

import concurrent.futures
import csv
import gzip
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests
import requests.adapters
from tqdm import tqdm

BUCKET_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"
N_LISTING_SHARDS = 16
IMAGES_CSV_URL = f"{BUCKET_URL}/images/metadata/images.csv.gz"
IMAGE_BASE_URL = f"{BUCKET_URL}/images/small"

# Preference order when a listing has more than one English-tagged title
# (e.g. both en_US and en_GB) — first match wins.
_ENGLISH_TAG_PREFERENCE = ("en_US", "en_GB", "en_CA", "en_AU", "en_IN")


@dataclass(frozen=True)
class Product:
    item_id: str
    title: str
    image_id: str
    image_url: str
    image_path: Path | None = None  # set once download_product_images runs


def _download_file(
    url: str,
    dest: Path,
    session: requests.Session | None = None,
    chunk_size: int = 1 << 16,
) -> Path:
    """Stream `url` to `dest`. Skips the request entirely if `dest` already
    exists — this is what makes re-running this module after a partial
    download resumable at the file level (not byte-range resumable, but
    idempotent, which is what actually matters for a one-time dataset pull)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    http = session or requests
    with http.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(tmp_dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    tmp_dest.rename(dest)
    return dest


def download_listings_metadata(dest_dir: Path) -> list[Path]:
    """Download all 16 listing metadata shards (~87MB total) to dest_dir/listings/.

    Shards are named by hex digit (listings_0..9, listings_a..f), not decimal
    0..15 — confirmed by listing the bucket directly rather than guessing.
    """
    listings_dir = dest_dir / "listings"
    paths = []
    for i in range(N_LISTING_SHARDS):
        shard = format(i, "x")
        url = f"{BUCKET_URL}/listings/metadata/listings_{shard}.json.gz"
        paths.append(_download_file(url, listings_dir / f"listings_{shard}.json.gz"))
    return paths


def download_images_index(dest_dir: Path) -> Path:
    """Download the image_id -> relative-path index (~6.4MB) to dest_dir/."""
    return _download_file(IMAGES_CSV_URL, dest_dir / "images.csv.gz")


def _iter_listings(listing_paths: list[Path]) -> Iterator[dict]:
    """Yield each listing (one JSON object per line) across all shard files."""
    for path in listing_paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _english_title(listing: dict) -> str | None:
    """Return the best English `item_name` value, or None if the listing
    has no English-tagged title."""
    names = listing.get("item_name", [])
    by_tag = {entry["language_tag"]: entry["value"] for entry in names if "language_tag" in entry}
    for tag in _ENGLISH_TAG_PREFERENCE:
        if tag in by_tag:
            return by_tag[tag]
    for tag, value in by_tag.items():
        if tag.startswith("en_") or tag == "en":
            return value
    return None


def _load_image_index(images_csv_path: Path) -> dict[str, str]:
    """Parse images.csv.gz into an image_id -> relative-path dict."""
    index = {}
    with gzip.open(images_csv_path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            index[row["image_id"]] = row["path"]
    return index


def select_subset(
    listing_paths: list[Path],
    images_csv_path: Path,
    n_products: int,
    seed: int = 0,
) -> list[Product]:
    """Deterministically select `n_products` English-titled, image-having
    listings from the full metadata.

    A fixed seed matters here: evaluation numbers are only comparable across
    code changes if the underlying data sample didn't silently change too.
    Sampling happens over the *entire* qualifying population (not just the
    first N listings encountered), so the subset isn't biased toward
    whatever happens to sort first in the raw files.

    The same `item_id` (ASIN) can appear as a separate listing entry per
    Amazon marketplace (e.g. sold in both the US and UK catalogs) — each
    such duplicate would otherwise be sampled as if it were an independent
    product while actually downloading to the exact same image filename.
    Deduplicated by `item_id` (first occurrence kept) before sampling, so
    every product in the result is unique and the returned count always
    matches the number of distinct images actually downloaded.
    """
    image_index = _load_image_index(images_csv_path)

    candidates_by_item_id: dict[str, Product] = {}
    for listing in _iter_listings(listing_paths):
        item_id = listing["item_id"]
        if item_id in candidates_by_item_id:
            continue
        image_id = listing.get("main_image_id")
        if not image_id or image_id not in image_index:
            continue
        title = _english_title(listing)
        if not title:
            continue
        candidates_by_item_id[item_id] = Product(
            item_id=item_id,
            title=title,
            image_id=image_id,
            image_url=f"{IMAGE_BASE_URL}/{image_index[image_id]}",
        )

    candidates = list(candidates_by_item_id.values())
    if n_products > len(candidates):
        raise ValueError(
            f"requested {n_products} products but only {len(candidates)} "
            "English-titled, image-having listings are available"
        )

    rng = random.Random(seed)
    return rng.sample(candidates, n_products)


def download_product_images(
    products: list[Product], dest_dir: Path, max_workers: int = 32
) -> list[Product]:
    """Download each product's main image to dest_dir/<item_id>.jpg, concurrently.

    Downloading thousands of small images one at a time is latency-bound,
    not bandwidth-bound (each file is only a few KB, but a sequential HTTPS
    round-trip is ~100-300ms) — at 15k images that's 25-75 minutes serial vs.
    roughly a minute or two with `max_workers` connections in flight. A single
    shared `requests.Session` reuses TCP/TLS connections across requests
    instead of renegotiating one per image.

    Returns new Product records with `image_path` filled in. Already-cached
    images are skipped (see `_download_file`), so re-running this after an
    interrupted download only fetches what's still missing.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(product: Product, session: requests.Session) -> Product:
        image_path = dest_dir / f"{product.item_id}.jpg"
        _download_file(product.image_url, image_path, session=session)
        return Product(
            item_id=product.item_id,
            title=product.title,
            image_id=product.image_id,
            image_url=product.image_url,
            image_path=image_path,
        )

    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers, pool_maxsize=max_workers
        )
        session.mount("https://", adapter)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fetch, product, session) for product in products]
            updated = [
                future.result()
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Downloading product images",
                )
            ]

    by_item_id = {p.item_id: p for p in updated}
    return [by_item_id[p.item_id] for p in products]


def build_subset(dest_dir: Path, n_products: int = 15_000, seed: int = 0) -> list[Product]:
    """End-to-end: download metadata + image index, select the subset,
    download its images, and write a manifest.json for reproducibility."""
    listing_paths = download_listings_metadata(dest_dir)
    images_csv_path = download_images_index(dest_dir)
    products = select_subset(listing_paths, images_csv_path, n_products, seed)
    products = download_product_images(products, dest_dir / "images")

    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "n_products": n_products,
                "seed": seed,
                "products": [
                    {
                        "item_id": p.item_id,
                        "title": p.title,
                        "image_path": str(p.image_path),
                    }
                    for p in products
                ],
            },
            indent=2,
        )
    )
    return products
