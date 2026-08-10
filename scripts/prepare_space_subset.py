"""Build a small, git-shippable subset of the real data + checkpoints, for
deploying the web demo to a Hugging Face Space (docs/07_deployment.md's
"Optional: a small hosted demo" section). Not part of the core pipeline —
the full 15,000-item catalog stays local-only (data/, checkpoints/ are
git-ignored, same as always); this script only prepares the smaller bundle
that actually gets pushed to the separate Hugging Face Space git repo (see
scripts/deploy_space.py), so the public GitHub repo's size/scope is
unaffected.

The trained checkpoints are catalog-size-independent (a fixed linear
projection head, applied identically to any number of catalog items) — no
retraining needed, just subset the cached embeddings/manifest to a smaller
catalog and copy the matching image files; the checkpoints are copied
as-is.

Usage: python scripts/prepare_space_subset.py [--n 750] [--seed 0]
Reads: data/raw/{manifest.json,images/}, data/cache/*, checkpoints/*.pkl
Writes: space_build/{data/raw,data/cache,checkpoints}/... (git-ignored)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

DATA_RAW = Path("data/raw")
DATA_CACHE = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")
OUT_DIR = Path("space_build")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=750)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((DATA_RAW / "manifest.json").read_text(encoding="utf-8"))
    products_by_id = {p["item_id"]: p for p in manifest["products"]}
    item_ids_full = json.loads((DATA_CACHE / "item_ids.json").read_text(encoding="utf-8"))
    image_embeddings = np.load(DATA_CACHE / "image_embeddings.npy")

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(item_ids_full))
    sample_idx = np.sort(rng.choice(len(item_ids_full), size=n, replace=False))
    sample_item_ids = [item_ids_full[i] for i in sample_idx]

    out_images_dir = OUT_DIR / "data/raw/images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    sample_products = []
    for item_id in sample_item_ids:
        product = products_by_id[item_id]
        src_path = Path(product["image_path"])
        shutil.copy(src_path, out_images_dir / src_path.name)
        # Rewrite to a forward-slash path relative to the repo root the
        # Space actually runs from — the original manifest's path style
        # (e.g. Windows backslashes from this dev machine) won't resolve on
        # the Space's Linux container.
        sample_products.append({**product, "image_path": f"data/raw/images/{src_path.name}"})

    (OUT_DIR / "data/raw/manifest.json").write_text(
        json.dumps(
            {"n_products": len(sample_products), "seed": args.seed, "products": sample_products}
        ),
        encoding="utf-8",
    )

    out_cache_dir = OUT_DIR / "data/cache"
    out_cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_cache_dir / "image_embeddings.npy", image_embeddings[sample_idx])
    (out_cache_dir / "item_ids.json").write_text(json.dumps(sample_item_ids), encoding="utf-8")

    out_checkpoints_dir = OUT_DIR / "checkpoints"
    out_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    for name in ("matryoshka_head.pkl", "baseline_head.pkl"):
        shutil.copy(CHECKPOINT_DIR / name, out_checkpoints_dir / name)

    print(f"Wrote {n}-item subset (seed={args.seed}) to {OUT_DIR}/")


if __name__ == "__main__":
    main()
