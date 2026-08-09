"""Interactive query CLI: type a text query, see ranked product results and
real per-stage timing, using the real trained Matryoshka (or baseline) head,
the real cached catalog embeddings, and the real funnel-search pattern from
eval/funnel_search.py — the same code path measured in Week 3
(scripts/evaluate.py), applied one query at a time instead of in bulk.

Usage: run from the repo root with the project's dev venv active:
    python -m matryoshka_search.demo.cli
    python -m matryoshka_search.demo.cli --query "wireless bluetooth headphones"
    python -m matryoshka_search.demo.cli --model baseline --low-dim 16

Reads:
    data/raw/manifest.json (item_id -> title, image_path)
    data/cache/{image_embeddings.npy, item_ids.json}
    checkpoints/{matryoshka,baseline}_head.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from flax import nnx

from matryoshka_search.eval.funnel_search import funnel_search
from matryoshka_search.model.backbone import OpenClipBackbone
from matryoshka_search.model.matryoshka_head import MatryoshkaProjectionHead

MANIFEST_PATH = Path("data/raw/manifest.json")
CACHE_DIR = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")
FULL_DIM = 512
DEFAULT_LOW_DIM = 64
DEFAULT_SHORTLIST_SIZE = 100
DEFAULT_K = 10


def load_catalog(
    manifest_path: Path, cache_dir: Path
) -> tuple[list[str], dict[str, str], dict[str, str], np.ndarray]:
    """Load item_ids (row order matching the cached embeddings) plus
    item_id -> title / image_path lookups from the manifest. Looked up by
    item_id rather than assumed positional, since these are two separately
    loaded files whose row-alignment is a convention, not an invariant."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    titles = {p["item_id"]: p["title"] for p in manifest["products"]}
    image_paths = {p["item_id"]: p["image_path"] for p in manifest["products"]}
    item_ids = json.loads((cache_dir / "item_ids.json").read_text(encoding="utf-8"))
    image_embeddings = np.load(cache_dir / "image_embeddings.npy")
    return item_ids, titles, image_paths, image_embeddings


def load_head(name: str) -> nnx.Module:
    head = MatryoshkaProjectionHead(FULL_DIM, FULL_DIM, rngs=nnx.Rngs(0))
    with open(CHECKPOINT_DIR / f"{name}.pkl", "rb") as f:
        state = pickle.load(f)
    nnx.update(head, state)
    return head


def project_catalog(head: nnx.Module, image_embeddings: np.ndarray) -> np.ndarray:
    return np.asarray(head(image_embeddings))


def run_search(
    query_text: str,
    encode_text: Callable[[str], np.ndarray],
    project: Callable[[np.ndarray], np.ndarray],
    catalog_proj: np.ndarray,
    low_dim: int,
    shortlist_size: int,
    k: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Encode -> project -> funnel search, timing each stage separately.

    `encode_text`/`project` are injected as plain callables (rather than
    concrete backbone/head objects) so this function is testable without
    loading torch or a real checkpoint.
    """
    t0 = time.perf_counter()
    query_emb = encode_text(query_text)
    t1 = time.perf_counter()
    query_proj = project(query_emb)
    t2 = time.perf_counter()
    indices = funnel_search(query_proj, catalog_proj, low_dim, shortlist_size, k)
    t3 = time.perf_counter()

    timings = {
        "encode_seconds": t1 - t0,
        "project_seconds": t2 - t1,
        "search_seconds": t3 - t2,
        "total_seconds": t3 - t0,
    }
    return indices, timings


def format_results(
    indices: np.ndarray,
    item_ids: Sequence[str],
    titles: dict[str, str],
    image_paths: dict[str, str],
) -> list[str]:
    lines = []
    for rank, idx in enumerate(indices, start=1):
        item_id = item_ids[idx]
        title = titles.get(item_id, "<unknown title>")
        path = image_paths.get(item_id, "<unknown path>")
        lines.append(f"{rank:>2}. [{item_id}] {title}\n      image: {path}")
    return lines


def _format_timings(timings: dict[str, float]) -> str:
    return (
        f"encode={timings['encode_seconds'] * 1000:.1f}ms  "
        f"project={timings['project_seconds'] * 1000:.1f}ms  "
        f"search={timings['search_seconds'] * 1000:.1f}ms  "
        f"total={timings['total_seconds'] * 1000:.1f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["matryoshka", "baseline"], default="matryoshka")
    parser.add_argument("--low-dim", type=int, default=DEFAULT_LOW_DIM)
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORTLIST_SIZE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--query", default=None, help="Run a single query non-interactively and exit."
    )
    args = parser.parse_args()

    print(f"Loading catalog from {MANIFEST_PATH} / {CACHE_DIR} ...")
    item_ids, titles, image_paths, image_embeddings = load_catalog(MANIFEST_PATH, CACHE_DIR)
    print(f"Loading {args.model} head checkpoint ...")
    head = load_head(f"{args.model}_head")
    catalog_proj = project_catalog(head, image_embeddings)
    print(f"Catalog: {len(item_ids)} products, projected dim={catalog_proj.shape[1]}.")
    print("Loading frozen OpenCLIP backbone (one-time, real PyTorch model load) ...")
    backbone = OpenClipBackbone()

    def encode_text(text: str) -> np.ndarray:
        return backbone.encode_texts([text])[0]

    def project(x: np.ndarray) -> np.ndarray:
        return np.asarray(head(x[None, :]))[0]

    def run_and_print(query: str) -> None:
        indices, timings = run_search(
            query, encode_text, project, catalog_proj, args.low_dim, args.shortlist_size, args.k
        )
        print(
            f"\nTop {len(indices)} results for {query!r} "
            f"(model={args.model}, low_dim={args.low_dim}):"
        )
        for line in format_results(indices, item_ids, titles, image_paths):
            print(line)
        print(f"\n[timing] {_format_timings(timings)}")

    if args.query is not None:
        run_and_print(args.query)
        return

    print("\nType a query and press enter. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        run_and_print(query)


if __name__ == "__main__":
    main()
