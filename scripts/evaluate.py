"""Week 3 evaluation: retrieval quality at every Matryoshka dimension,
retention vs. the literature benchmarks, a real funnel-search demonstration,
and latency measurement — all on the real trained checkpoints and cached
embeddings, not synthetic data.

Usage: run from the repo root with the project's dev venv active:
    python scripts/evaluate.py

Reads:
    data/cache/{image,text}_embeddings.npy, data/cache/split.json
    checkpoints/{matryoshka,baseline}_head.pkl
Writes:
    checkpoints/evaluation_report.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from flax import nnx

from matryoshka_search.eval.funnel_search import brute_force_search, funnel_search
from matryoshka_search.eval.latency import benchmark
from matryoshka_search.eval.metrics import ndcg_at_k, recall_at_k, retention
from matryoshka_search.model.matryoshka_head import MatryoshkaProjectionHead
from matryoshka_search.train.loss import MATRYOSHKA_DIMS

CACHE_DIR = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")
FULL_DIM = 512
K = 10
# Two low_dim choices deliberately: 64 retains most quality for either model
# (see per-dimension retention below), so it alone doesn't show much
# contrast between Matryoshka and baseline funnel search. 16 is where the
# per-dimension retention curves diverge sharply between the two models —
# testing both, rather than assuming which will look better, is the point.
FUNNEL_LOW_DIMS = (16, 64)
FUNNEL_SHORTLIST_SIZE = 100
FUNNEL_FINAL_K = 10
FUNNEL_SAMPLE_SIZE = (
    300  # subsample of val queries for the funnel-search demo/latency (loops per-query)
)


def _load_head(name: str) -> nnx.Module:
    head = MatryoshkaProjectionHead(FULL_DIM, FULL_DIM, rngs=nnx.Rngs(0))
    with open(CHECKPOINT_DIR / f"{name}.pkl", "rb") as f:
        state = pickle.load(f)
    nnx.update(head, state)
    return head


def _quality_at_each_dim(
    query_emb: np.ndarray, catalog_emb: np.ndarray, relevant: np.ndarray
) -> dict[int, dict[str, float]]:
    results = {}
    for dim in MATRYOSHKA_DIMS:
        q, c = query_emb[:, :dim], catalog_emb[:, :dim]
        q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
        c_norm = c / np.linalg.norm(c, axis=1, keepdims=True)
        similarity = q_norm @ c_norm.T
        results[dim] = {
            "recall@10": recall_at_k(similarity, relevant, k=K),
            "ndcg@10": ndcg_at_k(similarity, relevant, k=K),
        }
    return results


def _funnel_vs_brute_force(
    query_emb: np.ndarray,
    catalog_emb: np.ndarray,
    relevant: np.ndarray,
    n_samples: int,
    low_dim: int,
) -> dict[str, float]:
    funnel_hits = 0
    brute_force_hits = 0
    for i in range(n_samples):
        funnel_top_k = funnel_search(
            query_emb[i], catalog_emb, low_dim, FUNNEL_SHORTLIST_SIZE, FUNNEL_FINAL_K
        )
        brute_force_top_k = brute_force_search(query_emb[i], catalog_emb, FUNNEL_FINAL_K)
        funnel_hits += int(relevant[i] in funnel_top_k)
        brute_force_hits += int(relevant[i] in brute_force_top_k)
    return {
        "funnel_recall@10": funnel_hits / n_samples,
        "brute_force_recall@10": brute_force_hits / n_samples,
        "funnel_retention": funnel_hits / brute_force_hits if brute_force_hits else float("nan"),
    }


def _measure_latency(
    query_emb: np.ndarray, catalog_emb: np.ndarray, low_dim: int
) -> dict[str, dict]:
    query = query_emb[0]
    full_dim_stats = benchmark(lambda: brute_force_search(query, catalog_emb, K), n_trials=30)
    funnel_stats = benchmark(
        lambda: funnel_search(query, catalog_emb, low_dim, FUNNEL_SHORTLIST_SIZE, FUNNEL_FINAL_K),
        n_trials=30,
    )
    return {
        "brute_force_full_dim": full_dim_stats.__dict__,
        "funnel_search": funnel_stats.__dict__,
    }


def main() -> None:
    image_embeddings = np.load(CACHE_DIR / "image_embeddings.npy")
    text_embeddings = np.load(CACHE_DIR / "text_embeddings.npy")
    split = json.loads((CACHE_DIR / "split.json").read_text())
    val_idx = np.array(split["val_idx"])
    print(f"Catalog: {len(image_embeddings)} items. Val queries: {len(val_idx)}.")

    report: dict = {"k": K, "funnel_low_dims": FUNNEL_LOW_DIMS, "n_val_queries": len(val_idx)}

    # --- Retrieval quality at every dimension, for raw/matryoshka/baseline ---
    print("\n=== Raw frozen CLIP (no projection head), full dim only ===")
    raw_text_val = text_embeddings[val_idx]
    raw_quality = _quality_at_each_dim(raw_text_val, image_embeddings, val_idx)[FULL_DIM]
    print(raw_quality)
    report["raw_full_dim_quality"] = raw_quality

    for model_name in ["matryoshka", "baseline"]:
        print(f"\n=== {model_name} head: quality at each dimension ===")
        head = _load_head(f"{model_name}_head")
        image_proj = np.asarray(head(image_embeddings))
        text_proj = np.asarray(head(text_embeddings))

        quality = _quality_at_each_dim(text_proj[val_idx], image_proj, val_idx)
        for dim, metrics in quality.items():
            r10, n10 = metrics["recall@10"], metrics["ndcg@10"]
            print(f"  dim={dim:>3}: recall@10={r10:.4f}  ndcg@10={n10:.4f}")

        recall_retention = retention({d: m["recall@10"] for d, m in quality.items()}, FULL_DIM)
        print(f"  Recall@10 retention vs full-dim: {recall_retention}")

        report[f"{model_name}_quality_by_dim"] = quality
        report[f"{model_name}_recall_retention"] = recall_retention

        sample_idx = val_idx[:FUNNEL_SAMPLE_SIZE]
        report[f"{model_name}_funnel_vs_brute_force"] = {}
        for low_dim in FUNNEL_LOW_DIMS:
            print(
                f"\n=== Funnel search demo (low_dim={low_dim}), {model_name} embeddings, "
                f"{FUNNEL_SAMPLE_SIZE} queries ==="
            )
            funnel_result = _funnel_vs_brute_force(
                text_proj[sample_idx], image_proj, sample_idx, FUNNEL_SAMPLE_SIZE, low_dim
            )
            print(funnel_result)
            report[f"{model_name}_funnel_vs_brute_force"][low_dim] = funnel_result

        latency_dim = FUNNEL_LOW_DIMS[-1]
        print(f"\n=== Latency ({model_name}, low_dim={latency_dim}, 30 trials) ===")
        latency = _measure_latency(text_proj[sample_idx], image_proj, latency_dim)
        print(latency)
        report[f"{model_name}_latency"] = latency

    (CHECKPOINT_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved evaluation_report.json to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
