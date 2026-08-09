"""Extrapolate the Week 3 15,000-item measurements (storage + latency) to
the full 147,702-item ABO catalog (docs/01_research_background.md) and
identify where brute-force search would stop being viable.

This is explicitly analytical extrapolation, not a new measurement at that
scale, per docs/04_methodology.md's "Cost projection" section and
docs/adr/0002-brute-force-vs-ann-index.md. Two assumptions, stated so they
can be checked rather than taken on faith:

    1. Stage 1 (low-dim brute-force, touches every catalog item) and a plain
       brute-force full-dim search both scale ~linearly with catalog size —
       both are a single dense matrix multiply against the full catalog.
    2. Stage 2 (full-dim rerank of a fixed-size shortlist) is
       catalog-size-independent — it only ever touches `shortlist_size`
       items, never the full catalog.

The query used for timing is a real projected catalog embedding (not a live
OpenCLIP-encoded query) — for a pure linear-algebra latency measurement the
vector's content doesn't affect matmul cost, and this keeps the script
CPU-only and fast to run without loading the PyTorch backbone.

Usage: python scripts/cost_projection.py
Reads: data/cache/image_embeddings.npy, checkpoints/matryoshka_head.pkl
Writes: checkpoints/cost_projection_report.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from flax import nnx

from matryoshka_search.eval.funnel_search import brute_force_search
from matryoshka_search.eval.latency import benchmark
from matryoshka_search.model.matryoshka_head import MatryoshkaProjectionHead
from matryoshka_search.train.loss import MATRYOSHKA_DIMS

CACHE_DIR = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")
FULL_DIM = 512
FULL_CATALOG_SIZE = 147_702  # real ABO catalog size, docs/01_research_background.md
FUNNEL_LOW_DIM = 64
SHORTLIST_SIZE = 100
FINAL_K = 10
LATENCY_BUDGET_SECONDS = 0.1  # illustrative "interactive" latency budget
BYTES_PER_FLOAT32 = 4


def _load_head(name: str) -> nnx.Module:
    head = MatryoshkaProjectionHead(FULL_DIM, FULL_DIM, rngs=nnx.Rngs(0))
    with open(CHECKPOINT_DIR / f"{name}.pkl", "rb") as f:
        state = pickle.load(f)
    nnx.update(head, state)
    return head


def _storage_bytes(n_items: int, dim: int, bytes_per_value: int = BYTES_PER_FLOAT32) -> int:
    return n_items * dim * bytes_per_value


def main() -> None:
    image_embeddings = np.load(CACHE_DIR / "image_embeddings.npy")
    n_measured = image_embeddings.shape[0]
    scale_factor = FULL_CATALOG_SIZE / n_measured
    print(
        f"Measured catalog: {n_measured} items. Extrapolating to "
        f"{FULL_CATALOG_SIZE} items ({scale_factor:.2f}x)."
    )

    head = _load_head("matryoshka_head")
    catalog_proj = np.asarray(head(image_embeddings))
    query = catalog_proj[0]
    shortlist = np.arange(SHORTLIST_SIZE)

    print("\n=== Measured latency at the real 15k scale (30 trials each) ===")
    full_dim_stats = benchmark(
        lambda: brute_force_search(query, catalog_proj, k=FINAL_K), n_trials=30
    )
    stage1_stats = benchmark(
        lambda: brute_force_search(
            query[:FUNNEL_LOW_DIM], catalog_proj[:, :FUNNEL_LOW_DIM], k=SHORTLIST_SIZE
        ),
        n_trials=30,
    )
    stage2_stats = benchmark(
        lambda: brute_force_search(query, catalog_proj[shortlist], k=FINAL_K), n_trials=30
    )
    print(f"Brute-force full-dim (n={n_measured}): {full_dim_stats.mean_seconds * 1000:.2f}ms")
    print(
        f"Funnel Stage 1 (low_dim={FUNNEL_LOW_DIM}, n={n_measured}): "
        f"{stage1_stats.mean_seconds * 1000:.2f}ms"
    )
    print(
        f"Funnel Stage 2 (shortlist={SHORTLIST_SIZE}, catalog-size-independent): "
        f"{stage2_stats.mean_seconds * 1000:.2f}ms"
    )

    print(f"\n=== Extrapolated to {FULL_CATALOG_SIZE} items (analytical, not measured) ===")
    extrapolated_full_dim = full_dim_stats.mean_seconds * scale_factor
    extrapolated_stage1 = stage1_stats.mean_seconds * scale_factor
    extrapolated_funnel = extrapolated_stage1 + stage2_stats.mean_seconds  # Stage 2 doesn't scale
    speedup_measured = full_dim_stats.mean_seconds / (
        stage1_stats.mean_seconds + stage2_stats.mean_seconds
    )
    speedup_extrapolated = extrapolated_full_dim / extrapolated_funnel
    print(f"Brute-force full-dim: {extrapolated_full_dim * 1000:.1f}ms")
    print(f"Funnel search: {extrapolated_funnel * 1000:.1f}ms")
    print(
        f"Speedup at measured scale: {speedup_measured:.2f}x -> "
        f"at full-catalog scale: {speedup_extrapolated:.2f}x "
        f"(essentially flat, not growing further: Stage 2 is already negligible next "
        f"to Stage 1 even at {n_measured} items, so the ratio is already close to its "
        f"asymptotic ceiling of full_dim_stage_cost/stage1_cost — note this is well "
        f"under the naive {FULL_DIM / FUNNEL_LOW_DIM:.0f}x = full_dim/low_dim ratio, "
        f"since fixed per-query overhead (argsort, L2-normalize) doesn't shrink "
        f"proportionally to dimension)"
    )

    full_dim_threshold_items = int(
        n_measured * (LATENCY_BUDGET_SECONDS / full_dim_stats.mean_seconds)
    )
    stage1_threshold_items = int(n_measured * (LATENCY_BUDGET_SECONDS / stage1_stats.mean_seconds))
    print(
        f"\nAt this measured per-item cost, a *plain full-dim* brute-force search would "
        f"cross a {LATENCY_BUDGET_SECONDS * 1000:.0f}ms budget at ~{full_dim_threshold_items:,} "
        f"items. Stage 1 as actually implemented (brute-force at low_dim={FUNNEL_LOW_DIM}, not "
        f"full dim) is far cheaper per item, so *its* threshold is ~{stage1_threshold_items:,} "
        "items — past whichever threshold is relevant, that stage would need an ANN index "
        "(see adr/0002), not brute force."
    )

    print("\n=== Storage projection (float32, one stored full-dim vector per catalog item) ===")
    storage_by_dim: dict[int, dict[str, float]] = {}
    for dim in MATRYOSHKA_DIMS:
        measured_mb = _storage_bytes(n_measured, dim) / 1e6
        full_catalog_mb = _storage_bytes(FULL_CATALOG_SIZE, dim) / 1e6
        storage_by_dim[dim] = {"measured_mb": measured_mb, "full_catalog_mb": full_catalog_mb}
        print(
            f"  dim={dim:>3}: {measured_mb:>9.2f}MB @ {n_measured} items -> "
            f"{full_catalog_mb:>10.2f}MB @ {FULL_CATALOG_SIZE} items"
        )

    dim512_full_mb = storage_by_dim[FULL_DIM]["full_catalog_mb"]
    dim_low_full_mb = storage_by_dim[FUNNEL_LOW_DIM]["full_catalog_mb"]
    two_index_overhead_factor = (dim512_full_mb + dim_low_full_mb) / dim512_full_mb
    print(
        f"\nBecause Matryoshka's low-dim view is a prefix of the same stored vector, "
        f"funnel search needs zero extra storage beyond the {dim512_full_mb:.1f}MB "
        f"full-{FULL_DIM}-dim catalog. A system that instead stored a "
        f"separately-trained {FUNNEL_LOW_DIM}-dim index alongside the full-dim index "
        f"would need {dim512_full_mb + dim_low_full_mb:.1f}MB — "
        f"{two_index_overhead_factor:.2f}x more, for the same two-stage search."
    )

    report = {
        "n_measured": n_measured,
        "full_catalog_size": FULL_CATALOG_SIZE,
        "scale_factor": scale_factor,
        "funnel_low_dim": FUNNEL_LOW_DIM,
        "shortlist_size": SHORTLIST_SIZE,
        "measured_latency_seconds": {
            "brute_force_full_dim": full_dim_stats.mean_seconds,
            "funnel_stage1": stage1_stats.mean_seconds,
            "funnel_stage2": stage2_stats.mean_seconds,
        },
        "extrapolated_latency_seconds_at_full_catalog": {
            "brute_force_full_dim": extrapolated_full_dim,
            "funnel_search": extrapolated_funnel,
        },
        "speedup": {
            "at_measured_scale": speedup_measured,
            "at_full_catalog_scale": speedup_extrapolated,
        },
        "latency_budget_seconds": LATENCY_BUDGET_SECONDS,
        "brute_force_full_dim_viability_threshold_items": full_dim_threshold_items,
        "funnel_stage1_viability_threshold_items": stage1_threshold_items,
        "storage_by_dim": storage_by_dim,
        "non_matryoshka_two_index_storage_overhead_factor": two_index_overhead_factor,
    }
    (CHECKPOINT_DIR / "cost_projection_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved cost_projection_report.json to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
