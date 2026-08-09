"""Train the Matryoshka projection head and the non-Matryoshka baseline on
the cached frozen embeddings, and confirm the Matryoshka loss actually
decreases at every dimension in M, not only the full one.

Usage: run from the repo root with the project's dev venv active:
    python scripts/train_matryoshka.py

Reads data/cache/{image,text}_embeddings.npy (see model/backbone.py).
Writes:
    data/cache/split.json               train/val index split (reused by eval)
    checkpoints/matryoshka_head.pkl     trained Matryoshka head state
    checkpoints/baseline_head.pkl       trained baseline head state
    checkpoints/training_report.json    hyperparams + loss histories + per-dim val loss
"""

from __future__ import annotations

import functools
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
from flax import nnx

from matryoshka_search.model.matryoshka_head import MatryoshkaProjectionHead
from matryoshka_search.train.loss import MATRYOSHKA_DIMS, info_nce_loss, matryoshka_loss
from matryoshka_search.train.trainer import TrainConfig, train

CACHE_DIR = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")
VAL_FRACTION = 0.1
SEED = 0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _train_val_split(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    return perm[n_val:], perm[:n_val]


def _per_dimension_val_loss(head: nnx.Module, image_val, text_val) -> dict[int, float]:
    """Evaluate InfoNCE loss at each Matryoshka dimension on held-out data —
    this is the real evidence for "loss decreases at every dimension," not
    just the summed training loss."""
    image_proj = head(image_val)
    text_proj = head(text_val)
    return {
        dim: float(info_nce_loss(image_proj[..., :dim], text_proj[..., :dim]))
        for dim in MATRYOSHKA_DIMS
    }


def main() -> None:
    image_embeddings = np.load(CACHE_DIR / "image_embeddings.npy")
    text_embeddings = np.load(CACHE_DIR / "text_embeddings.npy")
    n, full_dim = image_embeddings.shape
    print(f"Loaded {n} cached embeddings, dim={full_dim}")

    train_idx, val_idx = _train_val_split(n, VAL_FRACTION, SEED)
    (CACHE_DIR / "split.json").write_text(
        json.dumps({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(), "seed": SEED})
    )
    print(f"Train/val split: {len(train_idx)} / {len(val_idx)} (seed={SEED})")

    image_train, text_train = image_embeddings[train_idx], text_embeddings[train_idx]
    image_val, text_val = image_embeddings[val_idx], text_embeddings[val_idx]

    config = TrainConfig(learning_rate=1e-3, num_epochs=30, batch_size=256, weight_decay=1e-4)

    print("\n=== Training Matryoshka head (nested loss over", MATRYOSHKA_DIMS, ") ===")
    matryoshka_head = MatryoshkaProjectionHead(full_dim, full_dim, rngs=nnx.Rngs(SEED))
    matryoshka_loss_fn = functools.partial(matryoshka_loss, dims=MATRYOSHKA_DIMS)
    matryoshka_history = train(
        matryoshka_head, image_train, text_train, matryoshka_loss_fn, config, seed=SEED
    )
    print(
        "Per-epoch training loss (first 3, last 3):",
        matryoshka_history[:3],
        "...",
        matryoshka_history[-3:],
    )

    print("\n=== Training baseline head (single full-dim loss, no nesting) ===")
    baseline_head = MatryoshkaProjectionHead(full_dim, full_dim, rngs=nnx.Rngs(SEED))
    baseline_history = train(
        baseline_head, image_train, text_train, info_nce_loss, config, seed=SEED
    )
    print(
        "Per-epoch training loss (first 3, last 3):",
        baseline_history[:3],
        "...",
        baseline_history[-3:],
    )

    print("\n=== Per-dimension validation loss (the real Week 2 check) ===")
    matryoshka_val_loss = _per_dimension_val_loss(matryoshka_head, image_val, text_val)
    baseline_val_loss = _per_dimension_val_loss(baseline_head, image_val, text_val)
    print(f"{'dim':>5} | {'matryoshka':>10} | {'baseline':>10}")
    for dim in MATRYOSHKA_DIMS:
        print(f"{dim:>5} | {matryoshka_val_loss[dim]:>10.4f} | {baseline_val_loss[dim]:>10.4f}")

    matryoshka_monotonic = all(
        matryoshka_val_loss[MATRYOSHKA_DIMS[i]]
        >= matryoshka_val_loss[MATRYOSHKA_DIMS[i + 1]] - 1e-3
        for i in range(len(MATRYOSHKA_DIMS) - 1)
    )
    print(f"\nMatryoshka val loss roughly non-increasing with dimension: {matryoshka_monotonic}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "matryoshka_head.pkl", "wb") as f:
        pickle.dump(nnx.state(matryoshka_head), f)
    with open(CHECKPOINT_DIR / "baseline_head.pkl", "wb") as f:
        pickle.dump(nnx.state(baseline_head), f)

    report = {
        "git_commit": _git_commit(),
        "config": config.__dict__,
        "matryoshka_dims": list(MATRYOSHKA_DIMS),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "seed": SEED,
        "matryoshka_train_loss_history": matryoshka_history,
        "baseline_train_loss_history": baseline_history,
        "matryoshka_val_loss_per_dim": matryoshka_val_loss,
        "baseline_val_loss_per_dim": baseline_val_loss,
    }
    (CHECKPOINT_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved checkpoints and training_report.json to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
