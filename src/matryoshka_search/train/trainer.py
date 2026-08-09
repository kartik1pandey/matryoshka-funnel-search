"""Optax training loop for the Matryoshka projection head.

STATUS: skeleton — real implementation lands in Week 2 (see docs/06_plan.md).

Deliberately separated from train/loss.py: the loss function is passed in,
so this same loop trains both the Matryoshka model and the single-dimension
baseline described in docs/04_methodology.md#baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 20
    batch_size: int = 256
    weight_decay: float = 1e-4


def train(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    loss_fn: Callable,
    config: TrainConfig,
):
    """Train a projection head on cached frozen embeddings.

    Args:
        image_embeddings: (n, backbone_dim) cached frozen image embeddings.
        text_embeddings: (n, backbone_dim) cached frozen text embeddings, matched by index.
        loss_fn: e.g. train.loss.matryoshka_loss (partially applied with `dims`)
            or a single-dimension InfoNCE loss for the baseline.
        config: optimizer/training hyperparameters.

    Returns:
        Trained parameters and a per-epoch loss history (per docs/06_plan.md
        Week 2's check that loss decreases at every dimension, not only the
        full one).
    """
    raise NotImplementedError("See docs/06_plan.md Week 2.")
