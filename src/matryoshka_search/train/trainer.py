"""Optax training loop for the Matryoshka projection head.

Deliberately separated from train/loss.py: the loss function is passed in,
so this same loop trains both the Matryoshka model and the single-dimension
baseline described in docs/04_methodology.md#baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3
    num_epochs: int = 20
    batch_size: int = 256
    weight_decay: float = 1e-4


def train(
    head: nnx.Module,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    loss_fn: Callable,
    config: TrainConfig,
    seed: int = 0,
) -> list[float]:
    """Train `head` in place on cached frozen embeddings.

    Args:
        head: an nnx.Module (e.g. MatryoshkaProjectionHead) — mutated in
            place by training; there is no separate "trained parameters"
            return value because nnx modules carry their own state.
        image_embeddings: (n, backbone_dim) cached frozen image embeddings.
        text_embeddings: (n, backbone_dim) cached frozen text embeddings,
            matched by index with image_embeddings.
        loss_fn: (image_proj, text_proj) -> scalar loss — e.g. a partial
            application of train.loss.matryoshka_loss with `dims` bound, or
            a single-dimension InfoNCE loss for the non-Matryoshka baseline
            (docs/04_methodology.md#baseline). Passing a different loss_fn
            through the same loop is what keeps the baseline comparison
            honest: same data, same optimizer, same schedule, only the loss
            shape differs.
        config: optimizer/training hyperparameters.
        seed: batch-shuffling seed, for reproducible epoch order across runs.

    Returns:
        Per-epoch mean loss history — used to confirm loss actually
        decreases (docs/06_plan.md Week 2's check), and at every dimension
        when `loss_fn` is the Matryoshka nested loss, not just the full one.
    """
    n = image_embeddings.shape[0]
    if n < config.batch_size:
        raise ValueError(f"batch_size ({config.batch_size}) exceeds dataset size ({n})")

    num_steps_per_epoch = n // config.batch_size
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=max(1, config.num_epochs * num_steps_per_epoch),
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    optimizer = nnx.Optimizer(head, tx, wrt=nnx.Param)

    image_embeddings_jnp = jnp.asarray(image_embeddings)
    text_embeddings_jnp = jnp.asarray(text_embeddings)

    def step_loss_fn(model: nnx.Module, image_batch: jnp.ndarray, text_batch: jnp.ndarray):
        return loss_fn(model(image_batch), model(text_batch))

    grad_fn = nnx.value_and_grad(step_loss_fn)

    rng = np.random.default_rng(seed)
    loss_history = []
    for _ in range(config.num_epochs):
        perm = rng.permutation(n)
        epoch_losses = []
        # Drops a partial trailing batch each epoch rather than padding it —
        # simplest correct behavior; at 15k examples and batch_size=256 this
        # drops at most 255 examples per epoch, immaterial to training.
        for start in range(0, num_steps_per_epoch * config.batch_size, config.batch_size):
            batch_idx = perm[start : start + config.batch_size]
            image_batch = image_embeddings_jnp[batch_idx]
            text_batch = text_embeddings_jnp[batch_idx]
            loss, grads = grad_fn(head, image_batch, text_batch)
            optimizer.update(head, grads)
            epoch_losses.append(float(loss))
        loss_history.append(float(np.mean(epoch_losses)))

    return loss_history
