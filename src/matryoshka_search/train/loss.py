"""InfoNCE contrastive loss and the Matryoshka nested-loss composition.

This is the single most interview-relevant piece of code in this repo — the
concrete implementation of the Matryoshka Representation Learning mechanism
(Kusupati et al., NeurIPS 2022), not a black-box library call. See
docs/04_methodology.md for the derivation and docs/01_research_background.md
for the paper citation.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

# Canonical nested truncation lengths, per docs/02_architecture.md and
# docs/04_methodology.md. Exported so training/eval code has one shared
# source of truth instead of each caller retyping the same tuple.
MATRYOSHKA_DIMS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)


def info_nce_loss(image_emb: Array, text_emb: Array, temperature: float = 0.07) -> Array:
    """Symmetric CLIP-style contrastive loss over a batch of matched pairs.

    Row i of `image_emb` and row i of `text_emb` are assumed to be the true
    match; every other row in the same batch is an in-batch negative.

    Args:
        image_emb: (batch, dim) image embeddings (need not be pre-normalized).
        text_emb: (batch, dim) text embeddings, same batch order as image_emb.
        temperature: softmax temperature; lower = sharper distribution.

    Returns:
        Scalar loss (mean of the image->text and text->image cross-entropy terms).
    """
    image_norm = image_emb / jnp.linalg.norm(image_emb, axis=-1, keepdims=True)
    text_norm = text_emb / jnp.linalg.norm(text_emb, axis=-1, keepdims=True)

    logits = (image_norm @ text_norm.T) / temperature  # (batch, batch)
    labels = jnp.arange(logits.shape[0])

    image_to_text = -jnp.take_along_axis(
        jnp.log(jax_softmax(logits)), labels[:, None], axis=1
    ).mean()
    text_to_image = -jnp.take_along_axis(
        jnp.log(jax_softmax(logits.T)), labels[:, None], axis=1
    ).mean()

    return 0.5 * (image_to_text + text_to_image)


def jax_softmax(logits: Array) -> Array:
    """Numerically stable softmax along the last axis."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp, axis=-1, keepdims=True)


def matryoshka_loss(
    image_emb: Array,
    text_emb: Array,
    dims: Sequence[int],
    temperature: float = 0.07,
) -> Array:
    """Sum of independent InfoNCE losses over nested truncations of the embedding.

    This is the actual Matryoshka mechanism: every dimension in `dims` gets
    its own contrastive loss term, computed on the truncated prefix of the
    embedding, and all terms share gradients back into the same underlying
    projection head. That per-dimension loss is what forces early dimensions
    to be independently useful rather than only meaningful as part of the
    full-length vector.

    Args:
        image_emb: (batch, full_dim) image embeddings, full-length output of the projection head.
        text_emb: (batch, full_dim) text embeddings, same shape.
        dims: nested truncation lengths to train, e.g. (8, 16, 32, 64, 128, 256, 512).
        temperature: shared temperature for every per-dimension InfoNCE term.

    Returns:
        Scalar total loss (sum, not mean, across dims — see docs/04_methodology.md
        for why summation rather than averaging is the specified mechanism).
    """
    full_dim = image_emb.shape[-1]
    for d in dims:
        if d <= 0 or d > full_dim:
            raise ValueError(f"invalid Matryoshka dimension {d} for embeddings of size {full_dim}")

    per_dim_losses = jnp.stack(
        [info_nce_loss(image_emb[..., :d], text_emb[..., :d], temperature) for d in dims]
    )
    return jnp.sum(per_dim_losses)
