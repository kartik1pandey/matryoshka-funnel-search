"""Trainable JAX/Flax Matryoshka projection head.

The nested-dimension structure (M = {8, 16, ..., 512}) is a property of how
this module's output is sliced and loss-computed (train/loss.py), not of
the architecture itself: the head always outputs the full-dimension vector.
See docs/04_methodology.md.
"""

from __future__ import annotations

from flax import nnx
from jax import Array


class MatryoshkaProjectionHead(nnx.Module):
    """Projects a frozen backbone embedding to a new space trained for
    graceful truncation at every dimension in M.

    A single linear layer (input_dim -> output_dim), per
    docs/02_architecture.md's "small MLP (or even a single linear layer to
    start)" — start with the simplest thing that can be trained and
    evaluated end-to-end; a small MLP is a drop-in swap here later if the
    evaluation in Week 3 shows the linear head is the bottleneck, not
    something to reach for pre-emptively.

    Applied identically (same shared weights) to both the image-tower and
    text-tower frozen embeddings — the caller is responsible for passing
    each modality's embeddings through this same module instance, which is
    what makes the projection genuinely cross-modal rather than two
    independently-trained ones.
    """

    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.linear(x)
