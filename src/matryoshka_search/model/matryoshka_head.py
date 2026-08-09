"""Trainable JAX/Flax Matryoshka projection head.

STATUS: skeleton — real implementation lands in Week 2 (see docs/06_plan.md).

The nested-dimension structure (M = {8, 16, ..., 512}) is a property of how
this module's output is sliced and loss-computed (train/loss.py), not of
the architecture itself: the head always outputs the full-dimension vector.
See docs/04_methodology.md.
"""

from __future__ import annotations

from flax import nnx


class MatryoshkaProjectionHead(nnx.Module):
    """Projects a frozen backbone embedding to a new space trained for
    graceful truncation at every dimension in M.

    Starts as a single linear layer (input_dim -> output_dim); structured so
    a small MLP can be swapped in without changing callers.
    """

    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs) -> None:
        raise NotImplementedError("See docs/06_plan.md Week 2.")

    def __call__(self, x):
        raise NotImplementedError
