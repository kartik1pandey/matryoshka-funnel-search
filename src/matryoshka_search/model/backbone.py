"""Frozen PyTorch OpenCLIP backbone wrapper + embedding precompute/cache.

STATUS: skeleton — real implementation lands in Week 1 (see docs/06_plan.md).

This is the one module in the codebase allowed to import PyTorch for
anything beyond the initial `pip install`. Keeping the frozen backbone
contained here makes "we never fine-tune this" a checkable property: outside
this file, nothing else should import torch or touch `requires_grad`.

See docs/02_architecture.md#component-1--backbone-pytorch-frozen and
docs/adr/0001-pytorch-backbone-jax-head.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OpenClipBackbone:
    """Frozen OpenCLIP ViT-B-32 (openai weights) image/text encoder.

    Always runs under `torch.no_grad()` in `eval()` mode. Returns plain
    NumPy arrays — this class is the PyTorch->NumPy boundary referenced in
    docs/adr/0001-pytorch-backbone-jax-head.md.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        raise NotImplementedError(
            "Load open_clip_torch's ViT-B-32 (openai) model + preprocess transform here; "
            "see docs/06_plan.md Week 1."
        )

    def encode_images(self, image_paths: list[Path]) -> np.ndarray:
        """Encode a batch of images to (n, 512) frozen embeddings."""
        raise NotImplementedError

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of text strings to (n, 512) frozen embeddings."""
        raise NotImplementedError


def precompute_and_cache_embeddings(
    backbone: OpenClipBackbone,
    image_paths: list[Path],
    texts: list[str],
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen backbone once over the dataset and cache the outputs.

    The backbone's outputs never change (it's frozen), so this is meant to
    run exactly once per dataset subset — every later stage (training, eval)
    should read from `cache_dir`, not re-invoke PyTorch. See
    docs/02_architecture.md's note on why this precompute step is what makes
    mixing PyTorch and JAX cheap.
    """
    raise NotImplementedError("See docs/06_plan.md Week 1.")
