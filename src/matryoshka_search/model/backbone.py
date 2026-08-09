"""Frozen PyTorch OpenCLIP backbone wrapper + embedding precompute/cache.

This is the one module in the codebase allowed to import PyTorch for
anything beyond the initial `pip install`. Keeping the frozen backbone
contained here makes "we never fine-tune this" a checkable property: outside
this file, nothing else should import torch or touch `requires_grad`.

See docs/02_architecture.md#component-1--backbone-pytorch-frozen and
docs/adr/0001-pytorch-backbone-jax-head.md.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import open_clip
import torch
from PIL import Image


class OpenClipBackbone:
    """Frozen OpenCLIP ViT-B-32 (openai weights) image/text encoder.

    Always runs under `torch.no_grad()` in `eval()` mode, with every
    parameter's `requires_grad` explicitly disabled — belt-and-suspenders
    against ever accidentally fine-tuning this. Returns plain NumPy arrays;
    this class is the PyTorch->NumPy boundary referenced in
    docs/adr/0001-pytorch-backbone-jax-head.md.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        # OpenAI's original ViT-B-32 checkpoint was trained with the QuickGELU
        # activation. Plain "ViT-B-32" in open_clip defaults to quick_gelu=False,
        # which silently mismatches the pretrained weights' activation function
        # (open_clip warns about this at load time) and degrades embedding
        # quality — "ViT-B-32-quickgelu" is the variant that matches OpenAI's
        # original training config. Confirmed via `open_clip.list_pretrained()`.
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval().to(self.device)
        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path], batch_size: int = 64) -> np.ndarray:
        """Encode a list of image files to (n, embed_dim) frozen embeddings.

        Unnormalized (no L2 normalize here) — the contrastive loss in
        train/loss.py normalizes internally, and keeping the cached
        embeddings unnormalized preserves the backbone's raw output in case
        a later stage needs it (e.g. inspecting embedding norms).
        """
        all_features = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = torch.stack(
                [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(self.device)
            features = self.model.encode_image(images)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """Encode a list of text strings to (n, embed_dim) frozen embeddings."""
        all_features = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = self.tokenizer(list(batch_texts)).to(self.device)
            features = self.model.encode_text(tokens)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)


class Encoder(Protocol):
    """The subset of OpenClipBackbone's interface precompute_and_cache_embeddings
    actually needs — lets tests inject a fake encoder without importing torch."""

    def encode_images(self, image_paths: Sequence[Path], batch_size: int = ...) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str], batch_size: int = ...) -> np.ndarray: ...


def precompute_and_cache_embeddings(
    backbone: Encoder,
    image_paths: Sequence[Path],
    texts: Sequence[str],
    cache_dir: Path,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the frozen backbone once over the dataset and cache the outputs.

    The backbone's outputs never change (it's frozen), so this is meant to
    run exactly once per dataset subset — every later stage (training, eval)
    should read from `cache_dir`, not re-invoke PyTorch. See
    docs/02_architecture.md's note on why this precompute step is what makes
    mixing PyTorch and JAX cheap. Idempotent: if both cache files already
    exist, the backbone is never invoked at all.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_cache_path = cache_dir / "image_embeddings.npy"
    text_cache_path = cache_dir / "text_embeddings.npy"

    if image_cache_path.exists() and text_cache_path.exists():
        return np.load(image_cache_path), np.load(text_cache_path)

    image_embeddings = backbone.encode_images(image_paths, batch_size=batch_size)
    text_embeddings = backbone.encode_texts(texts, batch_size=batch_size)

    np.save(image_cache_path, image_embeddings)
    np.save(text_cache_path, text_embeddings)
    return image_embeddings, text_embeddings


def precompute_from_manifest(
    manifest_path: Path,
    cache_dir: Path,
    backbone: Encoder | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a data/abo_dataset.py manifest.json and precompute+cache embeddings
    for every product in it, in manifest order.

    Returns (image_embeddings, text_embeddings, item_ids) — item_ids preserves
    the row alignment so later stages can map an embedding row back to a
    product. `backbone` defaults to a real OpenClipBackbone if not given
    (injectable for testing without loading the actual model).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    products = manifest["products"]
    item_ids = [p["item_id"] for p in products]
    image_paths = [Path(p["image_path"]) for p in products]
    texts = [p["title"] for p in products]

    resolved_backbone: Encoder = backbone if backbone is not None else OpenClipBackbone()
    image_embeddings, text_embeddings = precompute_and_cache_embeddings(
        resolved_backbone, image_paths, texts, cache_dir, batch_size=batch_size
    )

    item_ids_cache_path = cache_dir / "item_ids.json"
    if not item_ids_cache_path.exists():
        item_ids_cache_path.write_text(json.dumps(item_ids), encoding="utf-8")

    return image_embeddings, text_embeddings, item_ids
