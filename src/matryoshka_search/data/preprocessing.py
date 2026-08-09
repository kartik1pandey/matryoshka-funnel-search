"""Image/text preprocessing for CLIP input.

STATUS: skeleton — real implementation lands in Week 1 (see docs/06_plan.md).

Deliberately delegates to open_clip's own image_transform rather than
reimplementing CLIP's resize/crop/normalize pipeline — subtle preprocessing
mismatches are a classic, hard-to-detect source of silently degraded
embeddings.
"""

from __future__ import annotations

from pathlib import Path


def load_and_preprocess_image(path: Path):
    """Load an image and apply OpenCLIP's expected preprocessing transform."""
    raise NotImplementedError("See docs/06_plan.md Week 1.")


def normalize_text(title: str, description: str | None = None, max_tokens: int = 77) -> str:
    """Combine title (+ optional short description) into CLIP's text input,
    truncated to CLIP's token limit."""
    raise NotImplementedError("See docs/06_plan.md Week 1.")
