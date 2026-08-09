"""Amazon Berkeley Objects (ABO) subset download and deterministic selection.

STATUS: skeleton — real implementation lands in Week 1 (see docs/06_plan.md).

See docs/01_research_background.md for the dataset source and the license
caveat that must be checked before this is used for anything beyond a
personal research/portfolio project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Product:
    item_id: str
    image_path: Path
    title: str


def download_abo_metadata(dest_dir: Path) -> Path:
    """Download the ABO listings metadata archive to `dest_dir` (resumable).

    Source: https://registry.opendata.aws/amazon-berkeley-objects/
    """
    raise NotImplementedError("See docs/06_plan.md Week 1.")


def select_subset(metadata_path: Path, n_products: int, seed: int = 0) -> list[Product]:
    """Deterministically select `n_products` from the full metadata.

    A fixed seed matters here: evaluation numbers are only comparable across
    code changes if the underlying data sample didn't silently change too.
    """
    raise NotImplementedError("See docs/06_plan.md Week 1.")
