"""Deploy the web demo to a Hugging Face Space.

A one-shot operational script, not part of the tested pipeline or CI — run
manually when you want to (re)publish the hosted demo linked from the
README. Uploads the package source plus the reduced dataset built by
scripts/prepare_space_subset.py (run that first).

Usage:
    python scripts/prepare_space_subset.py
    HF_TOKEN=hf_xxx python scripts/deploy_space.py --repo-id <username>/matryoshka-funnel-search

Uploads (via huggingface_hub, not a manual git clone/push): src/, app.py,
pyproject.toml, LICENSE, a generated Space README.md + requirements.txt,
and space_build/{data,checkpoints} from the prepare step above.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_README = """\
---
title: Matryoshka Funnel Search
emoji: 🪆
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Matryoshka Funnel Search — live demo

Real two-stage funnel search over a 750-item sample of Amazon Berkeley
Objects product listings, using a from-scratch-trained Matryoshka
projection head on a frozen OpenCLIP backbone. Full code, real results
(measured on the full 15,000-item catalog), and the honest comparison
against the literature: <https://github.com/kartik1pandey/matryoshka-funnel-search>
"""

REQUIREMENTS = """\
--extra-index-url https://download.pytorch.org/whl/cpu
-e .[torch,jax,web]
"""


def _stage(staging_dir: Path, repo_root: Path, space_build_dir: Path) -> None:
    shutil.copytree(repo_root / "src", staging_dir / "src")
    shutil.copy(repo_root / "app.py", staging_dir / "app.py")
    shutil.copy(repo_root / "pyproject.toml", staging_dir / "pyproject.toml")
    shutil.copy(repo_root / "LICENSE", staging_dir / "LICENSE")
    shutil.copytree(space_build_dir / "data", staging_dir / "data")
    shutil.copytree(space_build_dir / "checkpoints", staging_dir / "checkpoints")
    (staging_dir / "README.md").write_text(SPACE_README, encoding="utf-8")
    (staging_dir / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id", required=True, help="e.g. yourusername/matryoshka-funnel-search"
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in the environment before running this script.")

    repo_root = Path(__file__).resolve().parent.parent
    space_build_dir = repo_root / "space_build"
    if not (space_build_dir / "data").exists():
        raise SystemExit("Run scripts/prepare_space_subset.py first.")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = Path(tmp) / "space"
        staging_dir.mkdir()
        _stage(staging_dir, repo_root, space_build_dir)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            repo_type="space",
            commit_message="Deploy web demo",
        )

    print(f"Deployed to https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
