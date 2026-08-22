# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Project scaffold: `src/` package layout, docs/, CI, Docker, pre-commit.
- Documentation set explaining the research motivation, architecture, methodology, and deployment plan (see `docs/`).
- Real ABO dataset download + deterministic subset selection (`data/abo_dataset.py`).
- Frozen OpenCLIP backbone wrapper with embedding precompute/cache (`model/backbone.py`).
- Matryoshka projection head in Flax NNX (`model/matryoshka_head.py`) and the Matryoshka nested-loss + InfoNCE implementation (`train/loss.py`).
- Optax training loop (`train/trainer.py`) and the training orchestration entry point (`scripts/train_matryoshka.py`), trained on the real cached embeddings.
- Retrieval-quality metrics, funnel search, and latency benchmarking (`eval/`), and the Week 3 evaluation entry point (`scripts/evaluate.py`) with real Recall@K/nDCG@K/latency results.
- Cost projection to the full 147,702-item ABO catalog, with an honest analytical-extrapolation methodology (`scripts/cost_projection.py`).
- Interactive terminal demo (`demo/cli.py`) and a Gradio browser demo (`demo/web.py`, `app.py`).
- Real-data result charts embedded in the README (`scripts/generate_result_charts.py`).

### Fixed
- Pinned `jax<0.11.1` — 0.11.1 shipped a PEP 695 type-alias syntax change in `jax/numpy/__init__.pyi` that broke mypy under this project's Python 3.11 target on a fresh CI install, same class of issue as the existing `numpy`/`orbax-checkpoint` pins.
