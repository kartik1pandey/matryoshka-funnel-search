<div align="center">

<img src="./assets/banner.png" alt="Matryoshka Funnel Search" width="100%" />

# 🪆 Matryoshka Funnel Search

### Cross-framework (PyTorch + JAX) reproduction of production-style Matryoshka Representation Learning "funnel search" on Amazon's public product catalog

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-frozen%20backbone-EE4C2C)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-trainable%20head-4285F4)](https://jax.readthedocs.io/)
[![Paper](https://img.shields.io/badge/arXiv-2205.13147-b31b1b)](https://arxiv.org/abs/2205.13147)

**[Overview](#-overview) · [Method](#-method) · [Tech Stack](#-tech-stack) · [Quickstart](#-quickstart) · [Status](#%EF%B8%8F-status) · [Results](#-results) · [Citation](#-citation)**

</div>

---

## 📌 Overview

**Can a self-trained Matryoshka model, applied to Amazon's own public
product catalog, reproduce the storage/latency-vs-accuracy tradeoff that
production embedding systems (OpenAI, Voyage AI, Nomic, Google Gemini
embeddings) report at scale — cheaply enough for one person to build in a
few weeks?**

This repo is a from-scratch, cross-framework reproduction of
**Matryoshka Representation Learning (MRL)** — Kusupati et al., NeurIPS 2022
([arXiv:2205.13147](https://arxiv.org/abs/2205.13147)) — applied to a
text-to-image product search system over the
[Amazon Berkeley Objects](https://registry.opendata.aws/amazon-berkeley-objects/)
dataset, benchmarked honestly against retention numbers reported in the
literature. It is a faithful reproduction of an established,
production-adopted technique, not a novelty claim.

## 🧠 Method

<div align="center">
<img src="./assets/architecture.png" alt="Two-stage funnel search architecture" width="85%" />
</div>

1. A **frozen PyTorch OpenCLIP backbone** (`ViT-B-32`, `openai` weights) encodes product images and text into a shared 512-dim space. Never fine-tuned.
2. A **small trainable JAX/Flax projection head** is trained with the Matryoshka nested-loss trick: contrastive image-text loss is computed independently at every truncation length in `M = {8, 16, 32, 64, 128, 256, 512}` and summed, forcing early dimensions to be independently useful.
3. **Two-stage "funnel search"** — a real, named production pattern (see [Milvus's writeup](https://milvus.io/blog/matryoshka-embeddings-detail-at-multiple-scales.md)), not invented here — searches the full catalog cheaply at low dimension, then reranks only the shortlist at full dimension.

## 🧰 Tech Stack

| | |
|---|---|
| Frozen backbone | PyTorch + OpenCLIP `ViT-B-32` (`openai` weights) |
| Trainable head | JAX + Flax (NNX) + Optax |
| Dataset | [Amazon Berkeley Objects (ABO)](https://registry.opendata.aws/amazon-berkeley-objects/) — **verify the current license on the live page** before any use beyond personal/research/portfolio purposes (sources have disagreed between CC BY-NC 4.0 and CC BY 4.0) |

## ⚡ Quickstart

```bash
git clone https://github.com/kartik1pandey/matryoshka-funnel-search.git
cd matryoshka-funnel-search
make setup     # editable install (torch + jax + dev extras) + pre-commit hooks
make check     # lint, typecheck, test — same as CI
make demo      # interactive query CLI (needs data/checkpoints from the training pipeline)
```

This project intentionally runs two ML frameworks side by side (frozen
PyTorch backbone, trainable JAX head) — install the `torch` and `jax` extras
independently if you hit a dependency conflict between them. JAX GPU support
on native Windows is limited; use WSL2 or a cloud GPU instance for training,
CPU is fine for development and the demo path.

## 📁 Repo Layout

```
src/matryoshka_search/
├── data/     # ABO subset download + preprocessing
├── model/    # frozen PyTorch backbone + trainable JAX Matryoshka head
├── train/    # Matryoshka nested loss + Optax training loop
├── eval/     # retention metrics, funnel search, latency benchmarks
└── demo/     # interactive query CLI
tests/        # unit tests (CI-enforced)
```

## 🗺️ Status

✅ Weeks 1–3 complete, Week 4 in progress: real ABO data (15,000 products),
real frozen OpenCLIP embeddings, a real trained Matryoshka head + baseline, a
full funnel-search evaluation (Recall@K/nDCG@K at every dimension, retention
vs. literature, real latency — see [Results](#-results) below), and a working
interactive query demo (`make demo`). Remaining: full-catalog (147k) cost
projection and the final writeup pass.

## 📊 Results

Measured on 1,500 held-out validation queries against the real cached
embeddings for all 15,000 sampled ABO products (Recall@10, retention vs. the
512-dim reference point):

| Dimension | Recall@10 (MRL) | Recall@10 (baseline) | Retention (MRL) | Retention (baseline) |
|---|---|---|---|---|
| 512 | 0.681 | 0.677 | 100% | 100% |
| 256 | 0.662 | 0.650 | 97.2% | 96.0% |
| 64  | 0.600 | 0.509 | 88.1% | 75.1% |
| 16  | 0.355 | 0.145 | 52.2% | 21.4% |
| 8   | 0.139 | 0.041 | 20.4% | 6.1% |

**256-dim retention (97.2%) lands inside the 94–98% range reported by
independent academic benchmarking of production embedding models** — see
[docs/08_results.md](./docs/08_results.md) for the full table (all 7
dimensions, nDCG@10, funnel-search-specific retention, honest discussion of
where the gap to the literature widens at aggressive truncation, and why).

**Funnel search vs. brute force** (Stage 1 low-dim + Stage 2 full-dim
rerank, real latency on the 15,000-item catalog): **~4.2x faster** (20.1ms →
4.8ms per query) for a 1.5-point Recall@10 cost at `low_dim=64`. At a more
aggressive `low_dim=16`, the MRL model still recovers 93.6% of brute-force
recall — the baseline collapses to 62% — which is the specific scenario this
whole project exists to demonstrate.

## 📚 Citation

This project reproduces, and does not claim to improve on, the following work:

```bibtex
@inproceedings{kusupati2022matryoshka,
  title={Matryoshka Representation Learning},
  author={Kusupati, Aditya and Bhatt, Gantavya and Rege, Aniket and Wallingford, Matthew and Sinha, Aditya and Ramanujan, Vivek and Howard-Snyder, William and Chen, Kaifeng and Kakade, Sham and Jain, Prateek and Farhadi, Ali},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## 📄 License

Code: [MIT](./LICENSE). The ABO dataset has its own license — verify the
current terms on the [registry page](https://registry.opendata.aws/amazon-berkeley-objects/)
before relying on either.
