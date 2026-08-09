# Production image for the Matryoshka funnel-search demo/eval CLI.
# CPU-only base: this project's model is small enough that GPU is an
# optimization, not a requirement, for inference (training is documented
# separately in docs/03_environment_setup.md for GPU/cloud use).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by Pillow/OpenCLIP image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# CPU wheels for torch to keep the image small; swap for a CUDA base image
# and the matching torch/jax CUDA extras for GPU training (see docs/07_deployment.md).
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[torch,jax]"

COPY scripts ./scripts

ENTRYPOINT ["python", "-m", "matryoshka_search.demo.cli"]
