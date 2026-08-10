"""Gradio browser UI for the funnel-search demo.

Same real pipeline as demo/cli.py (encode -> project -> funnel search) —
imports and reuses its tested functions directly rather than reimplementing
retrieval logic; this module only adds a UI around them, plus loading both
the Matryoshka and baseline heads/catalogs up front so the UI can switch
between them without reloading anything per-query.

Usage: pip install -e ".[web]"; python -m matryoshka_search.demo.web
(or `python app.py` from the repo root — the Hugging Face Spaces entry point
imports `demo` from here; see app.py's docstring)

Reads the same files as demo/cli.py: data/raw/manifest.json,
data/cache/{image_embeddings.npy, item_ids.json},
checkpoints/{matryoshka,baseline}_head.pkl.
"""

from __future__ import annotations

import gradio as gr
import numpy as np

from matryoshka_search.demo.cli import (
    CACHE_DIR,
    MANIFEST_PATH,
    load_catalog,
    load_head,
    project_catalog,
    run_search,
)
from matryoshka_search.model.backbone import OpenClipBackbone
from matryoshka_search.train.loss import MATRYOSHKA_DIMS

_ITEM_IDS, _TITLES, _IMAGE_PATHS, _IMAGE_EMBEDDINGS = load_catalog(MANIFEST_PATH, CACHE_DIR)
_HEADS = {name: load_head(f"{name}_head") for name in ("matryoshka", "baseline")}
_CATALOG_PROJ = {name: project_catalog(head, _IMAGE_EMBEDDINGS) for name, head in _HEADS.items()}
_BACKBONE = OpenClipBackbone()


def _encode_text(text: str) -> np.ndarray:
    return _BACKBONE.encode_texts([text])[0]


def _search(
    query: str, model_name: str, low_dim: int, k: int = 10
) -> tuple[list[tuple[str, str]], str]:
    if not query.strip():
        return [], "Type a query above and press Search."

    head = _HEADS[model_name]

    def project(x: np.ndarray) -> np.ndarray:
        return np.asarray(head(x[None, :]))[0]

    indices, timings = run_search(
        query, _encode_text, project, _CATALOG_PROJ[model_name], low_dim, 100, k
    )
    gallery = [
        (_IMAGE_PATHS[_ITEM_IDS[i]], f"{rank}. {_TITLES[_ITEM_IDS[i]]}")
        for rank, i in enumerate(indices, start=1)
    ]
    timing_text = (
        f"encode {timings['encode_seconds'] * 1000:.1f}ms  ·  "
        f"project {timings['project_seconds'] * 1000:.1f}ms  ·  "
        f"search {timings['search_seconds'] * 1000:.1f}ms  ·  "
        f"total {timings['total_seconds'] * 1000:.1f}ms"
    )
    return gallery, timing_text


def _on_submit(query: str, model_name: str, low_dim_str: str):
    return _search(query, model_name, int(low_dim_str))


with gr.Blocks(title="Matryoshka Funnel Search") as demo:
    gr.Markdown(
        "# 🪆 Matryoshka Funnel Search\n"
        "Real two-stage funnel search over 15,000 real Amazon product listings — "
        "a frozen OpenCLIP backbone plus a from-scratch-trained Matryoshka "
        "projection head. Pick a Stage-1 dimension and compare against the "
        "non-Matryoshka baseline. "
        "[Code + full writeup](https://github.com/kartik1pandey/matryoshka-funnel-search)."
    )
    with gr.Row():
        query = gr.Textbox(
            label="Search query",
            placeholder="e.g. wireless bluetooth headphones",
            scale=3,
        )
        model_name = gr.Radio(list(_HEADS), value="matryoshka", label="Model", scale=1)
        low_dim = gr.Dropdown(
            [str(d) for d in MATRYOSHKA_DIMS], value="64", label="Stage 1 dim", scale=1
        )
    search_button = gr.Button("Search", variant="primary")
    results = gr.Gallery(label="Top 10 results", columns=5, height="auto")
    timing = gr.Textbox(label="Per-stage timing", interactive=False)

    search_button.click(_on_submit, inputs=[query, model_name, low_dim], outputs=[results, timing])
    query.submit(_on_submit, inputs=[query, model_name, low_dim], outputs=[results, timing])


if __name__ == "__main__":
    demo.launch()
