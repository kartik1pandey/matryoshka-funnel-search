"""Hugging Face Spaces entry point.

Thin wrapper only — the real Gradio app lives in
src/matryoshka_search/demo/web.py, same "entry point vs. implementation"
split as scripts/ vs src/ elsewhere in this project.
"""

from matryoshka_search.demo.web import demo

if __name__ == "__main__":
    demo.launch()
