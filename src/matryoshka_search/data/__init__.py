"""ABO dataset download and deterministic subset selection.

See docs/05_code_walkthrough.md#dataabo_datasetpy. CLIP-side image/text
preprocessing lives in model/backbone.py instead (delegates directly to
open_clip's own transform/tokenizer) — there is no separate preprocessing
module in this package.
"""
