#!/bin/bash
set -Eeuo pipefail

uv pip install mkdocs
uv pip install mkdocs-material
uv pip install mkdocstrings[python]
uv pip install mkdocs-glightbox

# Regenerate Colab notebooks + "Open in Colab" badges from guides/examples.
uv run python shell/gen_notebooks.py

# Build the static site, then correct the asset-link depths that Zensical
# miscomputes for docstring-injected images (see shell/fix_doc_image_paths.py).
# `zensical serve` hosts at the root URL, where the surplus `../` is clamped and
# the bug stays hidden — so we build + fix + serve the *static* site to preview
# exactly what ships to the `/synalinks/` GitHub Pages sub-path.
uv run zensical build --clean
uv run python shell/fix_doc_image_paths.py site
# Zensical copies every docs/ file into the site, including the theme
# custom_dir (docs/.overrides) — drop that template fragment from the output.
rm -rf site/.overrides
uv run python -m http.server --directory site 8000