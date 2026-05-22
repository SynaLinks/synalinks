#!/bin/bash
set -Eeuo pipefail

uv pip install mkdocs
uv pip install mkdocs-material
uv pip install mkdocstrings[python]
uv pip install mkdocs-glightbox

# Regenerate Colab notebooks + "Open in Colab" badges from guides/examples.
uv run python shell/gen_notebooks.py

uv run zensical serve