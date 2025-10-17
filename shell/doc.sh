#!/bin/bash
set -Eeuo pipefail

uv pip install mkdocs
uv pip install mkdocs-material
uv pip install mkdocstrings[python]
uv pip install mkdocs-glightbox
uv pip install mkdocs-llmstxt

uv run mkdocs serve