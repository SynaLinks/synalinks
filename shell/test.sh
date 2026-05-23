#!/bin/bash
set -Eeuo pipefail

# Install dev/test dependencies (keras-tuner, etc.) so optional-dependency
# tests run instead of silently skipping on a fresh checkout.
uv pip install --group dev

uv run pytest --cov-config=pyproject.toml
uvx --from 'genbadge[coverage]' genbadge coverage -i coverage.xml