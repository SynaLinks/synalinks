#!/bin/bash
set -Eeuo pipefail

# Install dev/test dependencies (pytest, pytest-cov, keras-tuner, etc.) so the
# test runner and optional-dependency tests run instead of silently skipping on
# a fresh checkout.
uv pip install --group dev

uv run pytest --cov-config=pyproject.toml
uvx --from 'genbadge[coverage]' genbadge coverage -i coverage.xml