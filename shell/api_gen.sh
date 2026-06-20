#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Generating api directory with public APIs..."
# Generate API Files
uv run "${base_dir}"/api_gen.py

echo "Formatting api directory..."
# Format ONLY the generated files (the `synalinks/api` package and the
# generated top-level `synalinks/__init__.py`) — NOT the whole repo. Reusing
# the repo-wide `shell/format.sh` here reformatted every file in the tree on
# every regen; see `shell/format.sh` when you do want a repo-wide format.
api_paths=("${base_dir}/synalinks/api" "${base_dir}/synalinks/__init__.py")
uvx ruff check --config "${base_dir}/pyproject.toml" --fix "${api_paths[@]}"
uvx ruff format --config "${base_dir}/pyproject.toml" "${api_paths[@]}"
