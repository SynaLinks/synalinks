#!/usr/bin/env python
"""Generate Colab-ready notebooks from the guide/example scripts.

For every ``docs/**/*.md`` page that renders a guide/example via an
``::: guides.<name>`` / ``::: examples.<name>`` mkdocstrings directive,
this script:

1. Builds ``notebooks/<guides|examples>/<slug>.ipynb`` from the matching
   ``.py`` file (module docstring -> markdown cell, code -> code cell),
   prepending a bootstrap cell that installs Synalinks, prompts for any
   hosted API keys, and installs/starts Ollama + pulls models when the
   example uses local ``ollama/`` models.
2. Injects (idempotently) an "Open in Colab" badge at the top of the
   ``.md`` page, pointing at the generated notebook on GitHub.

The notebook name drops the source file's ordinal prefix and is grouped
by guides/examples (``guides/22_sql_agent.py`` -> ``guides/sql_agent``).

The ``.py`` files stay the single source of truth: their code is copied
verbatim (only mkdocs-only ``# --8<--`` markers are stripped). The prose
is rewritten so it renders in a notebook: mermaid diagrams become images
(kroki.io), relative image paths become absolute, and cross-page ``.md``
links are unwrapped to plain text.

Usage:
    uv run python shell/gen_notebooks.py
"""

from __future__ import annotations

import ast
import base64
import json
import re
import shutil
import zlib
from pathlib import Path

# --- Repo coordinates Colab/raw fetch from (see mkdocs.yml repo_url) ----------
GITHUB_OWNER = "SynaLinks"
GITHUB_REPO = "synalinks"
GITHUB_BRANCH = "main"
NOTEBOOK_DIR = "notebooks"

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUT_DIR = REPO_ROOT / NOTEBOOK_DIR
RAW_BASE = (
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
)

BADGE_START = "<!-- colab-badge:start -->"
BADGE_END = "<!-- colab-badge:end -->"

# Hosted-provider model prefix -> the env var litellm expects for it.
HOSTED_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

DIRECTIVE_RE = re.compile(r"^:::\s+(guides|examples)\.([A-Za-z0-9_]+)\s*$", re.M)
OLLAMA_RE = re.compile(r"ollama/([A-Za-z0-9_.:\-]+)")
HOSTED_RE = re.compile(r"\b(" + "|".join(HOSTED_ENV) + r")/[A-Za-z0-9_.:\-]+")
ORDINAL_RE = re.compile(r"^\d+[a-z]?_")  # 22_ , 5a_ , 12b_

MERMAID_RE = re.compile(r"```mermaid[ \t]*\n(.*?)\n[ \t]*```", re.S)
IMG_REL_RE = re.compile(r"(!\[[^\]]*\]\()(?:\.\./)+assets/([^)]+)\)")
MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\.md[^)]*\)")
SNIPPET_MARKER_RE = re.compile(r"^[ \t]*#[ \t]*-{2}8<-{2}.*\n?", re.M)


def slug(stem: str) -> str:
    """guides stem -> notebook slug: drop the leading ordinal prefix."""
    return ORDINAL_RE.sub("", stem)


def kroki_image(diagram: str, diagram_type: str = "mermaid", fmt: str = "png") -> str:
    """Encode a diagram for kroki.io so it renders as an <img> in a notebook."""
    payload = zlib.compress(diagram.encode("utf-8"), 9)
    encoded = base64.urlsafe_b64encode(payload).decode("ascii")
    return f"https://kroki.io/{diagram_type}/{fmt}/{encoded}"


def render_prose(text: str) -> str:
    """Rewrite mkdocs-only markdown so it renders inside a notebook."""
    text = MERMAID_RE.sub(lambda m: f"![diagram]({kroki_image(m.group(1))})", text)
    text = IMG_REL_RE.sub(
        lambda m: f"{m.group(1)}{RAW_BASE}/docs/assets/{m.group(2)})", text
    )
    text = MD_LINK_RE.sub(lambda m: m.group(1), text)  # unwrap dead cross-page links
    return text


def split_docstring_and_code(py_path: Path) -> tuple[str, str]:
    """Return (rendered docstring, cleaned code) for a guide/example file."""
    source = py_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree, clean=False) or ""
    lines = source.splitlines(keepends=True)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        code = "".join(lines[tree.body[0].end_lineno :])
    else:
        code = source
    code = SNIPPET_MARKER_RE.sub("", code)  # drop mkdocs snippet-region comments
    return render_prose(docstring.strip("\n")), code.strip("\n") + "\n"


def build_bootstrap(code: str) -> str:
    parts = [
        "# @title Setup — run me first",
        "%pip install -q synalinks python-dotenv",
    ]

    hosted = sorted({HOSTED_ENV[m] for m in HOSTED_RE.findall(code)})
    if hosted:
        parts += [
            "",
            "import os, getpass",
            f"for _var in {hosted!r}:",
            "    if not os.environ.get(_var):",
            '        os.environ[_var] = getpass.getpass(f"{_var}: ")',
        ]

    ollama_models = sorted(set(OLLAMA_RE.findall(code)))
    if ollama_models:
        pulls = "\n".join(f"!ollama pull {m}" for m in ollama_models)
        parts += [
            "",
            "# This example uses local Ollama models. Install + start the server,",
            "# then pull the models it needs. Slow on a CPU runtime — prefer a GPU one.",
            "!curl -fsSL https://ollama.com/install.sh | sh",
            "import subprocess, time",
            'subprocess.Popen(["ollama", "serve"])',
            "time.sleep(5)",
            pulls,
        ]

    return "\n".join(parts)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook(docstring: str, code: str) -> dict:
    cells = []
    if docstring:
        cells.append(md_cell(docstring))
    cells.append(
        md_cell(
            "## Setup\nRun the cell below first to install Synalinks "
            "and configure access."
        )
    )
    cells.append(code_cell(build_bootstrap(code)))
    cells.append(code_cell(code))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def colab_url(package: str, name: str) -> str:
    return (
        f"https://colab.research.google.com/github/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/blob/{GITHUB_BRANCH}/{NOTEBOOK_DIR}/{package}/{name}.ipynb"
    )


def badge_block(package: str, name: str) -> str:
    url = colab_url(package, name)
    return (
        f"{BADGE_START}\n"
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({url})\n"
        f"{BADGE_END}\n"
    )


def inject_badge(md_path: Path, package: str, name: str) -> None:
    text = md_path.read_text(encoding="utf-8")
    block = badge_block(package, name)
    if BADGE_START in text:
        text = re.sub(
            re.escape(BADGE_START) + r".*?" + re.escape(BADGE_END) + r"\n?",
            block,
            text,
            flags=re.S,
        )
    else:
        text = block + "\n" + text
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)  # fully regenerated — drop stale notebooks
    count, seen = 0, {}
    for md_path in sorted(DOCS_DIR.rglob("*.md")):
        match = DIRECTIVE_RE.search(md_path.read_text(encoding="utf-8"))
        if not match:
            continue
        package, stem = match.group(1), match.group(2)
        py_path = REPO_ROOT / package / f"{stem}.py"
        if not py_path.exists():
            print(f"  ! {md_path.relative_to(REPO_ROOT)} -> missing {py_path.name}")
            continue

        name = slug(stem)
        clash = seen.get((package, name))
        if clash:
            print(f"  ! slug clash: {package}/{stem} vs {clash} -> keeping ordinal")
            name = stem
        seen[(package, name)] = stem

        docstring, code = split_docstring_and_code(py_path)
        out_path = OUT_DIR / package / f"{name}.ipynb"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(build_notebook(docstring, code), indent=1) + "\n", encoding="utf-8"
        )
        inject_badge(md_path, package, name)
        count += 1
        print(f"  ✓ {package}/{stem}.py -> {NOTEBOOK_DIR}/{package}/{name}.ipynb")

    print(f"\nGenerated {count} notebook(s) in {NOTEBOOK_DIR}/.")
    print("Commit the notebooks to the default branch so Colab can fetch them.")


if __name__ == "__main__":
    main()
