"""
# Documents Datasets: Streaming `.txt` and `.md` Files Into a KnowledgeBase

This example uses two built-in `synalinks` dataset loaders that walk a
directory tree and stream documents into a `KnowledgeBase`:

- `synalinks.TextDataset` reads `.txt` files verbatim and
  yields `TextDocument(filepath, text)` rows — one row per file. Use
  it for any flat-text corpus (logs, transcripts, scraped pages,
  notes, ...).
- `synalinks.MarkdownDataset` parses `.md` files into
  `MarkdownDocument(filepath, title, sections=[MarkdownSection(...)])`
  — one row per file, with the file split into heading-delimited
  sections. Use it when you want section-level retrieval (each section
  becomes an independently searchable chunk under its own heading
  path). Heading detection is delegated to `markdown-it-py`, so
  ATX / setext headings, fenced code blocks, and YAML front matter all
  work as you'd expect.

Both datasets are inputs-only and accepted directly by
`KnowledgeBase.update(...)`, so files stream in batch-by-batch.

## Storage strategy

`TextDocument` rows store one row per `.txt` file — flat shape, direct
upsert with `filepath` as the primary key.

`MarkdownSection` rows store one row per heading region of every `.md`
file — each section gets a stable `section_id` (`<filepath>#<path>`)
as its primary key, so re-running the loader upserts deterministically.
The `MarkdownDocument` the dataset yields is the nested view of a file;
this example flattens it to per-section rows because BM25 / vector
search chunks better at heading granularity.
"""

import asyncio
import os
import shutil
import tempfile
from typing import List

from dotenv import load_dotenv

import synalinks


class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user's question.")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Answer grounded in retrieved sections.")
    sources: str = synalinks.Field(description="Section paths that backed the answer.")


# =============================================================================
# Sample corpus — written to a tempdir so the example is self-contained
# =============================================================================


SAMPLE_TXT_FILES = {
    "notes/release-notes.txt": (
        "Release 2.3.0 ships hybrid retrieval for the knowledge base,\n"
        "DuckDB FTS index rebuild on every batch update, and a new\n"
        "MarkdownDataset loader for documentation corpora.\n"
    ),
    "notes/triage.txt": (
        "Customer reported slow vector search on the documents index.\n"
        "Root cause: missing HNSW index on the embedding column after\n"
        "a wipe_on_start=True restart. Fix: re-run setup_schema().\n"
    ),
}


SAMPLE_MD_FILES = {
    "guides/getting-started.md": """\
---
title: Getting Started
author: Synalinks Team
---

# Getting Started with Synalinks

Synalinks is a neuro-symbolic language model framework. This guide walks
through installing the package and running your first program.

## Installation

Install via `uv`:

```bash
# This `#` is inside a fenced code block — markdown-it-py classifies
# this as a code-fence token, so it is NOT treated as a heading.
uv add synalinks
```

## Your First Program

A program is a graph of modules with typed inputs and outputs.

### Defining Inputs

Use `synalinks.Input(data_model=...)` to declare the entry point of a
program.

### Generating Outputs

Wire a `synalinks.Generator(...)` to the input to produce a structured
result.

## Next Steps

Read the agents guide once you're comfortable with the basics.
""",
    "guides/knowledge-base.md": """\
Knowledge Base
==============

The `KnowledgeBase` is a dual-store abstraction over DuckDB (SQL) and
LadybugDB (graph).

Primary Key Convention
----------------------

The primary key is the first declared field of your DataModel — Synalinks
does not inject a synthetic `uuid` column.

### Why no UUID

So a KnowledgeBase can be pointed at a pre-existing DuckDB file or a
Ladybug store without rewriting rows.

### When to declare a UUID yourself

If you genuinely want one, put it first in your DataModel and populate
it. The framework treats it like any other identifier.

## Search

### Full-Text Search

Backed by DuckDB's FTS extension (BM25-style scoring).

### Vector Search

Backed by DuckDB's VSS extension (HNSW index).

### Hybrid Search

Reciprocal-rank fusion over the two — the default for `RetrieveKnowledge`.
""",
}


def _write_sample_corpus(root: str) -> None:
    """Materialize the sample `.txt` and `.md` files under `root`."""
    for relpath, content in {**SAMPLE_TXT_FILES, **SAMPLE_MD_FILES}.items():
        full = os.path.join(root, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)


# =============================================================================
# End-to-end demo
# =============================================================================


async def main():
    load_dotenv()

    workdir = tempfile.mkdtemp(prefix="synalinks_docs_example_")
    corpus_root = os.path.join(workdir, "corpus")
    os.makedirs(corpus_root)
    _write_sample_corpus(corpus_root)

    db_path = os.path.join(workdir, "documents.db")

    try:
        embedding_model = synalinks.EmbeddingModel(
            model="gemini/gemini-embedding-001",
        )

        knowledge_base = synalinks.KnowledgeBase(
            uri=f"duckdb://{db_path}",
            data_models=[synalinks.TextDocument, synalinks.MarkdownSection],
            embedding_model=embedding_model,
            metric="cosine",
        )

        # ------------------------------------------------------------------
        # Step 1 — Iterate TextDataset for inspection.
        # ------------------------------------------------------------------
        print("Step 1: TextDataset preview")
        print("=" * 60)

        text_dataset = synalinks.TextDataset(root=corpus_root, batch_size=4)
        for batch in text_dataset:
            (xs,) = batch
            for doc in xs:
                print(f"  [{doc.filepath}]  ({len(doc.text)} chars)")

        # ------------------------------------------------------------------
        # Step 2 — Stream `.txt` files into the KnowledgeBase.
        # `KnowledgeBase.update(...)` accepts a Dataset directly and
        # writes batch-by-batch, so this works for arbitrarily large
        # corpora without materializing everything in memory.
        # ------------------------------------------------------------------
        print("\nStep 2: Streaming TextDocument rows into the KB")
        print("=" * 60)
        text_ids = await knowledge_base.update(text_dataset)
        print(f"  Stored {len(text_ids)} .txt files: {text_ids}")

        # ------------------------------------------------------------------
        # Step 3 — Iterate MarkdownDataset and inspect the
        # parsed structure. Each row is a MarkdownDocument carrying a
        # list of MarkdownSection objects.
        # ------------------------------------------------------------------
        print("\nStep 3: MarkdownDataset preview")
        print("=" * 60)

        md_dataset = synalinks.MarkdownDataset(root=corpus_root, batch_size=4)
        all_sections: List[synalinks.MarkdownSection] = []
        for batch in md_dataset:
            (xs,) = batch
            for structured in xs:
                print(f"\n  {structured.filepath}  —  title: {structured.title!r}")
                for sect in structured.sections:
                    prefix = "    " + ("  " * max(sect.level - 1, 0))
                    name = sect.section_name or "(preamble)"
                    print(f"{prefix}h{sect.level} {name}   path={sect.path!r}")
                    all_sections.append(sect)

        # ------------------------------------------------------------------
        # Step 4 — Flatten sections into the `MarkdownSection` table.
        # The dataset's `MarkdownDocument` is the user-facing
        # abstraction; for retrieval we store one row per section so
        # BM25 / vector search chunks at heading granularity.
        # ------------------------------------------------------------------
        print("\nStep 4: Storing flattened MarkdownSection rows")
        print("=" * 60)
        section_ids = await knowledge_base.update(all_sections)
        print(f"  Stored {len(section_ids)} sections")

        # ------------------------------------------------------------------
        # Step 5 — Hybrid search over the section index.
        # ------------------------------------------------------------------
        print("\nStep 5: Hybrid search over MarkdownSection")
        print("=" * 60)

        queries = [
            "how do I install synalinks",
            "why does the framework not inject a uuid",
            "vector search backed by which extension",
        ]
        for q in queries:
            print(f"\n  Q: {q}")
            hits = await knowledge_base.hybrid_fts_search(
                q, table_name="MarkdownSection", k=3
            )
            for h in hits:
                print(
                    f"    - [{h.get('filepath')}] {h.get('path')!r}"
                    f"  (score={h.get('score', 0.0):.3f})"
                )

        # ------------------------------------------------------------------
        # Step 6 — RAG: retrieve sections + ground an answer.
        # ------------------------------------------------------------------
        print("\nStep 6: RAG pipeline grounded on the section index")
        print("=" * 60)

        language_model = synalinks.LanguageModel(
            model="gemini/gemini-3.1-flash-lite-preview",
        )

        inputs = synalinks.Input(data_model=Query)
        retrieved = await synalinks.RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            search_type="hybrid",
            k=3,
            return_inputs=True,
            return_query=True,
        )(inputs)
        answer = await synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
            instructions=(
                "Answer the question using ONLY the retrieved sections. "
                "List the section paths you used in `sources`."
            ),
        )(retrieved)

        rag = synalinks.Program(
            inputs=inputs,
            outputs=answer,
            name="docs_rag",
            description="RAG over .txt + .md documents",
        )

        for q in [
            "How do I install Synalinks?",
            "Why doesn't the KnowledgeBase generate UUIDs automatically?",
        ]:
            result = await rag(Query(query=q))
            print(f"\n  Q: {q}")
            print(f"  A: {result.get('answer')}")
            print(f"  sources: {result.get('sources')}")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
