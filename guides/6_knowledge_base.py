"""
# Knowledge Base

So far the programs you have built have used only what the language
model already knows from its pre-training. That works for "What is the
capital of France?" — but not for "What did our company decide in
yesterday's meeting?" In this guide we add a memory that lives
*outside* the LM: a **knowledge base** (KB) the program can search at
runtime.

The mental picture to start with is a labeled filing cabinet. Each
drawer (which we'll call a **table**) holds records of one shape; an
**index** is a precomputed lookup structure (like the index at the back
of a textbook) that makes searches fast; a **query** is a request that
returns the records that best match.

A slightly more formal description: a knowledge base is a triple
`(S, I, Q)`.

- `S` is the set of stored records. Every record obeys a fixed schema
  (it has a known set of typed fields).
- `I` is one or more indices built over `S`.
- `Q` is a family of query operators that take a search request and
  return a **ranked** subset of `S` — records sorted from most to
  least relevant.

In Synalinks, `S` is defined by `DataModel` classes (the Pydantic-style
typed records you have seen since [Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/)). The indices are provided by
**DuckDB**, an embedded SQL engine, plus a couple of DuckDB extensions
for text and vector search. `Q` is exposed on a `KnowledgeBase`
object via five complementary retrieval methods —
`fulltext_search` (BM25), `similarity_search` (vector),
`regex_search` (RE2 patterns), `hybrid_fts_search` (vector + BM25
fused with Reciprocal Rank Fusion), and `hybrid_regex_search`
(vector + regex, same fusion) — plus a raw `query()` escape hatch
for arbitrary SQL.

## Why Put Knowledge Outside the Model

A language model is a fixed function. Once trained, its weights are
frozen. At inference time the only "memory" the model has is whatever
text you put into its **context window** — the bounded buffer of tokens
it reads on each call. That gives us two hard limits:

1. **Parameter cutoff.** Weights are frozen at the end of training. A
   fact discovered yesterday simply cannot appear in the model unless
   you either retrain it (expensive) or paste the fact into the
   context window at query time (cheap).
2. **Context bound.** The context window is finite — typically a few
   thousand to a few hundred thousand tokens. You cannot paste an
   entire corpus into every prompt. And even if you could, longer
   contexts degrade quality and cost more.

A knowledge base **externalizes** state. Retrieval — picking the
records relevant to a question — becomes a deterministic, auditable
preprocessing step that selects the small slice of context the
(non-deterministic) generator will then read. When something goes
wrong, you can isolate the bug to the boundary between *symbolic*
retrieval and *neural* generation, which is much easier to debug than
"the model just hallucinated."

```mermaid
graph LR
    A["Query"] --> B["Retriever (KB)"]
    B --> C["Top-k records"]
    C --> D["Generator (LM)"]
    A --> D
    D --> E["Grounded answer"]
```

The arrow from `Query` directly into the `Generator` is deliberate:
the original query is needed both to *select* the context (via the
retriever) and to tell the generator what the user actually asked.

This whole pattern — retrieve, then generate — has a name you will see
everywhere in the field: **RAG**, for Retrieval-Augmented Generation.

## Architecture

A single DuckDB file stores both the rows and the indices. DuckDB is
an **embedded SQL database**, similar in spirit to SQLite: it runs
inside your Python process, and the entire database is one file on
disk. Each `DataModel` class maps to one SQL table. Indices are built
lazily — the first call to a search method on a table triggers index
construction; subsequent calls reuse it.

```mermaid
graph TD
    A["DataModel classes"] --> B["KnowledgeBase"]
    B --> C["DuckDB file"]
    C --> D["Row store"]
    C --> E["FTS index (BM25)"]
    C --> F["HNSW vector index"]
    G["Search call"] --> H{"search_type"}
    H -->|fulltext| E
    H -->|similarity| F
    H -->|hybrid| I["Reciprocal-rank fusion"]
    E --> I
    F --> I
    D --> J["Ranked records"]
    E --> J
    F --> J
    I --> J
```

Because DuckDB is embedded, the database lives inside your Python
process. There is no server to start, no network hop, no separate
lifecycle to manage. The trade-off: two processes that try to write to
the same file at once have to coordinate through the filesystem, which
is fragile. For a production workload with many concurrent writers,
use a hosted store instead.

## Building a Knowledge Base

```python
import synalinks

class Document(synalinks.DataModel):
    \"\"\"A document in the knowledge base.\"\"\"
    id: str = synalinks.Field(description="Unique document ID")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Document content")

kb = synalinks.KnowledgeBase(
    uri="duckdb://my_database.db",
    data_models=[Document],
    embedding_model=embedding_model,  # required only for similarity/hybrid
    metric="cosine",
    wipe_on_start=False,
)
```

Two rules the system relies on are worth burning into memory:

- **The first declared field is the primary key.** A **primary key**
  is the field that uniquely identifies a record. `update()` here is
  an **upsert** (insert-or-update): if a record with the same key
  exists, it is replaced; if not, a new row is inserted. The big trap:
  reordering the fields of your `DataModel` silently changes which
  field is the key, which breaks deduplication. Keep the key field
  first and do not move it.
- **One table per `DataModel` class.** If you call `search` with a
  class you never passed to `data_models=[...]`, you get a loud error
  rather than a silent empty result. The framework prefers to fail
  loudly.

| Parameter | Meaning |
|-----------|---------|
| `uri` | Connection string. For embedded DuckDB use `duckdb://<path>`. |
| `data_models` | Schema set. Each class becomes a table. |
| `embedding_model` | Required for vector indices; optional otherwise. |
| `metric` | `cosine`, `l2` (Euclidean), or `ip` (inner product). |
| `wipe_on_start` | If `True`, drops all tables on construction. |

## The Five Search Operators

We walk through them in order of conceptual complexity:
**BM25 fulltext** → **vector similarity** → **regex** → and the
two **hybrid** combinations (vector + BM25, vector + regex) that
fuse pairs of them via Reciprocal Rank Fusion. Pick whichever
signal your query expresses. If you're unsure, **`hybrid_fts_search`
is the production-default** — it captures both lexical precision
and semantic recall.

### 1. Full-text search (BM25)

Full-text search answers the question: *"which records contain the
words in my query?"* The classic scoring function for this is **BM25**.
Intuitively, BM25 gives a record a higher score when:

- it contains more of the query terms (**term frequency** — the more
  often a word appears, the more relevant the record is),
- those terms are *rare* in the corpus overall (**inverse document
  frequency** — words like "the" appear everywhere and tell you almost
  nothing about which document is relevant; rare words are more
  informative), and
- the record is not unusually long (long records are penalized so they
  do not win just by accident of size).

Term-frequency contribution **saturates**, meaning each successive
occurrence of a word counts less than the previous one. The tenth
occurrence of "neural" adds less to the score than the first. BM25
ignores meaning entirely; it sees only the literal words.

```python
results = await kb.fulltext_search(
    "machine learning neural networks",
    table_name="Document",   # one table per call
    k=10,
    threshold=None,          # optional lower bound on BM25 score
)
```

Use BM25 when the user's vocabulary tends to match the corpus's
vocabulary, and when speed and predictability matter. Its main failure
mode is the **lexical gap**: a query like "how do computers learn?"
will find nothing in a corpus that only contains the phrase "machine
learning", because no query word literally appears in the documents.

### 2. Similarity search (vector)

A **vector embedding** is a fixed-length list of numbers — typically a
few hundred floats — produced by a neural network. The network is
trained so that *semantically* similar texts get *numerically* nearby
vectors. So "machine learning" and "how computers learn" land close
together in the vector space even though they share no words.

Similarity search works in three steps:

1. **At insert time**, each record's designated text field is
   converted to a vector by the embedding model.
2. **At query time**, the query is embedded by the *same* model.
3. The index returns the `k` records whose vectors are closest to the
   query vector under your chosen `metric` (typically cosine
   similarity).

The index structure is called **HNSW** — Hierarchical Navigable Small
World, an *approximate* nearest-neighbor data structure. The word
"approximate" is important: exact nearest-neighbor search over
millions of vectors would be too slow, so HNSW trades a tiny amount of
accuracy for orders-of-magnitude speedup.

```python
results = await kb.similarity_search(
    "how do computers learn",   # semantic match for "machine learning"
    table_name="Document",
    k=10,
    threshold=0.7,              # cosine similarity floor
)
```

This closes the lexical gap. Two cautions:

- **Embeddings are not free.** You pay for a model call at both
  insert time (per record) and query time (per query).
- **`threshold` is metric-dependent.** A value of `0.7` under cosine
  similarity (a score bounded in `[-1, 1]`, where 1 means identical)
  means "fairly similar." Under L2 distance (Euclidean distance,
  unbounded, where smaller means closer), `0.7` means something
  *entirely* different. Always pick the threshold in the units of the
  metric you actually configured.

### 3. Hybrid: Vector + BM25 (`hybrid_fts_search`)

Hybrid search runs both retrievers — BM25 *and* vector — and **fuses**
their rankings into a single combined ranking. Synalinks uses
**Reciprocal Rank Fusion (RRF)**: each candidate's final score is a
weighted sum of `1 / (k + rank)` from each retriever, where `rank` is
its position in that retriever's list. The intuition: being near the
top of *either* list is strong evidence, and RRF rewards documents
that show up well in multiple rankings — without requiring the
underlying scores to be on comparable scales.

```python
results = await kb.hybrid_fts_search(
    "machine learning basics",
    table_name="Document",
    k=10,
)
```

(There is no per-retriever weight knob — RRF is rank-based, so the
two retrievers contribute symmetrically. The only fusion knob is
`k_rank`, the RRF smoothing constant. Optional `similarity_threshold`
and `fulltext_threshold` arguments filter each retriever's input
before fusion.)

Hybrid is the standard default for production RAG. BM25 anchors
precise terminology (proper names, product codes, identifiers) where
literal matching is essential, while the vector path recovers
paraphrases and synonyms.

### 4. Hybrid: Vector + Regex (`hybrid_regex_search`)

The vector-plus-regex sibling of `hybrid_fts_search`. Same Reciprocal
Rank Fusion, but the second retriever is **regex matching** (RE2
syntax) instead of BM25. Use it when the query has both a *semantic*
shape ("error in the auth layer") and an *exact textual* shape
("`HTTP/\\d{3}\\s+ERROR`") — vectors get the semantics, regex pins
down the literal pattern, and the two signals merge.

```python
results = await kb.hybrid_regex_search(
    text_or_texts="error in auth",
    pattern_or_patterns=r"HTTP/\\d{3}\\s+ERROR",
    table_name="LogLine",
    k=10,
)
```

Pass `pattern_or_patterns=None` to skip the regex half (degenerates
to a plain vector search) — useful when the LM hasn't decided what
the literal shape should be. Without an embedding model configured,
the call gracefully falls back to regex-only.

### 5. Regex Only (`regex_search`)

Pure pattern matching against the string fields of each table.
DuckDB ships RE2 (Google's regex library), so evaluation is
**linear-time** — no catastrophic-backtracking surface even on
untrusted patterns.

```python
results = await kb.regex_search(
    pattern=r"\\bcustomer_id=\\d{6}\\b",
    table_name="LogLine",
    fields=["message"],          # optional column filter
    case_sensitive=False,        # default True
    k=10,
)
```

`fields` defaults to every string-typed field on the schema; supply
it when you want to scan only certain columns (e.g., the `body` of
an article but not its `tags`).

## CRUD: Storing and Reading Records

**CRUD** stands for **C**reate, **R**ead, **U**pdate, **D**elete —
the four basic database operations. Synalinks exposes them as async
methods on `KnowledgeBase`.

### Upsert

```python
doc = Document(
    id="doc1",
    title="Introduction to AI",
    content="Artificial intelligence is...",
)

await kb.update(doc.to_json_data_model())
```

Calling `update` twice with the same primary key replaces the existing
row; it does not append. If you want append semantics, generate a fresh
unique key (for example a UUID) per record before calling `update`.

### Read by primary key

```python
result = await kb.get(
    "doc1",
    table_name="Document",
)
```

### Enumerate

```python
all_docs = await kb.getall(
    table_name="Document",
    limit=100,
    offset=0,
)
```

### Delete records

```python
# One id at a time.
n_deleted = await kb.delete("doc1", table_name="Document")

# Or a batch — the return value is the number of rows that
# actually matched and got removed.
n_deleted = await kb.delete(
    ["doc1", "doc2", "ghost"],   # "ghost" doesn't exist → not counted
    table_name="Document",
)
```

The delete rebuilds the FTS and HNSW indexes once afterwards, so a
subsequent `fulltext_search` / `similarity_search` won't return the
deleted row as a stale hit.

### Drop a whole table

```python
# Returns True if a table was dropped, False if no such table existed.
dropped = await kb.drop_table("Document")
```

`drop_table` also drops the table's FTS index (DuckDB stores it in a
sibling `fts_main_<table>` schema that doesn't cascade with the table)
and the HNSW vector index, then forgets the table in the adapter's
known-models list so default-table searches stop seeing it.

### Raw SQL escape hatch

```python
results = await kb.sql(
    "SELECT id, title FROM Document WHERE title LIKE ?",
    params=["%Learning%"],
)
```

Always use **parameterized queries**: the `?` placeholder is filled in
by the database *after* the SQL has been parsed, so user input can
never be mistaken for SQL syntax. This is how you avoid SQL injection
attacks — a class of security vulnerability you should learn to spot
even if you never become a security engineer.

#### Letting an LM write the SQL: `read_only=True`

`kb.sql` is also how you let a language model write SQL against
the KB (an "SQL agent" — see the [SQL Agent example](https://synalinks.github.io/synalinks/Code%20Examples/SQL%20Agent/)).
The model's output is, by definition, untrusted: it may be
malformed, mutating, or trying to read files. Passing
`read_only=True` enables two layers of defence:

1. **Parser check (blocks writes).** The SQL is parsed with
   DuckDB's own parser and rejected unless every statement is a
   `SELECT`. This catches multi-statement injection
   (`SELECT 1; DROP TABLE x`), `COPY (SELECT …) TO 'file'`
   exfiltration, `ATTACH`, `EXPORT`, and every other side-effecting
   statement. It is the *only* layer that blocks writes — the
   adapter's underlying connection is read-write, so the parser is
   what keeps untrusted SQL read-only.
2. **Sandbox (blocks external I/O).** The persistent connection
   has `enable_external_access=false` set at construction time,
   so `SELECT` table functions that touch the filesystem or
   network — `read_csv`, `read_parquet`, `read_json`, `read_blob`,
   `glob`, the httpfs/S3 variants — return a permission error
   instead of leaking files. Without this layer,
   `SELECT * FROM read_csv('/etc/passwd', ...)` would pass the
   parser check because it is a syntactically valid `SELECT`.

```python
# What you give the LM:
result = await kb.sql(llm_generated_sql, read_only=True)
```

The default for `kb.sql` is `read_only=True`. Pass
`read_only=False` only from call sites *you* control — those skip
the parser check and accept any SQL on the same sandboxed
connection.

### Closing the KB

The KB holds a single persistent DuckDB connection for its
lifetime, so one process can run many operations back-to-back
without paying the open + extension-load cost on each call. The
trade-off: this process holds DuckDB's exclusive file lock until
the KB is closed. Call `kb.adapter.close()` (or just let the KB go
out of scope — `__del__` cleans up best-effort) before opening
another process against the same file.

### Encrypted databases

Pass `encryption_key=` to `KnowledgeBase` to open (or create) an
encrypted DuckDB file:

```python
kb = synalinks.KnowledgeBase(
    uri="vault.duckdb",
    data_models=[Document],
    encryption_key="my-passphrase",   # keep out of source control
)
```

A few things to know:

- **The key is never serialised.** It does not appear in
  `kb.get_config()`, in `repr(kb)`, or in any saved program file.
  When you reload a program that uses an encrypted KB, you must
  re-supply the key — exactly the same shape as a database
  password.
- **Wrong / missing key fails loudly.** `Invalid Input Error: Wrong
  encryption key used to open the database file` for a mismatch;
  `Cannot open encrypted database "…" without a key` for the
  no-key case.
- **One process at a time.** Encryption doesn't change the
  exclusive-file-lock story — only one adapter at a time can
  attach the file. Use separate files for separate processes, or
  put a shared service in front.

## Loading from Files

The CRUD methods above insert one row (or a list of rows) at a time —
fine for hand-curated content or live writes from your application, but
the wrong tool when you already have a CSV / Parquet / JSON / JSONL
file on disk and want to get its contents into the KB as fast as
possible. Two distinct paths cover the file-ingestion case, and they
trade speed for transformation power in opposite directions.

```mermaid
graph LR
    A["Source file"] --> B{"Does the source<br/>need transformation<br/>row-by-row?"}
    B -->|"No — load as-is"| C["kb.from_csv / from_parquet<br/>/ from_json / from_jsonl"]
    B -->|"Yes — rename,<br/>derive, reshape"| D["kb.update(CSVDataset / …)"]
    C --> E["Native DuckDB load<br/>(~25× faster)"]
    D --> F["Python row pipeline<br/>(Pydantic + Jinja)"]
```

Pick the **fast path** (`kb.from_*`) when the file can be loaded as-is
— you don't need to rename columns, derive fields, or otherwise
rewrite each row. The schema is inferred directly from the file, with
the first column promoted to PRIMARY KEY. Pick the **streaming path**
(`kb.update(<...>Dataset(...))`) when you do need to rewrite rows
through a Jinja template before storage. The streaming path is what
HuggingFace, Parquet, and CSV `Dataset` objects feed into.

The performance gap is large enough to matter — see
`benchmarks/bench_kb_ingest.py` for the full table. At 10 000 rows on
a typical laptop:

| Path | CSV | Parquet | JSON | JSONL |
|---|---|---|---|---|
| `kb.from_*` (fast) | ~500 ms | ~450 ms | ~560 ms | ~500 ms |
| `kb.update(<…>Dataset)` | ~12 s | ~12 s | ~11 s | ~12 s |

The streaming path is bottlenecked at ~850 rows/second by per-row
Python overhead (Pydantic validation, Jinja template rendering, schema
sanitization) regardless of source format. The fast path runs the
INSERT inside DuckDB, with no Python on the per-row hot loop.

### The fast path: `kb.from_csv` / `from_parquet` / `from_json` / `from_jsonl`

All four methods share the same shape. You don't pre-declare a
`DataModel` for the target table — the schema is read straight from
the file. The call returns the constructed `SymbolicDataModel`, which
is the handle you pass to subsequent `get` / `search` calls.

If you omit `name`, the table name is derived from the file's stem;
either way it's normalized to PascalCase, so kebab-case
(`my-articles.csv` → `MyArticles`), snake_case, and free-form
(`"my articles"`) all converge to the same identifier.

```python
documents = await kb.from_csv(
    "docs.csv",
    table_name="Document",                          # optional; here just
    table_description="Knowledge-base articles.",   # being explicit
)

# Equivalent — table named `Articles` from the filename stem:
articles = await kb.from_parquet("articles.parquet")
posts    = await kb.from_json("posts.json")
events   = await kb.from_jsonl("events.jsonl")

# Returned models carry the post-load table name — pass it back in:
hits = await kb.fulltext_search(
    "python",
    table_name=documents.get_schema()["title"],
    k=5,
)
```

What happens under the hood:

1. The persistent sandboxed connection is briefly torn down — DuckDB
   enforces a single-writer lock per database file, and the native
   readers (`read_csv`, `read_parquet`, `read_json`) need
   `enable_external_access=true`, which the sandboxed connection
   intentionally blocks.
2. A throwaway non-sandboxed connection runs `DESCRIBE SELECT *
   FROM read_*(?)` to introspect the file's column shape, then
   `CREATE TABLE IF NOT EXISTS <name> (...)` with the first column
   promoted to PRIMARY KEY.
3. One `INSERT INTO <name> (cols…) SELECT cols… FROM read_*(?) ON
   CONFLICT (pk) DO UPDATE SET …` — so existing rows are overwritten
   on a primary-key match, just like the single-row `update` call.
4. The persistent connection is reopened with the sandbox re-applied
   (so `kb.sql(read_only=True)` still refuses external readers
   afterwards).
5. The post-load table is reflected back into a `SymbolicDataModel`
   (using the same column-introspection helpers `kb` uses everywhere
   else), with the optional `description` attached at the schema's
   top level.
6. The FTS index is rebuilt. The HNSW vector index is rebuilt too,
   but only when an embedding model is configured *and* the table has
   at least one non-NULL embedding (see *Embeddings on the fast path*
   below).

If you want the symbolic data model for a table later — say after
re-opening a KB pointed at the same file — call
`kb.get_symbolic_data_models()` to enumerate every table the adapter
knows about.

#### Format-specific notes

**CSV.** Types are inferred by DuckDB's CSV reader — same behaviour
as the Parquet / JSON paths. A column of digits comes out as
`BIGINT`, a column of decimals as `DOUBLE`, a column of text as
`VARCHAR`. The auto-detector is conservative about strings that
look numeric: zero-padded IDs like `"00123"` stay text, so id columns
formatted with leading zeros survive intact. If you need a different
type than what was inferred, run `ALTER TABLE … ALTER COLUMN col
TYPE …` after the load. Pass `delimiter`, `encoding`, and `header`
to customize.

```python
docs = await kb.from_csv(
    "docs.tsv",
    table_name="Document",
    delimiter="\t",
    encoding="utf-8",
    header=True,
)
```

**Parquet.** The schema is explicit in the file footer, so there's no
inference guesswork — types match end-to-end whenever the source file
and the data model agree.

**JSON.** The file must be a top-level array of objects:
`[{"id": "a", "text": "..."}, ...]`. Single-object files raise a clear
error pointing at JSONL.

**JSONL** (one JSON object per line). Right for very large sources
that aren't a single array. Streamed inside DuckDB; not loaded into
memory.

#### Embeddings on the fast path

The bulk load does **not** insert the embedding column — the source
files typically don't contain precomputed vectors. The HNSW vector
index is auto-built only when an embedding model is configured *and*
the table already has rows with non-NULL embeddings (e.g., from a
previous `update()` call that populated them). So:

- **Don't need vector search?** `kb.from_*` is complete by itself.
  FTS works against the freshly-loaded rows immediately.
- **Need vector search?** Use the streaming path with a `Dataset` and
  the `EmbedKnowledge` module (see below), or run a follow-up `update`
  that populates the embedding column. The bulk path is for the
  "lexical-only" or "embeddings already in the file" cases.

### The streaming path: `kb.update(<...>Dataset(...))`

When the source rows need transformation — column renames, deriving a
field from two others, normalizing a date, anything Jinja-shaped —
build a `Dataset` and hand it to `kb.update`. The dataset iterates the
file batch-by-batch, runs each row through your Jinja `input_template`
to produce a JSON payload, validates it against the `DataModel`, and
sends each batch as one `update` call.

```python
ds = synalinks.CSVDataset(
    path="raw_docs.csv",
    input_data_model=Document,
    input_template='''{
        "id": {{ row_id | tojson }},
        "title": {{ headline | tojson }},
        "content": {{ body | tojson }}
    }''',
    batch_size=64,
)
ids = await kb.update(ds)
```

Here the source columns are `row_id`, `headline`, `body`, but the
stored shape is `id`, `title`, `content` — the template performs the
rename per row. The same pattern works for `synalinks.ParquetDataset`,
`synalinks.JSONDataset`, `synalinks.JSONLDataset`, and
`synalinks.HuggingFaceDataset`.

Streaming is memory-bounded: only one batch is held at a time,
regardless of source size. This is the path to use for files larger
than RAM, for HF datasets streamed off the network, and any time the
template needs to do real work.

| Knob | What it controls |
|------|------------------|
| `batch_size` | Examples per `adapter.update` call. Bigger = fewer DB round-trips and fewer FTS rebuilds. |
| `limit` | Cap on total rows iterated. Also enables `len(ds)` for streaming sources. |
| `repeat` | Emit each raw row N times in a row. Used by GRPO-style RL for rollouts. |

`kb.update(dataset)` only accepts **inputs-only** datasets (no
`output_template`). A labeled dataset configured for training raises a
clear `ValueError` — the KB stores records, not `(input, target)`
pairs.

## Knowledge Modules: KB Operations Inside Programs

The methods above are the low-level interface. **Modules** wrap them
as reusable building blocks for the Functional API, so you can drop
retrieval directly into a larger `Program`.

### RetrieveKnowledge

`RetrieveKnowledge` takes an input record (often the user query), asks the
language model to write a good search string from it, runs the chosen
search operator, and emits both the original input and the retrieved
records downstream.

```mermaid
graph LR
    A["Input record"] --> B["LM: synthesise query"]
    B --> C["KB.search (type)"]
    C --> D["Retrieved records"]
    A --> E["Output"]
    D --> E
```

```python
retrieved = await synalinks.RetrieveKnowledge(
    knowledge_base=kb,
    language_model=lm,
    search_type="hybrid_fts",   # see below
    k=10,
    return_inputs=True,
)(inputs)
```

`search_type` mirrors the KB's five operators:

- `"similarity"` — vector only.
- `"fulltext"` — BM25 only.
- `"hybrid_fts"` (default) — vector + BM25 fused with RRF. The legacy
  spelling `"hybrid"` is accepted as an alias.
- `"regex"` — RE2 regex against string fields. The LM is instructed
  to emit *regex patterns* in the `search` list instead of natural-
  language queries.
- `"hybrid_regex"` — vector + regex, fused with RRF. The LM emits
  both a natural-language `search` list (vector side) and a
  `patterns` list (regex side), which means the output schema picks
  up a `patterns` field for this mode only.

Setting `return_inputs=False` discards the original input from the output.
That is rarely what you want, because the generator downstream usually
needs both the question and the retrieved context to write a good answer.

### UpdateKnowledge

```python
stored = await synalinks.UpdateKnowledge(
    knowledge_base=kb,
)(extracted_data)
```

### EmbedKnowledge

```python
embedded = await synalinks.EmbedKnowledge(
    embedding_model=embedding_model,
    in_mask=["content"],   # subset of fields to embed
)(inputs)
```

`in_mask` is the explicit list of textual fields that get
concatenated and fed to the embedding model. Think of it as a
contract you set: embedding *every* field is wasteful and dilutes the
signal; embedding *none* means vector search will never find this
record (zero recall).

## A minimal RAG pipeline

```mermaid
graph LR
    A["Query"] --> B["RetrieveKnowledge"]
    B --> C["{query, retrieved}"]
    C --> D["Generator"]
    D --> E["Answer"]
```

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Answer based on context")

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    kb = synalinks.KnowledgeBase(
        uri="duckdb://knowledge.db",
        data_models=[Document],
    )

    inputs = synalinks.Input(data_model=Query)

    retrieved = await synalinks.RetrieveKnowledge(
        knowledge_base=kb,
        language_model=lm,
        search_type="fulltext",
        k=5,
        return_inputs=True,
    )(inputs)

    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(retrieved)

    rag = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="rag_pipeline",
    )

    result = await rag(Query(query="What is machine learning?"))
    print(result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Expected output

Running the demonstration below produces:

```
============================================================
Example 1: Knowledge Base with Full-Text Search
============================================================

Storing documents...
  Stored: Introduction to Python
  Stored: Machine Learning Basics
  Stored: Deep Learning
  Stored: Natural Language Processing

============================================================
Example 2: Full-Text Search
============================================================

Search: 'programming language'
Found 2 results:
  - Natural Language Processing: NLP enables computers to understand and process hu...
  - Introduction to Python: Python is a high-level programming language....

============================================================
Example 3: Get by ID
============================================================

Get doc2:
  Title: Machine Learning Basics
  Content: Machine learning is a subset of AI that enables systems to learn.

============================================================
Example 4: Get All Records
============================================================

All documents (4):
  - doc1: Introduction to Python
  - doc2: Machine Learning Basics
  - doc3: Deep Learning
  - doc4: Natural Language Processing

============================================================
Example 5: RAG Pipeline
============================================================

RAG Query: What is Python?
Answer: Python is a high-level programming language.

RAG Query: Tell me about neural networks
Answer: Deep learning uses neural networks with many layers.

============================================================
Example 6: Raw SQL Query
============================================================

SQL: SELECT WHERE title LIKE '%Learning%'
  - doc2: Machine Learning Basics
  - doc3: Deep Learning
```

Notice that BM25 ranks "Natural Language Processing" above "Introduction
to Python" for the query "programming language". The NLP record contains
the literal substring "Language" prominently (in a short title and body),
which gives it a high term-frequency contribution, even though the Python
record is the more semantically relevant answer. This is the textbook
lexical-overlap trap, and a good argument for hybrid search whenever
recall (finding the right answer) matters more than raw throughput.

## Things That Will Bite You

A short list of failure modes worth scanning for before you ship a KB:

- **Schema drift.** If you add, rename, or retype a field on a
  `DataModel`, existing rows do *not* automatically migrate to the
  new shape. During development, drop the database
  (`wipe_on_start=True`) or write a migration script.
- **Missing embedding model.** Calling `similarity_search` on a KB
  built with `embedding_model=None` raises an error at *query* time,
  not at construction time. Decide up front whether you will need
  vector search. (`hybrid_fts_search` and `hybrid_regex_search`
  degrade gracefully in this case — they fall back to the non-vector
  half rather than erroring.)
- **Primary-key collision.** `update` silently overwrites the
  existing row on a key match. If that is wrong for your use case,
  generate a unique key per record (a UUID, say) before calling
  `update`.
- **Threshold semantics depend on the metric.** Cosine thresholds are
  bounded in `[-1, 1]`; L2 thresholds are unbounded distances
  (smaller = closer); BM25 thresholds are unbounded scores. Tune the
  threshold per dataset *and* per metric; never reuse a magic number
  across them.

## Take-Home Summary

- A **knowledge base** is the triple `(S, I, Q)`: a set of
  typed records, one or more indices over them, and a family
  of query operators that return ranked matches.
- One `DataModel` class → one table. **The first declared
  field is the primary key.** `update` is an upsert.
- Five search operators (pick by what evidence you have):
  **`fulltext_search`** (BM25, lexical),
  **`similarity_search`** (vector, semantic, needs an
  embedding model), **`regex_search`** (RE2 patterns, linear-time
  evaluation), **`hybrid_fts_search`** (vector + BM25 fused with
  Reciprocal Rank Fusion — the standard default for production RAG),
  and **`hybrid_regex_search`** (vector + regex, same fusion). Plus
  a raw **`query()`** escape hatch — see the *Raw SQL* section below.
- The **`RetrieveKnowledge`** module drops retrieval into a
  `Program` directly; combined with a downstream `Generator`,
  that is **RAG** (Retrieval-Augmented Generation).
- File-on-disk ingestion has two paths: **`kb.from_csv`** /
  **`from_parquet`** / **`from_json`** / **`from_jsonl`** for
  native DuckDB bulk-load when the file's columns match the data
  model 1:1 (~25× faster than the streaming path), or
  **`kb.update(<…>Dataset(...))`** for memory-bounded streaming
  with Jinja-template transformation between source and stored
  shape. The fast path doesn't compute embeddings; for vector
  search either supply them precomputed via the streaming path or
  run a follow-up populating step.
- Externalizing state is what beats the LM's two hard limits:
  the parameter cutoff (frozen weights) and the context
  bound (finite window). Retrieval becomes a deterministic,
  auditable preprocessing step you can debug.

## API References

- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/)
- [RetrieveKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/RetrieveKnowledge%20module/)
- [UpdateKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/UpdateKnowledge%20module/)
- [EmbedKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/EmbedKnowledge%20module/)
"""

# --8<-- [start:source]
import asyncio
import os

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Document(synalinks.DataModel):
    """A document in the knowledge base."""

    id: str = synalinks.Field(description="Unique document ID")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Document content")


class Query(synalinks.DataModel):
    """User query."""

    query: str = synalinks.Field(description="User question")


class Answer(synalinks.DataModel):
    """Answer based on retrieved context."""

    answer: str = synalinks.Field(description="Answer based on the context provided")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_6_knowledge_base",
    # )

    # -------------------------------------------------------------------------
    # Create Knowledge Base
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Knowledge Base with Full-Text Search")
    print("=" * 60)

    db_path = "guides/guides_knowledge.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    kb = synalinks.KnowledgeBase(
        uri=f"duckdb://{db_path}",
        data_models=[Document],
        embedding_model=None,
        metric="cosine",
        wipe_on_start=True,
        name="guides_kb",
    )

    # -------------------------------------------------------------------------
    # Store Documents
    # -------------------------------------------------------------------------
    print("\nStoring documents...")

    documents = [
        Document(
            id="doc1",
            title="Introduction to Python",
            content="Python is a high-level programming language.",
        ),
        Document(
            id="doc2",
            title="Machine Learning Basics",
            content="Machine learning is a subset of AI that enables systems to learn.",
        ),
        Document(
            id="doc3",
            title="Deep Learning",
            content="Deep learning uses neural networks with many layers.",
        ),
        Document(
            id="doc4",
            title="Natural Language Processing",
            content="NLP enables computers to understand and process human language.",
        ),
    ]

    for doc in documents:
        await kb.update(doc.to_json_data_model())
        print(f"  Stored: {doc.title}")

    # -------------------------------------------------------------------------
    # Full-Text Search
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Full-Text Search")
    print("=" * 60)

    results = await kb.fulltext_search(
        "programming language",
        table_name="Document",
        k=10,
        threshold=None,
    )

    print("\nSearch: 'programming language'")
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r['title']}: {r['content'][:50]}...")

    # -------------------------------------------------------------------------
    # Get by ID
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Get by ID")
    print("=" * 60)

    result = await kb.get(
        "doc2",
        table_name="Document",
    )

    print("\nGet doc2:")
    print(f"  Title: {result['title']}")
    print(f"  Content: {result['content']}")

    # -------------------------------------------------------------------------
    # Get All Records
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: Get All Records")
    print("=" * 60)

    all_docs = await kb.getall(
        table_name="Document",
        limit=50,
        offset=0,
    )

    print(f"\nAll documents ({len(all_docs)}):")
    for doc in all_docs:
        print(f"  - {doc['id']}: {doc['title']}")

    # -------------------------------------------------------------------------
    # RAG Pipeline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 5: RAG Pipeline")
    print("=" * 60)

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    inputs = synalinks.Input(data_model=Query)

    retrieved = await synalinks.RetrieveKnowledge(
        knowledge_base=kb,
        language_model=lm,
        search_type="fulltext",
        k=2,
        return_inputs=True,
    )(inputs)

    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(retrieved)

    rag_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="rag_pipeline",
    )

    result = await rag_program(Query(query="What is Python?"))
    print("\nRAG Query: What is Python?")
    print(f"Answer: {result['answer']}")

    result = await rag_program(Query(query="Tell me about neural networks"))
    print("\nRAG Query: Tell me about neural networks")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Raw SQL Query
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 6: Raw SQL Query")
    print("=" * 60)

    results = await kb.sql(
        "SELECT id, title FROM Document WHERE title LIKE ?",
        params=["%Learning%"],
    )

    print("\nSQL: SELECT WHERE title LIKE '%Learning%'")
    for r in results:
        print(f"  - {r['id']}: {r['title']}")

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
