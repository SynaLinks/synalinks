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
typed records you have seen since [Guide 2](Data%20Models.md)). The indices are provided by
**DuckDB**, an embedded SQL engine, plus a couple of DuckDB extensions
for text and vector search. `Q` is exposed as three methods on a
`KnowledgeBase` object: `fulltext_search`, `similarity_search`, and
`hybrid_search`.

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

## The Three Search Operators

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
    data_models=[Document],  # None means: search every registered table
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
    data_models=[Document],
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

### 3. Hybrid search

Hybrid search runs both retrievers — BM25 *and* vector — and **fuses**
their rankings into a single combined ranking. Synalinks uses
**Reciprocal Rank Fusion (RRF)**: each candidate's final score is a
weighted sum of `1 / (k + rank)` from each retriever, where `rank` is
its position in that retriever's list. The intuition: being near the
top of *either* list is strong evidence, and RRF rewards documents
that show up well in multiple rankings — without requiring the
underlying scores to be on comparable scales.

```python
results = await kb.hybrid_search(
    "machine learning basics",
    data_models=[Document],
    k=10,
    bm25_weight=0.5,
    vector_weight=0.5,
)
```

Hybrid is the standard default for production RAG. BM25 anchors
precise terminology (proper names, product codes, identifiers) where
literal matching is essential, while the vector path recovers
paraphrases and synonyms.

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
    data_models=[Document],
)
```

### Enumerate

```python
all_docs = await kb.getall(
    Document,
    limit=100,
    offset=0,
)
```

### Delete

```python
await kb.delete(
    "doc1",
    data_models=[Document],
)
```

### Raw SQL escape hatch

```python
results = await kb.query(
    "SELECT id, title FROM Document WHERE title LIKE ?",
    params=["%Learning%"],
)
```

Always use **parameterized queries**: the `?` placeholder is filled in
by the database *after* the SQL has been parsed, so user input can
never be mistaken for SQL syntax. This is how you avoid SQL injection
attacks — a class of security vulnerability you should learn to spot
even if you never become a security engineer.

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
    search_type="hybrid",   # fulltext | similarity | hybrid
    k=10,
    return_inputs=True,
)(inputs)
```

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
  vector search.
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
- Three search operators:
  **`fulltext_search`** (BM25, lexical),
  **`similarity_search`** (vector, semantic, needs an
  embedding model), and
  **`hybrid_search`** (Reciprocal-Rank-Fusion of both — the
  standard default for production RAG).
- The **`RetrieveKnowledge`** module drops retrieval into a
  `Program` directly; combined with a downstream `Generator`,
  that is **RAG** (Retrieval-Augmented Generation).
- Externalizing state is what beats the LM's two hard limits:
  the parameter cutoff (frozen weights) and the context
  bound (finite window). Retrieval becomes a deterministic,
  auditable preprocessing step you can debug.

## API References

- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/)
- [RetrieveKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Base%20Modules/RetrieveKnowledge%20module/)
- [UpdateKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Base%20Modules/UpdateKnowledge%20module/)
- [EmbedKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Base%20Modules/EmbedKnowledge%20module/)
"""

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
        data_models=[Document],
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
        data_models=[Document],
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
        Document,
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

    results = await kb.query(
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
