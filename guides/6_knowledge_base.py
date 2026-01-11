"""
# Knowledge Base

A **Knowledge Base** in Synalinks is a structured storage system that enables
your LM applications to retrieve and reason over external data. Unlike simple
prompt injection, a Knowledge Base provides semantic search capabilities,
automatic chunking, and efficient retrieval - the foundation for building
Retrieval-Augmented Generation (RAG) systems.

## Why Knowledge Bases Matter

Language models have a knowledge cutoff and limited context windows. A
Knowledge Base solves both problems:

```mermaid
graph LR
    subgraph Without Knowledge Base
        A[Query] --> B[LLM]
        B --> C[Hallucination Risk]
    end
    subgraph With Knowledge Base
        D[Query] --> E[Retrieve Relevant Docs]
        E --> F[LLM + Context]
        F --> G[Grounded Answer]
    end
```

Knowledge Bases provide:

1. **Grounded Responses**: Answers based on actual data, not hallucinations
2. **Unlimited Knowledge**: Store documents beyond context limits
3. **Up-to-Date Information**: Add new data without retraining
4. **Source Attribution**: Track where answers come from

## Architecture

Synalinks Knowledge Base is built on DuckDB, providing:

```mermaid
graph TD
    A[DataModels] --> B[KnowledgeBase]
    B --> C[DuckDB Storage]
    B --> D[Full-Text Index]
    B --> E[Vector Index]
    F[Search Query] --> G{Search Type}
    G -->|fulltext| D
    G -->|similarity| E
    G -->|hybrid| H[Combine Both]
    D --> I[Results]
    E --> I
    H --> I
```

## Creating a Knowledge Base

Define DataModels for your documents, then create the Knowledge Base:

```python
import synalinks

class Document(synalinks.DataModel):
    \"\"\"A document in the knowledge base.\"\"\"
    id: str = synalinks.Field(description="Unique document ID")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Document content")

# Create the knowledge base
kb = synalinks.KnowledgeBase(
    uri="duckdb://my_database.db",    # Storage location
    data_models=[Document],            # What types to store
    embedding_model=embedding_model,   # For vector search (optional)
    metric="cosine",                   # Similarity metric
    wipe_on_start=False,               # Preserve existing data
)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `uri` | Database connection string (e.g., `duckdb://path.db`) |
| `data_models` | List of DataModel classes to store |
| `embedding_model` | EmbeddingModel for vector search (optional) |
| `metric` | Similarity metric: `cosine`, `l2`, or `ip` |
| `wipe_on_start` | Clear database on initialization |

## Search Methods

### Full-Text Search (BM25)

Uses the BM25 algorithm for traditional keyword-based search:

```python
results = await kb.fulltext_search(
    "machine learning neural networks",
    data_models=[Document], # If None search in all tables
    k=10,           # Number of results
    threshold=None, # Minimum score (optional)
)
```

Best for:

- Exact keyword matching
- When users search with specific terms
- Quick, lightweight search

### Similarity Search (Vector)

Uses embedding vectors for semantic search:

```python
results = await kb.similarity_search(
    "how do computers learn",  # Semantically matches "machine learning"
    data_models=[Document], # If None search in all tables
    k=10,
    threshold=0.7,  # Minimum similarity score
)
```

Best for:

- Semantic meaning matching
- Natural language queries
- Finding conceptually related content

### Hybrid Search

Combines both methods for best results:

```python
results = await kb.hybrid_search(
    "machine learning basics",
    data_models=[Document],
    k=10,
    bm25_weight=0.5,    # Weight for BM25 scores
    vector_weight=0.5,  # Weight for vector scores
)
```

Best for:

- Production RAG systems
- When you need both exact and semantic matching
- Complex queries that benefit from both approaches

## CRUD Operations

### Create/Update

The `update` method performs upsert (insert or update). The first field in
your DataModel is used as the primary key:

```python
doc = Document(
    id="doc1",
    title="Introduction to AI",
    content="Artificial intelligence is...",
)

await kb.update(doc.to_json_data_model())
```

### Read by ID

```python
result = await kb.get(
    "doc1",  # Primary key value
    data_models=[Document],
)
```

### List All

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

### Raw SQL

For complex queries, use raw SQL:

```python
results = await kb.query(
    "SELECT id, title FROM Document WHERE title LIKE ?",
    params=["%Learning%"],
)
```

## Knowledge Modules

Synalinks provides modules for integrating Knowledge Bases into programs:

### RetrieveKnowledge

Retrieves relevant documents using LM-generated search queries:

```mermaid
graph LR
    A[Input] --> B[Generate Query]
    B --> C[Search KB]
    C --> D[Context + Input]
```

```python
retrieved = await synalinks.RetrieveKnowledge(
    knowledge_base=kb,
    language_model=lm,
    search_type="hybrid",  # fulltext, similarity, or hybrid
    k=10,
    return_inputs=True,    # Include original input in output
)(inputs)
```

### UpdateKnowledge

Stores DataModels in the Knowledge Base:

```python
stored = await synalinks.UpdateKnowledge(
    knowledge_base=kb,
)(extracted_data)
```

### EmbedKnowledge

Generates embeddings for DataModels:

```python
embedded = await synalinks.EmbedKnowledge(
    embedding_model=embedding_model,
    in_mask=["content"],  # Which fields to embed
)(inputs)
```

## Building a RAG Pipeline

A complete RAG system combines retrieval with generation:

```mermaid
graph LR
    A[Query] --> B[RetrieveKnowledge]
    B --> C[Context + Query]
    C --> D[Generator]
    D --> E[Grounded Answer]
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

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Assume kb is already populated
    kb = synalinks.KnowledgeBase(
        uri="duckdb://knowledge.db",
        data_models=[Document],
    )

    inputs = synalinks.Input(data_model=Query)

    # Retrieve relevant documents
    retrieved = await synalinks.RetrieveKnowledge(
        knowledge_base=kb,
        language_model=lm,
        search_type="fulltext",
        k=5,
        return_inputs=True,
    )(inputs)

    # Generate answer using retrieved context
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

## Key Takeaways

- **DuckDB Backend**: Fast, embedded database with full-text and vector search
  capabilities. No external services required.

- **Three Search Types**: Full-text (BM25) for keywords, similarity for
  semantics, hybrid for best of both.

- **DataModel as Schema**: Your DataModels define the structure of stored
  documents. The first field is the primary key.

- **RetrieveKnowledge Module**: Automates query generation and retrieval for
  RAG pipelines. Combines seamlessly with Generator.

- **Upsert Semantics**: The `update` method inserts new records or updates
  existing ones based on the primary key.

- **Raw SQL Access**: For complex queries, you can use raw SQL directly.

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

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_6_knowledge_base",
    )

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

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

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
