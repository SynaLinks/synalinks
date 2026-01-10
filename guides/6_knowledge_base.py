"""
# Guide 6: Knowledge Base

Synalinks provides a powerful knowledge base built on DuckDB for storing
and retrieving structured data with full-text and vector search capabilities.

## Creating a Knowledge Base

```python
class Document(synalinks.DataModel):
    id: str = synalinks.Field(description="Document ID")
    title: str = synalinks.Field(description="Title")
    content: str = synalinks.Field(description="Content")

kb = synalinks.KnowledgeBase(
    uri="duckdb://my_database.db",
    data_models=[Document],
    embedding_model=embedding_model,
    metric="cosine",
    wipe_on_start=False,
)
```

## Search Methods

| Method | Description |
|--------|-------------|
| `fulltext_search` | BM25-based text search |
| `similarity_search` | Vector similarity search |
| `hybrid_search` | Combines full-text and vector search |

## Knowledge Modules

### UpdateKnowledge

Store data models in the knowledge base.

```python
stored = await synalinks.UpdateKnowledge(
    knowledge_base=knowledge_base,
)(extracted_data)
```

### RetrieveKnowledge

Retrieve relevant records using LM-generated search queries.

```python
results = await synalinks.RetrieveKnowledge(
    knowledge_base=knowledge_base,
    language_model=language_model,
    search_type="hybrid",
    k=10,
    return_inputs=True,
)(query_input)
```

### EmbedKnowledge

Generate embeddings for data models to enable similarity search.

```python
embedded = await synalinks.EmbedKnowledge(
    embedding_model=embedding_model,
    in_mask=["content"],
)(inputs)
```

## Running the Example

```bash
uv run python guides/6_knowledge_base.py
```
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
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
# STEP 2: Demonstrate Knowledge Base Features
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_6_knowledge_base",
    )

    # -------------------------------------------------------------------------
    # 2.1: Create Knowledge Base (Full-Text Only)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Knowledge Base with Full-Text Search")
    print("=" * 60)

    # Remove old database if exists
    db_path = "guides_knowledge.db"
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
    # 2.2: Store Documents
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
    # 2.3: Full-Text Search
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Full-Text Search")
    print("=" * 60)

    results = await kb.fulltext_search(
        "programming language",
        data_models=[Document.to_symbolic_data_model()],
        k=10,
        threshold=None,
    )

    print("\nSearch: 'programming language'")
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  - {r['title']}: {r['content'][:50]}...")

    # -------------------------------------------------------------------------
    # 2.4: Get by ID
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Get by ID")
    print("=" * 60)

    result = await kb.get(
        "doc2",
        data_models=[Document.to_symbolic_data_model()],
    )

    print("\nGet doc2:")
    print(f"  Title: {result['title']}")
    print(f"  Content: {result['content']}")

    # -------------------------------------------------------------------------
    # 2.5: Get All Records
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: Get All Records")
    print("=" * 60)

    all_docs = await kb.getall(
        Document.to_symbolic_data_model(),
        limit=50,
        offset=0,
    )

    print(f"\nAll documents ({len(all_docs)}):")
    for doc in all_docs:
        print(f"  - {doc['id']}: {doc['title']}")

    # -------------------------------------------------------------------------
    # 2.6: RAG Pipeline with RetrieveKnowledge
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 5: RAG Pipeline")
    print("=" * 60)

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    inputs = synalinks.Input(data_model=Query)

    # Retrieve relevant documents
    retrieved = await synalinks.RetrieveKnowledge(
        knowledge_base=kb,
        language_model=lm,
        search_type="fulltext",
        k=2,
        return_inputs=True,
    )(inputs)

    # Generate answer based on context
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
    # 2.7: Raw SQL Query
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

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. DUCKDB: Fast embedded database for knowledge storage
2. FULL-TEXT SEARCH: BM25-based search on text fields
3. HYBRID SEARCH: Combine full-text and vector search
4. RETRIEVEKNOWLEDGE: Module for RAG pipelines
5. RAW SQL: Execute custom queries when needed
6. FIRST FIELD: Used as primary key for upserts
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
