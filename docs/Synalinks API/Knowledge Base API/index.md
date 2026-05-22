# Knowledge Base API

A `KnowledgeBase` is a stateful container that lets Synalinks programs persist and retrieve structured knowledge through a pluggable storage backend. It pairs a JSON schema (derived from your `DataModel`s) with a database adapter that handles the underlying I/O — relational stores via [Database Adapters](Database Adapters/DuckDB Adapter.md) and graph stores via [Graph Database Adapters](Graph Database Adapters/Ladybug Adapter.md).

Knowledge bases are the storage layer used by knowledge-aware modules such as `EmbedKnowledge`, `UpdateKnowledge`, and `RetrieveKnowledge`, and they back higher-level agents like `VectorRAGAgent` and `SQLAgent`.

```python
import synalinks
import asyncio

class Document(synalinks.DataModel):
    title: str = synalinks.Field(
        description="The document title",
    )
    content: str = synalinks.Field(
        description="The document content",
    )

async def main():
    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large",
    )

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://:memory:",
        data_model=Document,
        embedding_model=embedding_model,
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Knowledge Base API overview

- [Knowledge Base](Knowledge Base.md)

---

### Database Adapters

- [DuckDB Adapter](Database Adapters/DuckDB Adapter.md)

---

### Graph Database Adapters

- [Ladybug Adapter](Graph Database Adapters/Ladybug Adapter.md)
