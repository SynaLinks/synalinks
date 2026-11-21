# Knowledge Base API

A `KnowledgeBase` is a stateful container that lets Synalinks programs persist and retrieve structured knowledge through a pluggable storage backend. It pairs a JSON schema (derived from your `DataModel`s) with a database adapter that handles the underlying I/O: SQL stores via the [Database Adapters](#database-adapters) (DuckDB, LanceDB) and graph stores via the [Graph Database Adapters](#graph-database-adapters) (LadybugDB, Relate). The URI picks the adapter.

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
        data_models=[Document],
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
- [LanceDB Adapter](Database Adapters/LanceDB Adapter.md)

---

### Graph Database Adapters

- [Ladybug Adapter](Graph Database Adapters/Ladybug Adapter.md)
- [Relate Adapter](Graph Database Adapters/Relate Adapter.md)

---

## Choosing a store

A `KnowledgeBase` has a SQL side (`uri=`) and a graph side (`graph_uri=`); each URI selects its adapter. All adapters of a side expose the same methods and return the same results, so switching is a one-line change.

| Side | URI | Adapter | Choose it when |
|---|---|---|---|
| SQL | `duckdb://...` (default) | [DuckDB](Database Adapters/DuckDB Adapter.md) | General purpose: full-text, vector, regex and hybrid search, raw SQL, encryption at rest, in-memory mode for tests. |
| SQL | `lancedb://...` | [LanceDB](Database Adapters/LanceDB Adapter.md) | Vector-heavy workloads: native vector search over columnar Lance datasets. SQL still works (DuckDB reads the Lance tables); no in-memory mode. |
| Graph | `ladybug://...` (default) | [LadybugDB](Graph Database Adapters/Ladybug Adapter.md) | General purpose property graph: typed node and edge tables, Cypher, vector and full-text search, graph algorithms (PageRank, communities) for GraphRAG. |
| Graph | `relate://...` | [Relate by Neo4j](Graph Database Adapters/Relate Adapter.md) | You want Neo4j's Cypher dialect and Lucene full-text, embedded with no server. Schema-free labels; graph algorithms run through `networkx`. Needs `pip install 'synalinks[relate]'`. |

```python
# Same code, other stores: only the URIs change
knowledge_base = synalinks.KnowledgeBase(
    uri="lancedb://./documents.lance",
    graph_uri="relate://./graph.relate",
    data_models=[Document],
    entity_models=[Person, City],
    relation_models=[LivesIn],
    embedding_model=embedding_model,
)
```
