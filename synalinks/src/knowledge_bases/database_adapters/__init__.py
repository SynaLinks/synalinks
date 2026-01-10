from synalinks.src.knowledge_bases.database_adapters.database_adapter import (
    DatabaseAdapter,
)
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import DuckDBAdapter


def get(uri):
    if not uri:
        return DuckDBAdapter
    if uri.startswith("duckdb"):
        return DuckDBAdapter
    else:
        return DuckDBAdapter
