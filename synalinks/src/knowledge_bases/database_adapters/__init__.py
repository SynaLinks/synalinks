from synalinks.src.knowledge_bases.database_adapters.database_adapter import (
    DatabaseAdapter,
)
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import DuckDBAdapter
from synalinks.src.knowledge_bases.database_adapters.lancedb_adapter import LanceDBAdapter


def get(uri):
    """Resolve a SQL/vector-store URI to its adapter class.

    ``lancedb://...`` selects the vector-native LanceDB adapter; everything
    else (including ``None`` and ``duckdb://...``) defaults to DuckDB.
    """
    if uri and uri.startswith("lancedb"):
        return LanceDBAdapter
    return DuckDBAdapter
