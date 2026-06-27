# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from synalinks.src.modules.embedding_models import get as _get_em


class DatabaseAdapter:
    """Base class for database adapters.

    DatabaseAdapter provides a unified interface for storing and retrieving
    structured data with optional embedding-based similarity search capabilities.

    Subclasses must implement the abstract methods to provide concrete
    database functionality.
    """

    def __init__(
        self,
        uri=None,
        embedding_model=None,
        data_models=None,
        metric="cosine",
        wipe_on_start=False,
        name=None,
        **kwargs,
    ):
        """Initialize the database adapter.

        Args:
            uri (str): The database connection URI or path.
            embedding_model (EmbeddingModel): Optional embedding model for
                vector similarity search.
            data_models (list): Optional list of SymbolicDataModel or DataModel
                classes to create tables for.
            metric (str): Distance metric for vector search. Options depend on
                the specific adapter implementation.
            wipe_on_start (bool): Whether to clear the database on initialization.
            name (str): Optional name for the adapter instance.
        """
        self.uri = uri
        self.embedding_model = _get_em(embedding_model)
        self.data_models = data_models or []
        self.metric = metric
        self.name = name

        if wipe_on_start:
            self.wipe_database()

    def wipe_database(self):
        """Clear all data from the database.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `wipe_database()` method"
        )

    def get_symbolic_data_models(self):
        """Retrieve all data models from the database schema.

        Returns:
            List[SymbolicDataModel]: List of symbolic data models representing
                the database schema.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            "`get_symbolic_data_models()` method"
        )

    async def update(
        self,
        data_model_or_data_models: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert or update records in the database.

        Args:
            data_model_or_data_models: A single JsonDataModel or a list of
                JsonDataModels to insert or update.

        Returns:
            The primary key value(s) of the inserted/updated records.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `update()` method"
        )

    async def from_csv(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Bulk-load a CSV file directly into the knowledge base.

        Skips the per-row Python pipeline for native database
        ingestion. The target table is created from the file's
        columns, so callers don't pre-declare a ``DataModel``.
        Adapters that don't support a native fast path may leave this
        unimplemented.

        Args:
            path: Path to the CSV file.
            table_name: Target table name (also the resulting schema
                title). Defaults to the file's stem when omitted.
            table_description: Optional natural-language description
                for the resulting schema.
            **kwargs: Adapter-specific options (e.g. delimiter,
                encoding, header).

        Returns:
            The ``SymbolicDataModel`` for the loaded table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `from_csv()` method"
        )

    async def from_parquet(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Bulk-load a Parquet file directly into the knowledge base.

        Skips the per-row Python pipeline for native database
        ingestion. The target table is created from the file's
        columns. Adapters without a native fast path may leave this
        unimplemented.

        Args:
            path: Path to the Parquet file.
            table_name: Target table name. Defaults to the file's stem.
            table_description: Optional schema description.
            **kwargs: Adapter-specific options.

        Returns:
            The ``SymbolicDataModel`` for the loaded table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `from_parquet()` method"
        )

    async def from_json(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Bulk-load a JSON file (top-level array of objects).

        Skips the per-row Python pipeline for native database
        ingestion.

        Args:
            path: Path to the JSON file.
            table_name: Target table name. Defaults to the file's stem.
            table_description: Optional schema description.
            **kwargs: Adapter-specific options.

        Returns:
            The ``SymbolicDataModel`` for the loaded table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `from_json()` method"
        )

    async def from_jsonl(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Bulk-load a JSON Lines (NDJSON) file.

        Skips the per-row Python pipeline for native database
        ingestion.

        Args:
            path: Path to the JSONL file.
            table_name: Target table name. Defaults to the file's stem.
            table_description: Optional schema description.
            **kwargs: Adapter-specific options.

        Returns:
            The ``SymbolicDataModel`` for the loaded table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `from_jsonl()` method"
        )

    async def rename(
        self,
        source: Any,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> Any:
        """Rename a table and/or update its description.

        Args:
            source: ``SymbolicDataModel`` or table-name string for
                the table to rename.
            table_name: New table name. Optional — pass to ``ALTER``
                the table.
            table_description: New schema description. Optional.

        Returns:
            The updated ``SymbolicDataModel``.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `rename()` method"
        )

    async def get(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
    ) -> Union[Optional[Any], List[Optional[Any]]]:
        """Retrieve records by primary key from a single table.

        Args:
            id_or_ids: A single primary key value, or a list of values.
            table_name: Target table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `get()` method"
        )

    async def getall(
        self,
        *,
        table_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Any]:
        """Retrieve all records from a single table with pagination.

        Args:
            table_name: Target table.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `getall()` method"
        )

    async def delete(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
    ) -> int:
        """Delete records by primary key from a single table.

        Args:
            id_or_ids: Primary key value, or a list of values.
            table_name: Target table.

        Returns:
            The number of rows actually deleted.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `delete()` method"
        )

    async def drop_table(self, table_name: str) -> bool:
        """Drop a table from the database.

        Args:
            table_name: Target table.

        Returns:
            ``True`` if a table was dropped, ``False`` if it did not exist.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `drop_table()` method"
        )

    async def sql(
        self,
        sql: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        **kwargs,
    ):
        """Execute a raw SQL query against the database.

        Counterpart of `GraphDatabaseAdapter.cypher` — the
        method is named after the query language so a KnowledgeBase
        that carries both a SQL store and a graph store has a clear
        per-language entry point (``kb.sql(...)`` vs ``kb.cypher(...)``).

        Args:
            sql: The SQL string.
            params: Optional parameters for parameterized queries.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
            **kwargs: Additional adapter-specific options.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `sql()` method"
        )

    async def similarity_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        table_name: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        k: int = 10,
        threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """Perform vector similarity search using embeddings.

        Either ``text_or_texts`` (embedded with the adapter's embedding
        model) or ``vector_or_vectors`` (a pre-computed query vector or
        list of vectors) selects what to search for. When vectors are
        supplied no embedding model is required.

        Args:
            text_or_texts: Query text or list of query texts. Ignored
                when ``vector_or_vectors`` is supplied.
            table_name: Target table.
            vector_or_vectors: Pre-computed query vector or list of
                vectors to search with directly.
            k: Maximum number of results.
            threshold: Optional vector distance threshold.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `similarity_search()` method"
        )

    async def fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        table_name: str,
        k: int = 10,
        threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """Perform full-text search on text fields.

        Args:
            text_or_texts: Query text or list of query texts.
            table_name: Target table.
            k: Maximum number of results.
            threshold: Optional minimum relevance on the normalized
                ``[0, 1]`` scale (the result set is min-max scaled so
                the best hit is ``1.0`` and the worst ``0.0``). Reported
                ``score`` is on the same scale, so it stays comparable
                across adapters with different raw BM25 ranges.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `fulltext_search()` method"
        )

    async def hybrid_fts_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        keywords: Optional[Union[str, List[str]]] = None,
        table_name: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """Hybrid retrieval combining vector similarity and BM25 fulltext.

        Uses Reciprocal Rank Fusion (RRF) to merge the two rankings.
        The sibling method ``hybrid_regex_search`` pairs vector with
        regex instead; the two cover the orthogonal "semantics" vs
        "exact textual shape" signals.

        Args:
            text_or_texts: Query text or list of query texts for the
                vector branch. Ignored when ``vector_or_vectors`` is
                supplied.
            table_name: Target table.
            keywords: Query text or list of query texts for the BM25
                branch. Aligns by position with the vector-branch
                queries; when omitted, the text is reused for both.
            vector_or_vectors: Pre-computed query vector(s) for the
                vector branch, used directly instead of embedding text.
            k: Maximum number of results.
            similarity_threshold: Optional vector-distance threshold.
            fulltext_threshold: Optional minimum fulltext relevance on
                the normalized ``[0, 1]`` scale.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `hybrid_fts_search()` method"
        )

    async def regex_search(
        self,
        pattern: str,
        *,
        table_name: str,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        k: int = 10,
        output_format: str = "json",
    ):
        """Find rows whose string fields match a regular expression.

        Args:
            pattern: The regex pattern.
            table_name: Target table.
            fields: Field names to match against. Defaults to every
                string-typed field on the schema.
            case_sensitive: When ``False``, matches case-insensitively.
            k: Maximum number of results.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `regex_search()` method"
        )

    async def hybrid_regex_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        pattern_or_patterns: Union[str, List[str], None] = None,
        table_name: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        output_format: str = "json",
    ):
        """Hybrid retrieval combining vector similarity and regex match.

        Sibling of `hybrid_fts_search`. Uses Reciprocal Rank
        Fusion (RRF) to merge the two rankings. The vector half
        captures semantics; the regex half captures exact textual
        shape (identifiers, codes, patterns).

        Args:
            text_or_texts: Query text or list of query texts for the
                vector half. Ignored when ``vector_or_vectors`` is
                supplied.
            pattern_or_patterns: Regex pattern or list of patterns for
                the regex half. ``None`` skips the regex half.
            table_name: Target table.
            vector_or_vectors: Pre-computed query vector(s) for the
                vector half, used directly instead of embedding text.
            k: Maximum number of results.
            similarity_threshold: Optional vector-distance threshold.
            fields: Forwarded to `regex_search`.
            case_sensitive: Forwarded to `regex_search`.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`hybrid_regex_search()` method"
        )

    async def hybrid_search(self, *args, **kwargs):
        """Deprecated alias of `hybrid_fts_search`.

        Kept so existing call sites keep working after the rename to
        ``hybrid_fts_search`` (which is symmetric with the newer
        ``hybrid_regex_search``). Prefer the new name in new code.
        """
        return await self.hybrid_fts_search(*args, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__} uri={self.uri}>"
