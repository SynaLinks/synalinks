# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.datasets.dataset import Dataset
from synalinks.src.knowledge_bases import database_adapters
from synalinks.src.modules.embedding_models import get as _get_em
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name
from synalinks.src.utils.progbar import Progbar


@synalinks_export("synalinks.KnowledgeBase")
class KnowledgeBase(SynalinksSaveable):
    """A knowledge base for storing and retrieving structured data.

    The KnowledgeBase provides a unified interface for storing structured data
    with support for full-text search and optional vector similarity search.
    It uses DuckDB as the underlying storage engine.

    ### Basic Usage

    ```python
    import synalinks

    class Document(synalinks.DataModel):
        id: str
        title: str
        content: str

    # Create a knowledge base without embeddings (full-text search only)
    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://my_database.db",
        data_models=[Document],
    )

    # Store a document
    doc = Document(id="1", title="Hello", content="Hello World!")
    await knowledge_base.update(doc.to_json_data_model())

    # Retrieve by ID
    result = await knowledge_base.get("1", table_name="Document")

    # Full-text search
    results = await knowledge_base.fulltext_search("Hello", k=10)
    ```

    ### With Vector Similarity Search

    ```python
    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large"
    )

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./my_database.db",
        data_models=[Document],
        embedding_model=embedding_model,
        metric="cosine",
    )

    # Hybrid search (combines BM25 fulltext + vector similarity, fused by RRF)
    results = await knowledge_base.hybrid_fts_search("semantic query", k=10)
    ```

    ### Retrieving Table Definitions

    ```python
    # Get all symbolic data models (table definitions) from the database
    symbolic_models = knowledge_base.get_symbolic_data_models()

    for model in symbolic_models:
        print(model.get_schema())
        # {'title': 'Document', 'type': 'object', 'properties': {...}, ...}
    ```

    Args:
        uri (str): The database connection URI. Use "duckdb://path/to/db.db"
            for DuckDB. If not provided, uses an in-memory database.
        data_models (list): Optional list of DataModel or SymbolicDataModel
            classes to create tables for.
        embedding_model (EmbeddingModel): Optional embedding model for
            vector similarity search.
        metric (str): The distance metric for vector search.
            Options: "cosine", "l2seq", "ip" (default: "cosine").
        wipe_on_start (bool): Whether to clear the database on initialization
            (default: False).
        name (str): Optional name for the knowledge base (used for serialization).
    """

    def __init__(
        self,
        *,
        uri=None,
        data_models=None,
        embedding_model=None,
        metric="cosine",
        wipe_on_start=False,
        name=None,
        encryption_key=None,
        **kwargs,
    ):
        self.adapter = database_adapters.get(uri)(
            uri=uri,
            data_models=data_models,
            embedding_model=embedding_model,
            metric=metric,
            wipe_on_start=wipe_on_start,
            name=name,
            encryption_key=encryption_key,
            **kwargs,
        )
        self.uri = uri
        self.data_models = data_models or []
        self.embedding_model = _get_em(embedding_model)
        self.metric = metric
        self.wipe_on_start = wipe_on_start
        if not name:
            self.name = auto_name("knowledge_base")
        else:
            self.name = name
        # `encryption_key` is deliberately NOT stored on `self` — it
        # lives only inside the adapter, and only as long as the
        # adapter does. This keeps the secret out of `get_config()`,
        # off-screen during repr/print, and unreferenced by any
        # serialization path. Callers must re-supply the key when
        # constructing a new KnowledgeBase against an encrypted file.

    async def update(
        self,
        data_model_or_data_models: Union[Any, List[Any], Dataset],
        *,
        verbose="auto",
    ) -> Union[Any, List[Any]]:
        """Insert or update records in the knowledge base.

        Args:
            data_model_or_data_models (JsonDataModel | List[JsonDataModel] | Dataset):
                A single ``JsonDataModel``, a list of ``JsonDataModel`` /
                ``DataModel`` instances, or a synalinks ``Dataset``.
                The ``Dataset`` form streams the source batch-by-batch
                (one ``adapter.update`` call per yielded batch) so memory
                stays bounded for large CSV / Parquet / HuggingFace
                sources. The dataset must be inputs-only — no
                ``output_template`` — because the knowledge base stores
                records, not ``(input, target)`` pairs; pass a
                labeled dataset and you'll get a ``ValueError``.

                Uses the first field as the primary key for upserts.
            verbose (int | str): ``"auto"``, ``0``, ``1``, or ``2``.
                Verbosity for the ``Dataset`` path; matches the
                trainer's ``fit()`` semantics. ``"auto"`` (default)
                resolves to ``1`` when a ``Dataset`` is passed (a
                per-batch progress bar — same widget ``fit()`` uses,
                with ETA when ``len(dataset)`` is known) and is a
                no-op for the scalar / list forms, which finish in a
                single adapter call.

        Returns:
            The primary key value(s) of the inserted/updated records.
            Scalar in / scalar out; list in / list out; ``Dataset`` in /
            flat list of every batch's ids concatenated.
        """
        if isinstance(data_model_or_data_models, Dataset):
            return await self._update_from_dataset(
                data_model_or_data_models, verbose=verbose
            )
        return await self.adapter.update(data_model_or_data_models)

    async def _update_from_dataset(
        self, dataset: Dataset, *, verbose="auto"
    ) -> List[Any]:
        """Stream a ``Dataset`` into the adapter one batch at a time.

        Each batch yielded by the dataset is converted to a list of
        DataModel / JsonDataModel instances and handed to
        ``adapter.update``. The returned ids from every batch are
        accumulated into one flat list — same order as the dataset
        produced them.

        Inputs-only is enforced: a dataset configured with an
        ``output_template`` represents ``(input, target)`` training
        data, which isn't what the knowledge base stores. The check is
        the dataset's public ``output_template`` attribute, not the
        per-batch tuple length — so the rejection happens upfront,
        before any rows are consumed.
        """
        if dataset.output_template is not None:
            raise ValueError(
                "KnowledgeBase.update accepts only inputs-only datasets "
                "(no `output_template`). The knowledge base stores "
                "records, not (input, target) pairs."
            )

        # "auto" → 1 in the Dataset branch (we know there's iteration to
        # display). Outside this branch verbose is dead anyway.
        if verbose == "auto":
            verbose = 1

        progbar = None
        if verbose:
            try:
                target = len(dataset)
            except (TypeError, NotImplementedError):
                target = None
            progbar = Progbar(target=target, verbose=verbose, unit_name="batch")

        ids: List[Any] = []
        step = 0
        for batch in dataset:
            x = batch[0]
            if len(x) == 0:
                continue
            batch_ids = await self.adapter.update(list(x))
            if isinstance(batch_ids, list):
                ids.extend(batch_ids)
            else:
                ids.append(batch_ids)
            step += 1
            if progbar is not None:
                progbar.update(step, values=[("rows", len(ids))])
        if progbar is not None:
            progbar.update(step, values=[("rows", len(ids))], finalize=True)
        return ids

    async def from_csv(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        header: bool = True,
    ) -> Any:
        """Bulk-load a CSV file directly into the knowledge base.

        Skips the Python row pipeline entirely (no Pydantic, no Jinja,
        no per-row INSERT) and instead delegates to the database's
        native CSV reader. Roughly two orders of magnitude faster than
        ``update(CSVDataset(...))`` for non-trivial files — see
        ``benchmarks/bench_kb_ingest.py``.

        The target table's schema is inferred directly from the
        file's columns, with the first column promoted to PRIMARY
        KEY. The returned :class:`SymbolicDataModel` is the handle
        you pass to subsequent search / get calls — you don't need
        to pre-declare a ``DataModel`` for this table.

        Use the streaming ``update(<...>Dataset(...))`` path instead
        when source rows need transformation before storage (column
        renames, derived fields, HuggingFace datasets, etc.).

        Args:
            path: Path to the CSV file.
            table_name: Target table name. Defaults to the file's stem
                (``/data/my-docs.csv`` → ``MyDocs``). Whatever value
                lands here is always normalized to PascalCase.
            table_description: Optional natural-language description
                attached to the resulting schema.
            delimiter: Field delimiter. Defaults to ``","``.
            encoding: File encoding. Defaults to ``"utf-8"``.
            header: Whether the first row is a header. Defaults to
                ``True``.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table.
        """
        return await self.adapter.from_csv(
            path,
            table_name=table_name,
            table_description=table_description,
            delimiter=delimiter,
            encoding=encoding,
            header=header,
        )

    async def from_parquet(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> Any:
        """Bulk-load a Parquet file directly into the knowledge base.

        Same trade-offs as :meth:`from_csv` — bypasses the Python row
        pipeline for native database ingestion. Parquet's schema is
        explicit in the file footer so there is no type-inference
        guesswork to worry about.

        Args:
            path: Path to the Parquet file.
            table_name: Target table name. Defaults to the file's stem
                coerced to PascalCase.
            table_description: Optional schema description.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table.
        """
        return await self.adapter.from_parquet(
            path, table_name=table_name, table_description=table_description
        )

    async def from_json(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> Any:
        """Bulk-load a JSON file (top-level array of objects).

        Same trade-offs as :meth:`from_csv` / :meth:`from_parquet` —
        bypasses the Python row pipeline. The file must contain a
        top-level JSON array. Use :meth:`from_jsonl` for the
        one-object-per-line NDJSON format.

        Args:
            path: Path to the JSON file.
            table_name: Target table name. Defaults to the file's stem
                coerced to PascalCase.
            table_description: Optional schema description.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table.
        """
        return await self.adapter.from_json(
            path, table_name=table_name, table_description=table_description
        )

    async def from_jsonl(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> Any:
        """Bulk-load a JSON Lines (NDJSON) file.

        Same trade-offs as :meth:`from_csv` / :meth:`from_parquet`,
        and the right call for very large JSON sources that aren't
        a single array.

        Args:
            path: Path to the JSONL file.
            table_name: Target table name. Defaults to the file's stem
                coerced to PascalCase.
            table_description: Optional schema description.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table.
        """
        return await self.adapter.from_jsonl(
            path, table_name=table_name, table_description=table_description
        )

    async def rename(
        self,
        source: Any,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> Any:
        """Rename a table and/or update its description.

        Pass at least one of ``table_name`` / ``table_description``.
        When ``table_name`` is given the underlying table is
        renamed via ``ALTER TABLE …``, the FTS / vector indexes are
        rebuilt under the new name, and the adapter's known-models
        list is updated so subsequent default-table searches find
        the table under its new identity.

        Args:
            source: ``SymbolicDataModel`` or table-name string for
                the table to rename. The string form is itself
                PascalCase-normalized, so callers can pass the
                same input they used in :meth:`from_csv` (e.g.
                ``"my-docs"``).
            table_name: New table name. Always normalized to
                PascalCase.
            table_description: Optional natural-language description
                attached to the resulting schema.

        Returns:
            A fresh :class:`SymbolicDataModel` for the (possibly
            renamed) table.
        """
        return await self.adapter.rename(
            source,
            table_name=table_name,
            table_description=table_description,
        )

    async def get(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
    ) -> Union[Optional[Any], List[Optional[Any]]]:
        """Retrieve one or more records by primary key from a single table.

        Args:
            id_or_ids: A single primary key value, or a list of values.
            table_name: Target table.

        Returns:
            A single JsonDataModel (or ``None``) when called with one id;
            a list of JsonDataModels (with ``None`` in the slots that did
            not match) when called with a list.
        """
        return await self.adapter.get(id_or_ids, table_name=table_name)

    async def getall(
        self,
        *,
        table_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Any]:
        """Retrieve all records from a table with pagination.

        Args:
            table_name: Target table.
            limit: Maximum number of records to return (default: 50).
            offset: Number of records to skip (default: 0).

        Returns:
            List of JsonDataModels.
        """
        return await self.adapter.getall(
            table_name=table_name, limit=limit, offset=offset
        )

    async def delete(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
    ) -> int:
        """Delete records by primary key from a single table.

        Pass a single id or a list. The FTS / vector indexes for the
        table are rebuilt afterwards so subsequent search calls
        don't return ghost rows.

        Args:
            id_or_ids: Primary key value, or a list of values.
            table_name: Target table.

        Returns:
            The number of rows actually deleted (0 if no id matched).
        """
        return await self.adapter.delete(id_or_ids, table_name=table_name)

    async def drop_table(self, table_name: str) -> bool:
        """Drop a table from the knowledge base.

        Removes the table's rows, FTS index, and HNSW vector index,
        then drops the table itself. Also forgets the table in the
        adapter's known-models list.

        Args:
            table_name: Target table.

        Returns:
            ``True`` if a table was dropped, ``False`` if it didn't
            exist to begin with.
        """
        return await self.adapter.drop_table(table_name)

    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        **kwargs,
    ) -> Union[List[Dict[str, Any]], str]:
        """Execute a raw SQL query against the knowledge base.

        Args:
            query (str): The SQL query to execute.
            params (dict): Optional list of parameters for parameterized queries.
            output_format: ``"json"`` (default, list of dicts —
                JSON-shaped Python data) or ``"csv"`` (CSV string,
                useful when handing the result to an LM).
            **kwargs (Any): Additional options. The most important one is
                ``read_only=True/False``. When ``True`` (the DuckDB adapter's
                default) two layers of defence apply:

                1. The SQL is parsed with the engine's own parser and any
                   non-``SELECT`` statement is rejected. This catches
                   multi-statement injection (e.g. ``SELECT 1; DROP TABLE x``),
                   ``COPY ... TO 'file'`` exfiltration, ``ATTACH``, ``EXPORT``,
                   and other side-effecting statements. This is the only
                   layer that blocks writes — the adapter's underlying
                   connection is read-write (one connection per adapter,
                   reused across operations), so the parser check is what
                   keeps untrusted SQL read-only.
                2. ``enable_external_access`` is disabled on that connection
                   at construction time, so ``SELECT`` table functions that
                   touch the host filesystem or network — ``read_csv``,
                   ``read_parquet``, ``read_json``, ``read_blob``,
                   ``read_text``, ``glob`` and the httpfs/S3 variants —
                   return a permission error instead of leaking files.
                   Without this layer,
                   ``SELECT * FROM read_csv('/etc/passwd', ...)`` would pass
                   defence (1) because it is a syntactically valid ``SELECT``.

                Pass ``read_only=False`` only from trusted call sites that
                genuinely need to mutate state. Those paths still run on
                the same sandboxed connection (no external I/O), but they
                bypass the parser check, so any SQL is accepted — keep them
                out of the LM-tool-call surface.

        Returns:
            (Union[List[Dict[str, Any]], str]): A list of dicts when
                ``output_format="json"``, or a CSV string when
                ``output_format="csv"``.
        """
        return await self.adapter.query(
            query, params=params, output_format=output_format, **kwargs
        )

    async def similarity_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        table_name: str,
        k: int = 10,
        threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """Vector similarity search against a single table.

        Args:
            text_or_texts: Query text or list of query texts.
            table_name: Target table (single-table search).
            k: Maximum number of results to return.
            threshold: Optional maximum vector-distance threshold.
            output_format: ``"json"`` (default, list of dicts —
                JSON-shaped Python data) or ``"csv"`` (CSV string,
                useful for handing results to an LM since CSV is
                ~30-50% fewer tokens than equivalent JSON).
        """
        return await self.adapter.similarity_search(
            text_or_texts,
            table_name=table_name,
            k=k,
            threshold=threshold,
            output_format=output_format,
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
        """BM25 full-text search against a single table.

        Args:
            text_or_texts: Query text or list of query texts.
            table_name: Target table.
            k: Maximum number of results.
            threshold: Optional minimum BM25 score.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).
        """
        return await self.adapter.fulltext_search(
            text_or_texts,
            table_name=table_name,
            k=k,
            threshold=threshold,
            output_format=output_format,
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

        DuckDB evaluates regexes with RE2, so patterns are linear-time
        and not vulnerable to catastrophic backtracking.

        Args:
            pattern: The regex pattern (RE2 syntax).
            table_name: Target table.
            fields: Field names to match against. Defaults to every
                string field on the schema. Names are snake_case-
                normalized to match stored column names.
            case_sensitive: When ``False``, match case-insensitively.
            k: Maximum number of results.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).
        """
        return await self.adapter.regex_search(
            pattern,
            table_name=table_name,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            output_format=output_format,
        )

    async def hybrid_fts_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        table_name: str,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """Reciprocal-Rank-Fusion of vector similarity + BM25 fulltext.

        Falls back to full-text-only when no embedding model is
        configured. The regex-side sibling is
        :meth:`hybrid_regex_search`.

        Args:
            text_or_texts: Query text or list of query texts.
            table_name: Target table.
            k: Maximum results.
            k_rank: RRF smoothing constant. Lower emphasizes top
                ranks more strongly (default: 60).
            similarity_threshold: Optional vector-distance threshold.
            fulltext_threshold: Optional BM25 threshold.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).
        """
        return await self.adapter.hybrid_fts_search(
            text_or_texts,
            table_name=table_name,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            output_format=output_format,
        )

    async def hybrid_search(self, *args, **kwargs):
        """Deprecated alias of :meth:`hybrid_fts_search`.

        Kept for backwards compatibility. The new name is symmetric
        with :meth:`hybrid_regex_search`; prefer it in new code.
        """
        return await self.hybrid_fts_search(*args, **kwargs)

    async def hybrid_regex_search(
        self,
        text_or_texts: Union[str, List[str]],
        pattern_or_patterns: Union[str, List[str], None] = None,
        *,
        table_name: str,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        output_format: str = "json",
    ):
        """Reciprocal-Rank-Fusion of vector similarity + regex.

        The regex-side counterpart to :meth:`hybrid_fts_search` (which
        pairs vector with BM25 fulltext). The two signals are
        orthogonal: vectors capture semantic similarity, regex
        captures exact textual shape. Ranks are fused with the same
        RRF formula.

        Args:
            text_or_texts: Natural-language query (or list) for the
                vector side.
            pattern_or_patterns: RE2 pattern (or list) for the regex
                side. ``None`` falls back to plain similarity search.
            table_name: Target table.
            k: Maximum results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Vector-distance threshold.
            fields: Forwarded to the regex side.
            case_sensitive: Forwarded to the regex side.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).
        """
        return await self.adapter.hybrid_regex_search(
            text_or_texts,
            pattern_or_patterns=pattern_or_patterns,
            table_name=table_name,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fields=fields,
            case_sensitive=case_sensitive,
            output_format=output_format,
        )

    def get_symbolic_data_models(self) -> List[Any]:
        """Retrieve all symbolic data models (table definitions) from the database.

        Returns a list of SymbolicDataModel objects representing each table
        in the database. This is useful for introspecting the database schema
        or for passing to search methods to limit the search scope.

        Returns:
            list: List of symbolic data models representing the database tables.

        Example:
            ```python
            symbolic_models = knowledge_base.get_symbolic_data_models()
            for model in symbolic_models:
                schema = model.get_schema()
                print(f"Table: {schema['title']}")
                print(f"Fields: {list(schema['properties'].keys())}")
            ```
        """
        return self.adapter.get_symbolic_data_models()

    def get_config(self):
        config = {
            "uri": self.uri,
            "name": self.name,
            "metric": self.metric,
            "wipe_on_start": self.wipe_on_start,
        }
        data_models_config = {
            "data_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        data_model.to_symbolic_data_model(
                            name="data_model" + (f"_{i}_" if i > 0 else "_") + self.name
                        )
                    )
                    if not is_symbolic_data_model(data_model)
                    else serialization_lib.serialize_synalinks_object(data_model)
                )
                for i, data_model in enumerate(self.data_models)
            ]
        }
        embedding_model_config = {}
        if self.embedding_model:
            embedding_model_config = {
                "embedding_model": serialization_lib.serialize_synalinks_object(
                    self.embedding_model,
                )
            }
        return {
            **data_models_config,
            **embedding_model_config,
            **config,
        }

    @classmethod
    def from_config(cls, config):
        data_models_config = config.pop("data_models", [])
        data_models = [
            serialization_lib.deserialize_synalinks_object(data_model)
            for data_model in data_models_config
        ]
        embedding_model = None
        if "embedding_model" in config:
            embedding_model = serialization_lib.deserialize_synalinks_object(
                config.pop("embedding_model"),
            )
        return cls(
            data_models=data_models,
            embedding_model=embedding_model,
            **config,
        )
