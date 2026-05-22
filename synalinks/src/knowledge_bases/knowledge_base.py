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
from synalinks.src.knowledge_bases import graph_database_adapters
from synalinks.src.knowledge_bases.graph_database_adapters import GraphDatabaseAdapter
from synalinks.src.modules.embedding_models import get as _get_em
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name
from synalinks.src.utils.progbar import Progbar


@synalinks_export("synalinks.KnowledgeBase")
class KnowledgeBase(SynalinksSaveable):
    """A knowledge base for storing and retrieving structured data.

    The KnowledgeBase provides a unified interface over two complementary
    stores: a SQL row/table store (DuckDB by default) and a property-graph
    store (LadybugDB by default). The two are orthogonal — SQL methods
    (``update``, ``sql``, ``similarity_search``, ...) route to the SQL
    adapter; graph methods (``update_entities``, ``cypher``,
    ``entity_similarity_search``, ...) route to the graph adapter.

    A no-args ``KnowledgeBase()`` instantiates BOTH stores under
    ``synalinks_home()`` (``database.db`` for SQL, ``database.lb`` for
    the graph) so the two sides are usable side-by-side without setup.
    Pass ``uri=`` alone for SQL-only, ``graph_uri=`` alone for
    graph-only, or both to point each side at a custom location.

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

    # Retrieve by ID (the first field, here 'id', is the primary key — see
    # the "Primary Key Convention" section below).
    result = await knowledge_base.get("1", table_name="Document")

    # Full-text search
    results = await knowledge_base.fulltext_search("Hello", k=10)
    ```

    ### Primary Key Convention

    Synalinks does not inject a synthetic ``uuid`` / ``_id`` column. The
    primary key is the **first declared field** of your DataModel, in
    declaration order, after skipping reserved structural fields:

    * For SQL tables (DuckDB): the first property of the schema.
    * For graph entities (Ladybug nodes): the first property after
      ``label``. ``label`` is the node-table name, not a column.
    * For graph relations (Ladybug edges): the first property after
      ``subj`` / ``label`` / ``obj``. Those three are reserved — the
      endpoints are resolved against the node tables, and the label is
      the edge-table name.

    Because the PK is just "whichever field you declared first", a
    KnowledgeBase can be pointed at a pre-existing DuckDB file or
    LadybugDB store without rewriting rows or renaming columns: declare
    your DataModel so its first field matches the column you already
    treat as the identifier (``id``, ``ticker``, ``isbn``, ``email``,
    whatever it happens to be) and the adapters will use it. If you
    *want* a UUID-style key, declare it explicitly as the first field
    and populate it yourself — generating identifiers is the caller's
    job, not the framework's.

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
        uri (str): SQL store connection URI (``"duckdb://path/to/db.db"``).
            When both ``uri`` and ``graph_uri`` are omitted, defaults to
            ``{synalinks_home()}/{name or 'database'}.db``. Pass ``uri``
            alone to opt out of the graph-side default.
        graph_uri (str): Graph store connection URI
            (``"ladybug://path/to/graph.lb"`` or
            ``"ladybug://:memory:"``). When both URIs are omitted,
            defaults to ``{synalinks_home()}/{name or 'database'}.lb``.
            Pass ``graph_uri`` alone to opt out of the SQL-side default.
        data_models (list): Optional list of DataModel or SymbolicDataModel
            classes to create tables for in the SQL store.
        entity_models (list): Optional list of entity (node) models for
            the graph store.
        relation_models (list): Optional list of relation (edge) models
            for the graph store.
        embedding_model (EmbeddingModel): Optional embedding model for
            vector similarity search; forwarded to both stores.
        metric (str): The distance metric for vector search.
            Options: "cosine", "l2sq", "ip" (default: "cosine").
        wipe_on_start (bool): Whether to clear the database on initialization
            (default: False).
        name (str): Optional name for the knowledge base (used for serialization
            and as the filename stem for the default ``.synalinks`` paths).
        encryption_key (str): Optional at-rest encryption key for the SQL
            store. Not forwarded to the graph store (LadybugDB has no
            encryption-at-rest support).
    """

    def __init__(
        self,
        *,
        uri=None,
        graph_uri=None,
        data_models=None,
        entity_models=None,
        relation_models=None,
        embedding_model=None,
        metric="cosine",
        wipe_on_start=False,
        name=None,
        encryption_key=None,
        **kwargs,
    ):
        # Two adapters can coexist on a single KnowledgeBase:
        #   * `sql_adapter` — row/table store, selected by `uri`
        #     (e.g. duckdb://...). Default backend is DuckDB.
        #   * `graph_adapter` — property-graph store, selected by
        #     `graph_uri` (e.g. ladybug://...). Default backend is
        #     LadybugDB.
        # The two stores are complementary, so a no-args
        # ``KnowledgeBase()`` instantiates BOTH against the same
        # ``synalinks_home()`` directory (``database.db`` for SQL,
        # ``database.lb`` for the graph). Passing only ``uri=`` keeps
        # the call SQL-only; passing only ``graph_uri=`` keeps it
        # graph-only — explicit URIs opt out of auto-pairing so a
        # caller targeting one engine isn't surprised by a second
        # file appearing on disk.
        self.sql_adapter = None
        self.graph_adapter = None

        auto_pair = uri is None and graph_uri is None
        want_sql = uri is not None or auto_pair
        want_graph = graph_uri is not None or auto_pair

        if want_sql:
            self.sql_adapter = database_adapters.get(uri)(
                uri=uri,
                data_models=data_models,
                embedding_model=embedding_model,
                metric=metric,
                wipe_on_start=wipe_on_start,
                name=name,
                encryption_key=encryption_key,
                **kwargs,
            )

        if want_graph:
            # `encryption_key` is intentionally NOT forwarded here:
            # LadybugDB has no encryption-at-rest support. A user that
            # passes it for a dual-adapter KB gets DuckDB encryption
            # for the SQL side and an unencrypted Ladybug graph store
            # (which is the same as if they'd omitted the kwarg).
            self.graph_adapter = graph_database_adapters.get(graph_uri)(
                uri=graph_uri,
                entity_models=entity_models,
                relation_models=relation_models,
                embedding_model=embedding_model,
                metric=metric,
                wipe_on_start=wipe_on_start,
                name=name,
                **kwargs,
            )

        self.uri = uri
        self.graph_uri = graph_uri
        self.data_models = data_models or []
        self.entity_models = entity_models or []
        self.relation_models = relation_models or []
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

                Upserts key off the first declared field of the model —
                see the "Primary Key Convention" section on the class
                docstring for how that's resolved (and why no UUID is
                injected).
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
        return await self.sql_adapter.update(data_model_or_data_models)

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
            batch_ids = await self.sql_adapter.update(list(x))
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
        return await self.sql_adapter.from_csv(
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
        return await self.sql_adapter.from_parquet(
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
        return await self.sql_adapter.from_json(
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
        return await self.sql_adapter.from_jsonl(
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
        return await self.sql_adapter.rename(
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
        return await self.sql_adapter.get(id_or_ids, table_name=table_name)

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
        return await self.sql_adapter.getall(
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
        return await self.sql_adapter.delete(id_or_ids, table_name=table_name)

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
        return await self.sql_adapter.drop_table(table_name)

    async def sql(
        self,
        sql: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        **kwargs,
    ) -> Union[List[Dict[str, Any]], str]:
        """Execute a raw SQL query against the knowledge base.

        Counterpart of :meth:`cypher` — the method is named after the
        query language so a dual-adapter KnowledgeBase has a clear
        per-language entry point.

        Args:
            sql (str): The SQL string to execute.
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
        return await self.sql_adapter.sql(
            sql, params=params, output_format=output_format, **kwargs
        )

    async def similarity_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        table_name: str,
        k: int = 10,
        threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        output_format: str = "json",
    ):
        """Vector similarity search against a single table.

        Args:
            text_or_texts: Query text or list of query texts.
            table_name: Target table (single-table search).
            k: Maximum number of results to return.
            threshold: Optional maximum vector-distance threshold.
            ef_search: HNSW search-time candidate-list depth.
                ``None`` keeps the index-time value (or the engine
                default). Higher = better recall, slower query.
            output_format: ``"json"`` (default, list of dicts —
                JSON-shaped Python data) or ``"csv"`` (CSV string,
                useful for handing results to an LM since CSV is
                ~30-50% fewer tokens than equivalent JSON).
        """
        return await self.sql_adapter.similarity_search(
            text_or_texts,
            table_name=table_name,
            k=k,
            threshold=threshold,
            ef_search=ef_search,
            output_format=output_format,
        )

    async def fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        table_name: str,
        k: int = 10,
        threshold: Optional[float] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        bm25_k: Optional[float] = None,
        output_format: str = "json",
    ):
        """BM25 full-text search against a single table.

        Args:
            text_or_texts: Query text or list of query texts.
            table_name: Target table.
            k: Maximum number of results.
            threshold: Optional minimum BM25 score.
            conjunctive: AND-mode query (every term must match).
                Default ``False`` keeps OR semantics.
            bm25_b: Optional override for BM25's ``b`` parameter
                (document-length normalization).
            bm25_k: Optional override for BM25's ``k1`` parameter
                (term-frequency saturation).
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).
        """
        return await self.sql_adapter.fulltext_search(
            text_or_texts,
            table_name=table_name,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            bm25_k=bm25_k,
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
        return await self.sql_adapter.regex_search(
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
        keywords: Optional[Union[str, List[str]]] = None,
        table_name: str,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        bm25_k: Optional[float] = None,
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
            ef_search: Forwarded to the vector branch; HNSW
                search-time candidate-list depth.
            conjunctive: Forwarded to the BM25 branch; AND-mode query.
            bm25_b: Forwarded to the BM25 branch; document-length
                normalization override.
            bm25_k: Forwarded to the BM25 branch; term-frequency
                saturation override.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).
        """
        return await self.sql_adapter.hybrid_fts_search(
            text_or_texts=text_or_texts,
            table_name=table_name,
            keywords=keywords,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            bm25_k=bm25_k,
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
        *,
        pattern_or_patterns: Union[str, List[str], None] = None,
        table_name: str,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
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
            ef_search: Forwarded to the vector branch; HNSW
                search-time candidate-list depth.
            fields: Forwarded to the regex side.
            case_sensitive: Forwarded to the regex side.
            output_format: ``"json"`` (list of dicts, default) / ``"csv"`` (text).
        """
        return await self.sql_adapter.hybrid_regex_search(
            text_or_texts=text_or_texts,
            pattern_or_patterns=pattern_or_patterns,
            table_name=table_name,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            ef_search=ef_search,
            fields=fields,
            case_sensitive=case_sensitive,
            output_format=output_format,
        )

    # ---------------------------------------------------------------------
    # Graph store API — orthogonal to the SQL store above.
    #
    # These methods require the underlying adapter to be a
    # ``GraphDatabaseAdapter`` (selected by the URI scheme, e.g.
    # ``ladybug://``). Calling them on a SQL-only KnowledgeBase raises
    # ``NotImplementedError`` with a clear message instead of an opaque
    # ``AttributeError``.
    # ---------------------------------------------------------------------

    def _require_graph_adapter(self) -> None:
        """Raise if no graph adapter is attached to this KnowledgeBase.

        The graph adapter is set up only when ``graph_uri`` is passed
        at construction time; calling a graph method on a SQL-only KB
        must fail with a clear message instead of an ``AttributeError``
        from accessing ``None``.
        """
        if not isinstance(self.graph_adapter, GraphDatabaseAdapter):
            raise NotImplementedError(
                "Graph operations require a graph database adapter "
                "(pass graph_uri='ladybug://...' at construction time)."
            )

    async def update_entities(
        self,
        entity_or_entities: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert or update one or more entities (nodes) in the graph.

        Graph-side counterpart of the SQL :meth:`update`. The name
        mirrors the :class:`Entities` data model; pass either a single
        ``Entity`` or a list — the return shape matches the input.

        Args:
            entity_or_entities: An ``Entity`` instance, or a list of
                them (or anything satisfying ``is_entity``).

        Returns:
            The node id(s) assigned by the backend. Scalar in / scalar
            out; list in / list out.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.update_entities(entity_or_entities)

    async def update_relations(
        self,
        relation_or_relations: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert or update one or more relations (edges) in the graph.

        Mirrors the :class:`Relations` data model. Each relation's
        ``subj`` and ``obj`` are upserted as needed so every edge has
        both endpoints.

        Args:
            relation_or_relations: A ``Relation`` instance, or a list
                of them (or anything satisfying ``is_relation``).

        Returns:
            The edge id(s) assigned by the backend. Scalar in / scalar
            out; list in / list out.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.update_relations(relation_or_relations)

    async def update_knowledge_graph(self, knowledge_graph: Any) -> Any:
        """Bulk-insert a full knowledge graph (entities + relations).

        Equivalent to calling :meth:`update_entities` then
        :meth:`update_relations`, but concrete adapters may optimize
        the combined path.

        Args:
            knowledge_graph: A ``KnowledgeGraph`` instance.

        Returns:
            A dict with ``{"entities": [...ids...], "relations":
            [...ids...]}``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.update_knowledge_graph(knowledge_graph)

    async def get_entity(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        label: str,
    ) -> Union[Optional[Any], List[Optional[Any]]]:
        """Retrieve one or more entities by primary key from a label.

        Args:
            id_or_ids: A single primary key value, or a list of values.
            label: The entity label (node type).

        Returns:
            A single ``JsonDataModel`` (or ``None``) for a scalar
            argument; a list (with ``None`` for misses) for a list
            argument.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.get_entity(id_or_ids, label=label)

    async def delete_entity(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        label: str,
    ) -> int:
        """Delete entities by primary key from a label.

        Incident relations are removed by the adapter.

        Args:
            id_or_ids: Primary key value, or a list of values.
            label: The entity label.

        Returns:
            The number of entities actually deleted.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.delete_entity(id_or_ids, label=label)

    async def delete_relation(
        self,
        *,
        label: str,
        source_id: Any,
        target_id: Any,
    ) -> int:
        """Delete a relation between two entities.

        Args:
            label: The relation label.
            source_id: The subject (source) entity's primary key.
            target_id: The object (target) entity's primary key.

        Returns:
            The number of edges actually deleted.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.delete_relation(
            label=label, source_id=source_id, target_id=target_id
        )

    async def cypher(
        self,
        query: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        **kwargs: Any,
    ) -> Union[List[Dict[str, Any]], str]:
        """Execute a raw Cypher query against the graph.

        The graph-store counterpart to :meth:`query` (which executes
        SQL). Kept under a distinct name to avoid ambiguity when the
        KnowledgeBase grows both surfaces.

        Args:
            query: The Cypher query string.
            params: Optional parameters for parameterized queries.
            output_format: ``"json"`` (default) or ``"csv"``.
            **kwargs: Adapter-specific options (e.g. ``read_only``).

        Returns:
            A list of dicts when ``output_format="json"``, or a CSV
            string when ``output_format="csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.cypher(
            query, params=params, output_format=output_format, **kwargs
        )

    async def entity_similarity_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        label: str,
        k: int = 10,
        threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        output_format: str = "json",
    ):
        """Vector similarity search over entities of a given label.

        Args:
            text_or_texts: Query text or list of query texts.
            label: The entity label to search within.
            k: Maximum number of results.
            threshold: Optional vector-distance threshold.
            ef_search: Engine-specific search-time recall knob (HNSW
                ``efs``). Higher = better recall but slower.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.entity_similarity_search(
            text_or_texts,
            label=label,
            k=k,
            threshold=threshold,
            ef_search=ef_search,
            output_format=output_format,
        )

    async def entity_fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        label: str,
        k: int = 10,
        threshold: Optional[float] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
    ):
        """BM25 full-text search over entities of a given label.

        Args:
            text_or_texts: Query text or list of query texts.
            label: The entity label to search within.
            k: Maximum number of results.
            threshold: Optional minimum BM25 score.
            conjunctive: AND-mode query (every term must match).
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.entity_fulltext_search(
            text_or_texts,
            label=label,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            output_format=output_format,
        )

    async def entity_regex_search(
        self,
        pattern: str,
        *,
        label: str,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        k: int = 10,
        output_format: str = "json",
    ):
        """Regex search over entities of a label.

        Graph-side counterpart of :meth:`regex_search`. Applies the
        pattern to every indexed string field on the entity (or to
        the caller-supplied subset via ``fields``) and returns rows
        whose any matching field hits.

        Args:
            pattern: The regex pattern.
            label: The entity label to search within.
            fields: Optional whitelist of fields.
            case_sensitive: When ``False``, matches case-insensitively.
            k: Maximum number of rows.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.entity_regex_search(
            pattern,
            label=label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            output_format=output_format,
        )

    async def entity_hybrid_regex_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        pattern_or_patterns: Optional[Union[str, List[str]]] = None,
        label: str,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """RRF fusion of vector similarity + regex match over entities.

        Sibling of :meth:`entity_hybrid_fts_search`. Falls through
        to :meth:`entity_similarity_search` when no patterns are
        supplied; falls through to :meth:`entity_regex_search` when
        no embedding model is configured.

        Args:
            text_or_texts: Query text or list of query texts for the
                vector branch.
            pattern_or_patterns: Regex pattern (or list) for the
                regex branch. ``None`` skips the regex side.
            label: The entity label.
            fields: Forwarded to :meth:`entity_regex_search`.
            case_sensitive: Forwarded to :meth:`entity_regex_search`.
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.entity_hybrid_regex_search(
            text_or_texts=text_or_texts,
            pattern_or_patterns=pattern_or_patterns,
            label=label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            output_format=output_format,
        )

    async def entity_hybrid_fts_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        keywords: Optional[Union[str, List[str]]] = None,
        label: str,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
    ):
        """RRF of vector similarity + BM25 fulltext over entities of a label.

        Graph-side counterpart of :meth:`hybrid_fts_search`.

        Args:
            text_or_texts: Query text or list of query texts.
            label: The entity label to search within.
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            fulltext_threshold: Optional BM25 threshold.
            ef_search: HNSW ``efs`` knob for the vector branch.
            conjunctive: AND vs OR for the BM25 branch.
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.entity_hybrid_fts_search(
            text_or_texts=text_or_texts,
            label=label,
            keywords=keywords,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            output_format=output_format,
        )

    async def relation_similarity_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        label: str,
        k: int = 10,
        threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        output_format: str = "json",
    ):
        """Vector similarity search over relations of a given label.

        The query text matches against BOTH endpoints (subject and
        object); the adapter returns one row per matched edge with
        its best (lowest) distance and a ``matched_on`` tag
        (``"subj"``, ``"obj"``, or ``"both"``).

        Args:
            text_or_texts: Query text or list of query texts.
            label: The relation label to search within.
            k: Maximum number of results.
            threshold: Optional vector-distance threshold per endpoint.
            ef_search: HNSW ``efs`` knob applied to both endpoint
                vector searches.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.relation_similarity_search(
            text_or_texts,
            label=label,
            k=k,
            threshold=threshold,
            ef_search=ef_search,
            output_format=output_format,
        )

    async def relation_fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        label: str,
        k: int = 10,
        threshold: Optional[float] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
    ):
        """BM25 fulltext search over relations of a given label.

        Per matched edge, the final ``score`` is the sum of the
        subject-side and object-side BM25 scores — either-endpoint
        union (edge surfaces if either endpoint matched).

        Args:
            text_or_texts: Query text or list of query texts.
            label: The relation label to search within.
            k: Maximum number of results.
            threshold: Optional minimum BM25 threshold applied per endpoint.
            conjunctive: AND-mode query (every term must match).
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.relation_fulltext_search(
            text_or_texts,
            label=label,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            output_format=output_format,
        )

    async def relation_regex_search(
        self,
        pattern: str,
        *,
        label: str,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        k: int = 10,
        output_format: str = "json",
    ):
        """Regex search over relations of a given label.

        Composed via :meth:`entity_regex_search` on each endpoint.
        Regex hits are binary; the row's ``score`` is 2.0 when both
        endpoints matched and 1.0 when only one did, with
        ``matched_on`` indicating the side(s).

        Args:
            pattern: The regex pattern.
            label: The relation label to search within.
            fields: Optional whitelist of fields, applied to both endpoints.
            case_sensitive: When ``False``, matches case-insensitively.
            k: Maximum number of rows.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.relation_regex_search(
            pattern,
            label=label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            output_format=output_format,
        )

    async def relation_hybrid_regex_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        pattern_or_patterns: Optional[Union[str, List[str]]] = None,
        label: str,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """RRF of vector similarity + regex match over relations.

        Per matched edge, the final ``rrf_score`` is the sum of the
        subject's and the object's hybrid scores — same 4-source-RRF
        reduction as :meth:`relation_hybrid_fts_search`. Falls through
        to :meth:`relation_similarity_search` when no patterns are
        supplied.

        Args:
            text_or_texts: Query text or list of query texts for the vector branch.
            pattern_or_patterns: Regex pattern (or list) for the regex branch.
            label: The relation label.
            fields: Forwarded to :meth:`entity_regex_search`.
            case_sensitive: Forwarded to :meth:`entity_regex_search`.
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.relation_hybrid_regex_search(
            text_or_texts=text_or_texts,
            pattern_or_patterns=pattern_or_patterns,
            label=label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            output_format=output_format,
        )

    async def relation_hybrid_fts_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        keywords: Optional[Union[str, List[str]]] = None,
        label: str,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
    ):
        """RRF of vector + BM25 fulltext over relations of a label.

        Either-endpoint union: per matched edge, the final
        ``rrf_score`` is the sum of the subject-side and
        object-side hybrid scores — equivalent to a 4-source RRF.
        Falls back to fulltext-only when no embedding model is
        configured.

        Args:
            text_or_texts: Query text or list of query texts.
            label: The relation label to search within.
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            fulltext_threshold: Optional BM25 score threshold.
            ef_search: HNSW ``efs`` knob for the vector branch.
            conjunctive: AND vs OR for the BM25 branch.
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.relation_hybrid_fts_search(
            text_or_texts=text_or_texts,
            label=label,
            keywords=keywords,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            output_format=output_format,
        )

    async def path_hybrid_fts_search(
        self,
        subj_text_or_texts: Union[str, List[str]],
        obj_text_or_texts: Union[str, List[str]],
        *,
        subj_keywords: Optional[Union[str, List[str]]] = None,
        obj_keywords: Optional[Union[str, List[str]]] = None,
        subj_label: str,
        obj_label: str,
        label: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
    ):
        """Hybrid variable-length path search where BOTH endpoints match.

        AND-semantics. Each side is hybrid-searched (vec + fts)
        independently; per matching path the ``rrf_score`` is the
        sum of the subject-side and object-side hybrid scores.
        Falls back to fulltext-only when no embedding model is
        configured.

        Args:
            subj_text_or_texts: Query text (or list) for the subject.
            obj_text_or_texts: Query text (or list) for the object.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint for every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            fulltext_threshold: Optional BM25 score threshold.
            ef_search: HNSW ``efs`` knob applied to both endpoints.
            conjunctive: AND vs OR for the BM25 branch.
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.path_hybrid_fts_search(
            subj_text_or_texts=subj_text_or_texts,
            obj_text_or_texts=obj_text_or_texts,
            subj_label=subj_label,
            obj_label=obj_label,
            subj_keywords=subj_keywords,
            obj_keywords=obj_keywords,
            label=label,
            min_hops=min_hops,
            max_hops=max_hops,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            output_format=output_format,
        )

    async def path_similarity_search(
        self,
        subj_text_or_texts: Union[str, List[str]],
        obj_text_or_texts: Union[str, List[str]],
        *,
        subj_label: str,
        obj_label: str,
        label: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        k: int = 10,
        subj_threshold: Optional[float] = None,
        obj_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        output_format: str = "json",
    ):
        """Variable-length path search where BOTH endpoints match.

        Returns paths of ``min_hops..max_hops`` edges whose start
        node is vector-close to ``subj_text_or_texts`` AND whose
        end node is vector-close to ``obj_text_or_texts``. ``label``
        is an optional rel-label constraint applied to every hop;
        when omitted, any edge type is allowed.

        Each row carries the full path: ``nodes`` (every node along
        the way, endpoints included), ``rels`` (every edge), and
        ``length`` (hop count), alongside the two endpoint distances
        and flattened endpoint PKs.

        Args:
            subj_text_or_texts: Query text (or list) for the subject.
            obj_text_or_texts: Query text (or list) for the object.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint for every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            subj_threshold: Optional subject-side distance threshold.
            obj_threshold: Optional object-side distance threshold.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.path_similarity_search(
            subj_text_or_texts,
            obj_text_or_texts,
            subj_label=subj_label,
            obj_label=obj_label,
            label=label,
            min_hops=min_hops,
            max_hops=max_hops,
            k=k,
            subj_threshold=subj_threshold,
            obj_threshold=obj_threshold,
            ef_search=ef_search,
            output_format=output_format,
        )

    async def path_fulltext_search(
        self,
        subj_text_or_texts: Union[str, List[str]],
        obj_text_or_texts: Union[str, List[str]],
        *,
        subj_label: str,
        obj_label: str,
        label: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        k: int = 10,
        threshold: Optional[float] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
    ):
        """BM25 variable-length path search, AND semantics.

        Same shape as :meth:`path_similarity_search` but driven by BM25
        fulltext on each endpoint. Per matched path, ``score`` is the
        sum of the subject-side and object-side BM25 scores.

        Args:
            subj_text_or_texts: Keyword query (or list) for the subject.
            obj_text_or_texts: Keyword query (or list) for the object.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint for every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            threshold: Optional minimum BM25 threshold per endpoint.
            conjunctive: AND-mode BM25 query.
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.path_fulltext_search(
            subj_text_or_texts=subj_text_or_texts,
            obj_text_or_texts=obj_text_or_texts,
            subj_label=subj_label,
            obj_label=obj_label,
            label=label,
            min_hops=min_hops,
            max_hops=max_hops,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
            output_format=output_format,
        )

    async def path_regex_search(
        self,
        subj_pattern: str,
        obj_pattern: str,
        *,
        subj_label: str,
        obj_label: str,
        label: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        k: int = 10,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        output_format: str = "json",
    ):
        """Regex variable-length path search, AND semantics.

        Both endpoints must match their respective regex pattern.
        Regex is binary; ranking is by path length (shorter first).

        Args:
            subj_pattern: Regex pattern for the subject endpoint.
            obj_pattern: Regex pattern for the object endpoint.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint for every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            fields: Optional whitelist of fields, applied to both endpoints.
            case_sensitive: When ``False``, matches case-insensitively.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.path_regex_search(
            subj_pattern=subj_pattern,
            obj_pattern=obj_pattern,
            subj_label=subj_label,
            obj_label=obj_label,
            label=label,
            min_hops=min_hops,
            max_hops=max_hops,
            k=k,
            fields=fields,
            case_sensitive=case_sensitive,
            output_format=output_format,
        )

    async def path_hybrid_regex_search(
        self,
        subj_text_or_texts: Union[str, List[str]],
        obj_text_or_texts: Union[str, List[str]],
        *,
        subj_pattern_or_patterns: Optional[Union[str, List[str]]] = None,
        obj_pattern_or_patterns: Optional[Union[str, List[str]]] = None,
        subj_label: str,
        obj_label: str,
        label: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        output_format: str = "json",
    ):
        """RRF of vector + regex variable-length path search, AND semantics.

        Each side is hybrid-searched (vec + regex) independently; the
        path's ``rrf_score`` is the sum of the two endpoint hybrid
        scores. Falls through to :meth:`path_similarity_search` when
        no patterns are supplied.

        Args:
            subj_text_or_texts: Query text (or list) for the subject vector branch.
            obj_text_or_texts: Query text (or list) for the object vector branch.
            subj_pattern_or_patterns: Regex pattern (or list) for the subject.
            obj_pattern_or_patterns: Regex pattern (or list) for the object.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint for every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            fields: Forwarded to the regex branch.
            case_sensitive: Forwarded to the regex branch.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.path_hybrid_regex_search(
            subj_text_or_texts=subj_text_or_texts,
            obj_text_or_texts=obj_text_or_texts,
            subj_pattern_or_patterns=subj_pattern_or_patterns,
            obj_pattern_or_patterns=obj_pattern_or_patterns,
            subj_label=subj_label,
            obj_label=obj_label,
            label=label,
            min_hops=min_hops,
            max_hops=max_hops,
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
        return self.sql_adapter.get_symbolic_data_models()

    def get_symbolic_entities(self) -> List[Any]:
        """Retrieve a ``SymbolicDataModel`` per node label in the graph.

        Graph-side counterpart of :meth:`get_symbolic_data_models`,
        split by graph role: returns only entity (node) schemas.
        Each schema carries a ``label`` ``const`` discriminator and
        one property per stored column.

        Returns:
            list[SymbolicDataModel]: one per existing node label.
        """
        self._require_graph_adapter()
        return self.graph_adapter.get_symbolic_entities()

    def get_symbolic_relations(self) -> List[Any]:
        """Retrieve a ``SymbolicDataModel`` per relation label in the graph.

        Each returned schema includes its endpoint node schemas under
        ``$defs`` and references them as ``subj`` / ``obj`` via
        ``$ref`` — same shape Pydantic v2 emits for a hand-written
        :class:`synalinks.Relation` subclass.

        Returns:
            list[SymbolicDataModel]: one per existing relation label.
        """
        self._require_graph_adapter()
        return self.graph_adapter.get_symbolic_relations()

    async def detect_communities(
        self,
        *,
        algorithm: str = "louvain",
        node_labels: Optional[List[str]] = None,
        rel_labels: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
    ) -> Any:
        """Run a community-detection algorithm on the graph store.

        Returns a :class:`KnowledgeGraphs` — one
        :class:`KnowledgeGraph` per detected community. Edges that
        straddle communities are dropped. See the adapter's
        documentation for algorithm-specific constraints (Louvain
        requires a single node label; WCC / SCC accept any number).

        Args:
            algorithm: ``"louvain"`` (default),
                ``"weakly_connected_components"``, or
                ``"strongly_connected_components"``.
            node_labels: Optional whitelist of NODE tables to
                project. ``None`` = every existing one.
            rel_labels: Optional whitelist of REL tables to project.
                ``None`` = every existing one.
            max_iterations: Optional upper bound on the algorithm's
                iteration count. ``None`` defers to the engine
                default.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.detect_communities(
            algorithm=algorithm,
            node_labels=node_labels,
            rel_labels=rel_labels,
            max_iterations=max_iterations,
        )

    async def pagerank(
        self,
        *,
        node_labels: Optional[List[str]] = None,
        rel_labels: Optional[List[str]] = None,
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        tolerance: Optional[float] = None,
        normalize_initial: Optional[bool] = None,
        k: Optional[int] = None,
        output_format: str = "json",
    ):
        """Rank entities by PageRank importance on the graph store.

        Returns rows shaped like
        ``{<pk_column>: <pk_value>, "label": <label>, "node": <full node>,
        "rank": <float>}`` sorted by ``rank`` descending. The per-label
        PK column name is preserved verbatim, mirroring
        :meth:`entity_similarity_search`.

        Args:
            node_labels: Optional whitelist of NODE tables. ``None``
                projects every existing one.
            rel_labels: Optional whitelist of REL tables. ``None``
                projects every existing one.
            damping_factor: Probability of following an edge vs
                teleporting; 0.85 is the textbook value.
            max_iterations: Upper bound on iterations before
                convergence.
            tolerance: Optional convergence threshold; the algorithm
                stops early when the L1 change between iterations
                falls below this value. ``None`` defers to the
                engine default.
            normalize_initial: Whether to normalize the initial rank
                vector. ``None`` defers to the engine default.
            k: Optional cap on returned rows.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.pagerank(
            node_labels=node_labels,
            rel_labels=rel_labels,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
            normalize_initial=normalize_initial,
            k=k,
            output_format=output_format,
        )

    async def local_graph_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        label: str,
        max_hops: int = 2,
        k: int = 10,
        threshold: Optional[float] = None,
        rel_label: Optional[str] = None,
        ef_search: Optional[int] = None,
    ):
        """GraphRAG-style *local* search on the graph store.

        Vector-matches ``k`` seed entities of ``label``, expands their
        ``max_hops`` undirected neighbourhood, and returns the deduped
        union as a :class:`KnowledgeGraph` — the local context subgraph
        for entity-centric questions ("what does the graph say around
        *these* entities"). See
        :meth:`GraphDatabaseAdapter.local_graph_search`.

        Args:
            text_or_texts: Query text (or list); neighbourhoods merge.
            label: Entity label whose vector index seeds the search.
            max_hops: Neighbourhood radius in edges (>= 1, default 2).
            k: Number of seed entities per query text.
            threshold: Optional seed vector-distance ceiling.
            rel_label: Optional rel-label constraint per hop.
            ef_search: Optional HNSW search-depth for the seed lookup.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.local_graph_search(
            text_or_texts,
            label=label,
            max_hops=max_hops,
            k=k,
            threshold=threshold,
            rel_label=rel_label,
            ef_search=ef_search,
        )

    async def build_communities(
        self,
        *,
        algorithm: str = "louvain",
        node_labels: Optional[List[str]] = None,
        rel_labels: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
        with_pagerank: bool = True,
        damping_factor: float = 0.85,
    ) -> int:
        """Materialize community membership (and PageRank) onto nodes.

        The index-time half of GraphRAG-global: run once after loading
        the graph so :meth:`global_graph_search` can read precomputed
        ``community`` / ``rank`` properties instead of re-clustering on
        every query. Idempotent. See
        :meth:`GraphDatabaseAdapter.build_communities`.

        Args:
            algorithm: Community-detection algorithm; see
                :meth:`detect_communities`.
            node_labels: Optional NODE-table whitelist (``None`` = all).
            rel_labels: Optional REL-table whitelist (``None`` = all).
            max_iterations: Optional clustering iteration cap.
            with_pagerank: Also stamp a PageRank importance score.
            damping_factor: PageRank damping factor.

        Returns:
            (int): the number of nodes stamped.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.build_communities(
            algorithm=algorithm,
            node_labels=node_labels,
            rel_labels=rel_labels,
            max_iterations=max_iterations,
            with_pagerank=with_pagerank,
            damping_factor=damping_factor,
        )

    async def global_graph_search(
        self,
        *,
        node_labels: Optional[List[str]] = None,
        k: int = 10,
        members_per_community: int = 10,
        output_format: str = "json",
    ):
        """GraphRAG-style *global* search on the graph store.

        Rolls up the community / rank properties
        :meth:`build_communities` stamped into one aggregate row per
        community (size, total rank, representative members), ordered
        by importance — the theme-centric counterpart to
        :meth:`local_graph_search` ("what are the overall patterns
        across the *whole* graph"). Requires :meth:`build_communities`
        to have run first. See
        :meth:`GraphDatabaseAdapter.global_graph_search`.

        Args:
            node_labels: Optional NODE-table whitelist (``None`` = every
                stamped table).
            k: Maximum number of communities to return.
            members_per_community: Cap on members carried per community.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        self._require_graph_adapter()
        return await self.graph_adapter.global_graph_search(
            node_labels=node_labels,
            k=k,
            members_per_community=members_per_community,
            output_format=output_format,
        )

    def _serialize_models(self, models, key):
        """Serialize a list of DataModels to their symbolic form.

        Shared between ``data_models``, ``entity_models``, and
        ``relation_models`` since each list goes through the same
        symbolic-model conversion before serialization.
        """
        return [
            (
                serialization_lib.serialize_synalinks_object(
                    model.to_symbolic_data_model(
                        name=key + (f"_{i}_" if i > 0 else "_") + self.name
                    )
                )
                if not is_symbolic_data_model(model)
                else serialization_lib.serialize_synalinks_object(model)
            )
            for i, model in enumerate(models)
        ]

    def get_config(self):
        config = {
            "uri": self.uri,
            "graph_uri": self.graph_uri,
            "name": self.name,
            "metric": self.metric,
            "wipe_on_start": self.wipe_on_start,
        }
        data_models_config = {
            "data_models": self._serialize_models(self.data_models, "data_model"),
            "entity_models": self._serialize_models(self.entity_models, "entity_model"),
            "relation_models": self._serialize_models(
                self.relation_models, "relation_model"
            ),
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
        def _deserialize(items):
            return [
                serialization_lib.deserialize_synalinks_object(item) for item in items
            ]

        data_models = _deserialize(config.pop("data_models", []))
        entity_models = _deserialize(config.pop("entity_models", []))
        relation_models = _deserialize(config.pop("relation_models", []))
        embedding_model = None
        if "embedding_model" in config:
            embedding_model = serialization_lib.deserialize_synalinks_object(
                config.pop("embedding_model"),
            )
        return cls(
            data_models=data_models,
            entity_models=entity_models,
            relation_models=relation_models,
            embedding_model=embedding_model,
            **config,
        )
