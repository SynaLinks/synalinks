# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.embedding_models import get as _get_em


class GraphDatabaseAdapter:
    """Base class for graph database adapters.

    GraphDatabaseAdapter provides a unified interface for storing and
    retrieving graph-structured data — entities, relations, and full
    knowledge graphs — with optional embedding-based similarity search
    and full-text search on entity properties.

    Orthogonal to `DatabaseAdapter` (the row/SQL surface): both
    can back a `KnowledgeBase` instance, but the user-facing API
    routes SQL methods (``update``/``get``/``query``/...) to the SQL
    adapter and graph methods (``add_entity``/``add_relation``/
    ``cypher``/...) to a graph adapter.

    Subclasses (e.g. ``LadybugAdapter``) must implement the abstract
    methods to provide concrete graph DB functionality.
    """

    def __init__(
        self,
        uri=None,
        embedding_model=None,
        entity_models=None,
        relation_models=None,
        metric="cosine",
        wipe_on_start=False,
        name=None,
        **kwargs,
    ):
        """Initialize the graph database adapter.

        Unlike `DatabaseAdapter`, schemas are split by graph
        role: entity schemas declare what node labels exist, relation
        schemas declare what edge labels exist. The split matches how
        a property-graph DB models the world and removes the
        ambiguity of a single ``data_models`` bucket.

        Args:
            uri (str): The graph database connection URI or path.
            embedding_model (EmbeddedModel): Optional embedding model for
                vector similarity search on entities.
            entity_models (list): Optional list of ``DataModel`` /
                ``SymbolicDataModel`` classes whose schemas declare the
                allowed entity (node) labels.
            relation_models (list): Optional list of ``DataModel`` /
                ``SymbolicDataModel`` classes whose schemas declare the
                allowed relation (edge) labels.
            metric (str): Distance metric for vector search. Options
                depend on the specific adapter implementation.
            wipe_on_start (bool): Whether to clear the database on
                initialization.
            name (str): Optional name for the adapter instance.
        """
        self.uri = uri
        self.embedding_model = _get_em(embedding_model)
        self.entity_models = entity_models or []
        self.relation_models = relation_models or []
        self.metric = metric
        self.name = name

        if wipe_on_start:
            self.wipe_database()

    def wipe_database(self):
        """Clear all nodes and edges from the graph database.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `wipe_database()` method"
        )

    def get_symbolic_entities(self) -> List[SymbolicDataModel]:
        """Retrieve a ``SymbolicDataModel`` per node label in the graph.

        Graph-side counterpart of
        `DatabaseAdapter.get_symbolic_data_models`, split by
        graph role: this method returns only entity (node) schemas.
        Useful for introspection and for passing as ``data_models``
        when wiring search APIs.

        Returns:
            list[SymbolicDataModel]: one per existing node table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`get_symbolic_entities()` method"
        )

    async def detect_communities(
        self,
        *,
        algorithm: str = "louvain",
        node_labels: Optional[List[str]] = None,
        rel_labels: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
    ) -> Any:
        """Run a community-detection algorithm and return the result
        as a `KnowledgeGraphs` — one `KnowledgeGraph`
        per detected community.

        Each community's `KnowledgeGraph` carries the nodes
        whose computed community-id matches, plus every relation
        whose two endpoints fall in the same community.
        Cross-community edges are dropped — they don't belong to
        any single community's subgraph by definition.

        Args:
            algorithm: ``"louvain"`` (default), ``"weakly_connected_components"``,
                or ``"strongly_connected_components"``. Louvain does
                modularity-based clustering; WCC / SCC return
                connected components under their respective edge
                semantics. Concrete adapters may support a wider set.
            node_labels: Optional restriction on which node tables
                participate. ``None`` means every existing node
                table. Some algorithms (e.g. Louvain) only support
                a single node label — the adapter raises if the
                constraint is violated.
            rel_labels: Optional restriction on which relation
                tables participate. ``None`` means every existing
                rel table.
            max_iterations: Optional upper bound on the algorithm's
                iteration count. ``None`` uses the engine default
                (Ladybug applies its own default for each algorithm).

        Returns:
            (KnowledgeGraphs): A list-wrapper around one
            `KnowledgeGraph` per detected community.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`detect_communities()` method"
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
        """Rank entities by PageRank importance.

        Each row carries the entity's PK column, its label, the raw
        node struct (so callers can extract any property), and the
        computed rank — sorted by rank descending. Mirrors the row
        shape of `entity_similarity_search` (PK column + node
        + scalar metric) so the two surfaces feel symmetric.

        Args:
            node_labels: Optional whitelist of NODE tables. ``None``
                = every existing one. Use this to restrict PageRank
                to a subgraph.
            rel_labels: Optional whitelist of REL tables. ``None``
                = every existing one.
            damping_factor: Probability of following an edge vs
                teleporting; 0.85 is the standard textbook value.
            max_iterations: Upper bound on iterations before
                convergence.
            tolerance: Optional convergence threshold; the algorithm
                stops early when the L1 change between iterations
                falls below this value. ``None`` defers to the
                engine default.
            normalize_initial: Whether to normalize the initial rank
                vector. ``None`` defers to the engine default.
            k: Optional cap on returned rows. ``None`` returns every
                ranked entity.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `pagerank()` method"
        )

    def get_symbolic_relations(self) -> List[SymbolicDataModel]:
        """Retrieve a ``SymbolicDataModel`` per relation label in the graph.

        Returns one symbolic model per edge table; the schema includes
        a ``subj`` / ``obj`` pair referencing the endpoint node tables
        via ``$ref`` into ``$defs``, plus any extra edge properties.

        Returns:
            list[SymbolicDataModel]: one per existing rel table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`get_symbolic_relations()` method"
        )

    async def update_entities(
        self,
        entity_or_entities: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert or update one or more entities (nodes) in the graph.

        Named to mirror the SQL `DatabaseAdapter.update` verb and
        the `Entities` data model. Accepts a scalar ``Entity``
        or a list of them; the return shape matches the input.

        Args:
            entity_or_entities: A single ``Entity`` instance, or a list
                of ``Entity`` instances (or anything whose schema
                satisfies ``is_entity``).

        Returns:
            The node id(s) assigned by the backend. Scalar in / scalar
            out; list in / list out.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `update_entities()` method"
        )

    async def update_relations(
        self,
        relation_or_relations: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert or update one or more relations (edges) in the graph.

        Each relation's ``subj`` and ``obj`` entities are upserted as
        needed so every edge always has both endpoints.

        Args:
            relation_or_relations: A single ``Relation`` instance, or
                a list of ``Relation`` instances (or anything whose
                schema satisfies ``is_relation``).

        Returns:
            The edge id(s) assigned by the backend. Scalar in / scalar
            out; list in / list out.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `update_relations()` method"
        )

    async def update_knowledge_graph(
        self,
        knowledge_graph: Any,
    ) -> Any:
        """Bulk-insert an entire knowledge graph (entities + relations).

        Equivalent to `update_entities` followed by
        `update_relations`, but concrete adapters may optimize
        the combined path (single transaction, batched writes, etc.).

        Args:
            knowledge_graph: A ``KnowledgeGraph`` instance whose
                ``entities`` and ``relations`` fields contain the data
                to store.

        Returns:
            A dict with ``{"entities": [...ids...], "relations":
            [...ids...]}``.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`update_knowledge_graph()` method"
        )

    async def rename(
        self,
        source: Any,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> SymbolicDataModel:
        """Rename a node/relation label and/or update its description.

        Graph counterpart of `DatabaseAdapter.rename`. The
        ``table_name`` kwarg keeps the SQL-side spelling so the
        `KnowledgeBase` can route ``rename`` through either
        adapter with the same signature; for the graph adapter it
        names the new node/relation **label**.

        Args:
            source: ``SymbolicDataModel`` or label string for the
                node/relation table to rename.
            table_name: New label. Optional — pass to ``ALTER`` the
                node or relation table.
            table_description: New schema description. Optional.

        Returns:
            The updated ``SymbolicDataModel`` for the renamed table.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `rename()` method"
        )

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
            A single ``JsonDataModel`` (or ``None``) when called with one
            id; a list with ``None`` in the slots that did not match
            when called with a list.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `get_entity()` method"
        )

    async def delete_entity(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        label: str,
    ) -> int:
        """Delete entities by primary key from a single label.

        Any relations incident to the deleted entities are removed as
        well (concrete adapters are expected to keep the graph
        referentially consistent).

        Args:
            id_or_ids: Primary key value, or a list of values.
            label: The entity label (node type).

        Returns:
            The number of entities actually deleted.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `delete_entity()` method"
        )

    async def delete_relation(
        self,
        *,
        label: str,
        source_id: Any,
        target_id: Any,
    ) -> int:
        """Delete a relation between two entities.

        Args:
            label: The relation label (edge type).
            source_id: The subject (source) entity's primary key.
            target_id: The object (target) entity's primary key.

        Returns:
            The number of edges actually deleted.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `delete_relation()` method"
        )

    async def cypher(
        self,
        query: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        **kwargs,
    ) -> Union[List[Dict[str, Any]], str]:
        """Execute a raw Cypher query against the graph.

        Args:
            query: The Cypher query string.
            params: Optional parameters for parameterized queries.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
            **kwargs: Additional adapter-specific options
                (e.g. ``read_only=True`` to refuse mutating statements).

        Returns:
            A list of dicts when ``output_format="json"``, or a CSV
            string when ``output_format="csv"``.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `cypher()` method"
        )

    async def entity_similarity_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        label: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        k: int = 10,
        threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        output_format: str = "json",
    ):
        """Vector similarity search over entities of a given label.

        Either ``text_or_texts`` (embedded) or ``vector_or_vectors`` (a
        pre-computed query vector or list of vectors) selects what to
        search for. When vectors are supplied no embedding model is
        required. Returns the ``k`` entities whose embedding is closest
        to the query.

        Args:
            text_or_texts: Query text or list of query texts. Ignored
                when ``vector_or_vectors`` is supplied.
            label: The entity label (node type) to search within.
            vector_or_vectors: Pre-computed query vector or list of
                vectors to search with directly.
            k: Maximum number of results.
            threshold: Optional vector-distance threshold.
            ef_search: Engine-specific search-time recall knob (HNSW
                ``efs``). Higher = better recall but slower. ``None``
                defers to the engine default.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`entity_similarity_search()` method"
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
            label: The entity label (node type) to search within.
            k: Maximum number of results.
            threshold: Optional minimum BM25 score.
            conjunctive: AND-mode query (every term must match).
                Default ``False`` keeps OR semantics.
            bm25_b: Optional override for BM25's ``b`` parameter
                (document-length normalization). ``None`` defers to
                the engine default (typically 0.75).
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`entity_fulltext_search()` method"
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
        """Find entities whose string properties match a regular expression.

        Graph-side counterpart of
        `DatabaseAdapter.regex_search`. The pattern is applied
        to every string field on the schema (or to a caller-supplied
        subset via ``fields``) and the union of matches is returned
        up to ``k`` rows.

        Args:
            pattern: The regex pattern.
            label: The entity label to search within.
            fields: Field names to match against. Defaults to every
                string-typed field. Names not present on the schema
                are silently dropped.
            case_sensitive: When ``False``, matches case-insensitively.
            k: Maximum number of results.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`entity_regex_search()` method"
        )

    async def entity_hybrid_regex_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        pattern_or_patterns: Optional[Union[str, List[str]]] = None,
        label: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        output_format: str = "json",
    ):
        """RRF fusion of vector similarity + regex match over entities.

        Sibling of `entity_hybrid_fts_search` — the regex side
        carries the orthogonal "exact textual shape" signal that
        BM25 doesn't capture. Degenerates to plain similarity search
        when no patterns are supplied, or to plain regex search when
        there are no vectors to search with.

        Args:
            text_or_texts: Query text or list of query texts for the
                vector branch. Ignored when ``vector_or_vectors`` is
                supplied.
            pattern_or_patterns: Regex pattern (or list) for the
                regex branch. ``None`` skips the regex side.
            label: The entity label to search within.
            vector_or_vectors: Pre-computed query vector(s) for the
                vector branch, used directly instead of embedding text.
            fields: Forwarded to `entity_regex_search`.
            case_sensitive: Forwarded to `entity_regex_search`.
            k: Maximum number of results.
            k_rank: RRF smoothing constant (default: 60).
            similarity_threshold: Optional vector-distance threshold.
            output_format: ``"json"`` (default) or ``"csv"``.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`entity_hybrid_regex_search()` method"
        )

    async def entity_hybrid_fts_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        keywords: Optional[Union[str, List[str]]] = None,
        label: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
    ):
        """Reciprocal-Rank-Fusion of vector similarity + BM25 fulltext.

        Graph-side counterpart of `DatabaseAdapter.hybrid_fts_search`.
        Falls back to fulltext-only when there are no vectors to search
        with.

        Args:
            text_or_texts: Query text or list of query texts for the
                vector branch. Ignored when ``vector_or_vectors`` is
                supplied.
            label: The entity label to search within.
            keywords: Query text or list of query texts for the BM25
                branch. Aligns by position with the vector-branch
                queries; when omitted, the text is reused for both.
            vector_or_vectors: Pre-computed query vector(s) for the
                vector branch, used directly instead of embedding text.
            k: Maximum number of results.
            k_rank: RRF smoothing constant; lower emphasizes top
                ranks more strongly (default: 60).
            similarity_threshold: Optional vector-distance threshold.
            fulltext_threshold: Optional minimum BM25 score.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`entity_hybrid_fts_search()` method"
        )

    async def relation_similarity_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        label: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        k: int = 10,
        threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        output_format: str = "json",
    ):
        """Vector similarity search over relations of a given label.

        The query matches against BOTH endpoints (subject and
        object); the union of subj-hits and obj-hits is taken, then
        deduplicated per ``(subj_pk, obj_pk)`` pair so each matched
        edge appears once with its best (lowest) distance. The
        ``matched_on`` field on each row indicates which side fired
        (``"subj"``, ``"obj"``, or ``"both"`` when both endpoint
        vectors surfaced the same edge).

        Args:
            text_or_texts: Query text or list of query texts. Ignored
                when ``vector_or_vectors`` is supplied.
            label: The relation label (edge type) to search within.
            vector_or_vectors: Pre-computed query vector or list of
                vectors to search with directly (matched against both
                endpoints).
            k: Maximum number of results.
            threshold: Optional vector-distance threshold applied to
                each endpoint search before the union.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`relation_similarity_search()` method"
        )

    async def relation_hybrid_fts_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        keywords: Optional[Union[str, List[str]]] = None,
        label: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
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

        Either-endpoint union: the query is hybrid-searched against
        the subject entity table and the object entity table; each
        endpoint match contributes its 2-source RRF score (vec + fts
        on that side) to any incident edge of ``label``. Per edge,
        the final score is the sum of the subj and obj contributions
        — mathematically equivalent to a single 4-source RRF over
        ``{subj_fts, subj_vec, obj_fts, obj_vec}``. ``matched_on``
        reports which side(s) actually fired.

        Falls back to fulltext-only when no embedding model is
        configured (same shape as `entity_hybrid_fts_search`).

        Args:
            text_or_texts: Query text or list of query texts for the
                vector branch (applied to both endpoints).
            label: The relation label (edge type) to search within.
            keywords: Query text or list of query texts for the BM25
                branch. Aligns by position with ``text_or_texts``;
                when omitted, the text is reused for both branches.
            k: Maximum number of results.
            k_rank: RRF smoothing constant (default: 60).
            similarity_threshold: Optional vector-distance threshold
                forwarded to the per-endpoint hybrid search.
            fulltext_threshold: Optional BM25 score threshold
                forwarded to the per-endpoint hybrid search.
            output_format: ``"json"`` (default) or ``"csv"``.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`relation_hybrid_fts_search()` method"
        )

    async def path_hybrid_fts_search(
        self,
        subj_text_or_texts: Optional[Union[str, List[str]]] = None,
        obj_text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        subj_keywords: Optional[Union[str, List[str]]] = None,
        obj_keywords: Optional[Union[str, List[str]]] = None,
        subj_label: str,
        obj_label: str,
        subj_vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        obj_vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
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
        """Hybrid (vec + BM25) variable-length path search, AND semantics.

        For each ``(subj_text, obj_text)`` pair both endpoints must
        hit. Each side is run through hybrid (vec + fts) so two
        signals contribute per endpoint; the path's combined
        ``rrf_score`` is the sum of the two endpoint RRF scores —
        mathematically equivalent to a 4-source RRF over the four
        underlying rankings.

        Falls back to fulltext-only when no embedding model is
        configured.

        Args:
            subj_text_or_texts: Query text (or list) for the subject
                vector branch.
            obj_text_or_texts: Query text (or list) for the object
                vector branch.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            subj_keywords: Query text (or list) for the subject BM25
                branch. Aligns by position with
                ``subj_text_or_texts``; when omitted, the text is
                reused for both branches on that side.
            obj_keywords: Same for the object side.
            label: Optional rel-label constraint applied to every
                hop in the path. ``None`` accepts any edge type.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            k_rank: RRF smoothing constant (default: 60).
            similarity_threshold: Optional vector-distance threshold.
            fulltext_threshold: Optional BM25 score threshold.
            output_format: ``"json"`` (default) or ``"csv"``.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`path_hybrid_fts_search()` method"
        )

    async def path_similarity_search(
        self,
        subj_text_or_texts: Optional[Union[str, List[str]]] = None,
        obj_text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        subj_label: str,
        obj_label: str,
        subj_vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        obj_vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
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

        Finds paths ``(s)-[*min_hops..max_hops]->(o)`` such that:

          * the subject ``s`` is vector-close to ``subj_text_or_texts``,
          * the object ``o`` is vector-close to ``obj_text_or_texts``,
          * the path connects them through ``min_hops..max_hops``
            intermediate edges.

        AND-semantics on the two query texts: rows survive only when
        both endpoint vectors clear their thresholds AND at least one
        path of valid length exists between them. ``label`` is an
        optional rel-label filter — when set, every hop in the path
        must be of that label; when ``None``, any edge type is
        allowed (Cypher's plain ``[*min..max]`` form).

        Each result row carries the full path: ``nodes`` (list of
        every node on the path, including the two endpoints),
        ``rels`` (list of every edge), ``length`` (hop count), plus
        the two endpoint distances and the flattened endpoint PKs.

        Args:
            subj_text_or_texts: Query text (or list) for the subject.
            obj_text_or_texts: Query text (or list) for the object.
            subj_label: The entity label of the subject endpoint.
                Required so the adapter knows which vector index
                to query for ``s``.
            obj_label: The entity label of the object endpoint.
                Required so the adapter knows which vector index
                to query for ``o``.
            label: Optional relation-label constraint applied to
                every hop in the path. ``None`` (default) accepts
                any edge type.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            subj_threshold: Optional subject-side distance ceiling.
            obj_threshold: Optional object-side distance ceiling.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`path_similarity_search()` method"
        )

    async def local_graph_search(
        self,
        text_or_texts: Optional[Union[str, List[str]]] = None,
        *,
        label: str,
        vector_or_vectors: Optional[Union[List[float], List[List[float]]]] = None,
        max_hops: int = 2,
        k: int = 10,
        threshold: Optional[float] = None,
        rel_label: Optional[str] = None,
        ef_search: Optional[int] = None,
    ) -> Any:
        """GraphRAG-style *local* search around vector-matched seeds.

        Finds the ``k`` entities of ``label`` closest to the query,
        then expands their ``max_hops`` neighbourhood (undirected) and
        returns the deduped union as a `KnowledgeGraph` — the
        local context subgraph a generator answers from. Entity-centric:
        "what does the graph say around *these* entities".

        Args:
            text_or_texts: Query text (or list); neighbourhoods merge.
            label: Entity label whose vector index seeds the search.
            max_hops: Neighbourhood radius in edges (>= 1).
            k: Number of seed entities per query text.
            threshold: Optional seed vector-distance ceiling.
            rel_label: Optional rel-label constraint per hop.
            ef_search: Optional HNSW search-depth for the seed lookup.

        Returns:
            (KnowledgeGraph): the deduped neighbourhood subgraph.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`local_graph_search()` method"
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

        The index-time half of GraphRAG-global: cluster the graph once
        and persist each node's community id (and importance rank) so
        that `global_graph_search` is a single aggregation read
        rather than a clustering pass per query. Idempotent.

        Args:
            algorithm: Community-detection algorithm; see
                `detect_communities`.
            node_labels: Optional NODE-table whitelist (``None`` = all).
            rel_labels: Optional REL-table whitelist (``None`` = all).
            max_iterations: Optional clustering iteration cap.
            with_pagerank: Also stamp a PageRank importance score.
            damping_factor: PageRank damping factor.

        Returns:
            (int): the number of nodes stamped.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `build_communities()` method"
        )

    async def global_graph_search(
        self,
        *,
        node_labels: Optional[List[str]] = None,
        k: int = 10,
        members_per_community: int = 10,
        output_format: str = "json",
    ) -> Any:
        """GraphRAG-style *global* search: per-community aggregates.

        The query-time half of GraphRAG-global. Reads the community /
        rank properties `build_communities` stamped and rolls
        them up into one row per community (size, aggregate rank,
        representative members), ordered by importance. Theme-centric:
        "what are the overall patterns across the *whole* graph". The
        LM map-reduce over the returned communities lives above the
        adapter.

        Args:
            node_labels: Optional NODE-table whitelist (``None`` = every
                stamped table).
            k: Maximum number of communities to return.
            members_per_community: Cap on members carried per community.
            output_format: ``"json"`` (default) or ``"csv"``.

        Returns:
            One aggregate row per community, ordered by importance.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            f"`global_graph_search()` method"
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} uri={self.uri}>"
