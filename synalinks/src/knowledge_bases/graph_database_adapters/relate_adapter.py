# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Relate graph adapter.

`Relate by Neo4j <https://github.com/relatedb/relate>`_ is an embedded,
in-process Cypher database: in memory or in a single file, with no server.
It speaks Neo4j's Cypher, with native vector and full-text indexes.

`RelateAdapter` is a `LadybugAdapter` whose engine-specific parts are
swapped: every search, upsert, GraphRAG and community surface is the
Ladybug one, built on three hooks this class overrides (the vector-index
call, the full-text-index call and its query text), so both adapters return
the same rows. What differs:

  * Relate is schema-free: a label exists once a node carries it, so the
    adapter keeps the schemas it is given (``entity_models`` /
    ``relation_models``, or the first entity of a free-form label) to answer
    `get_symbolic_entities` / `get_symbolic_relations`. Each label's primary
    key is held by a uniqueness constraint, which also lets a reopened store
    recover it.
  * Vector scores are converted to the distance Ladybug reports (lower is
    closer): ``1 - cos`` for ``cosine``, the (squared) Euclidean distance for
    ``l2`` / ``l2sq``. Relate has no dot-product index.
  * Full-text search is Lucene BM25: the query text is escaped (terms only,
    no Lucene syntax) and ``conjunctive=True`` requires every term.
    ``bm25_b`` and ``ef_search`` have no Relate equivalent and are ignored.
  * Relate stores no maps or nested lists, so a property of a JSON-typed
    field (an object, a list of objects) is stored as a JSON string and
    decoded on read.
  * Relate has no graph algorithms: PageRank, Louvain and the connected
    components come from `networkx`, run over the projected nodes and edges.

``cypher(..., read_only=True)`` adds one guard to the shared keyword
check: a ``CALL`` may only name a procedure Relate lists as ``READ``.
"""

import json
import re
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from synalinks.src.knowledge_bases.adapters_utils import IDENTIFIER_RE
from synalinks.src.knowledge_bases.adapters_utils import format_search_results
from synalinks.src.knowledge_bases.adapters_utils import resolve_db_path
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    LadybugAdapter,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    _assert_read_only_cypher,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    _cypher_type_to_json_property,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    sanitize_label,
)
from synalinks.src.modules.embedding_models import get as _get_em

#: The adapter's metric -> Relate's ``vector.similarity_function``.
METRICS = {"cosine": "cosine", "l2": "euclidean", "l2sq": "euclidean"}

#: Lucene query syntax, escaped so a query text is only ever terms.
_LUCENE_SPECIAL_RE = re.compile(r'([+\-!(){}\[\]^"~*?:\\/&|])')

#: The procedure a read-only ``CALL`` names.
_CALL_RE = re.compile(r"\bCALL\s+([A-Za-z_][\w.]*)", re.IGNORECASE)

#: Relate scalar types (``db.schema.*TypeProperties``) -> Ladybug's, so an
#: introspected label maps to JSON schema like a declared one.
_RELATE_TYPES = {
    "STRING": "STRING",
    "INTEGER": "INT64",
    "FLOAT": "DOUBLE",
    "BOOLEAN": "BOOL",
    "DATE": "DATE",
    "LOCAL DATETIME": "TIMESTAMP",
    "ZONED DATETIME": "TIMESTAMP",
}


def _relate_type(relate_type: str) -> str:
    """``"LIST<FLOAT NOT NULL> NOT NULL"`` -> ``"DOUBLE[]"``, as Ladybug
    names column types; ``STRING`` for anything else."""
    bare = relate_type.replace(" NOT NULL", "")
    if bare.startswith("LIST<") and bare.endswith(">"):
        return _RELATE_TYPES.get(bare[5:-1], "STRING") + "[]"
    return _RELATE_TYPES.get(bare, "STRING")


class RelateAdapter(LadybugAdapter):
    """Relate-backed implementation of `GraphDatabaseAdapter`.

    You don't create it directly: a `KnowledgeBase` uses it for a
    `relate://` graph URI and forwards its arguments. It needs the `relate`
    extra:

    ```shell
    pip install 'synalinks[relate]'
    ```

    It is a drop-in alternative to `LadybugAdapter` (same methods, same
    results) on Relate by Neo4j, which speaks Neo4j's Cypher. Each `Entity`
    model is a node label and each `Relation` model an edge type between
    the labels of its `subj` and `obj`. See the module documentation for the
    differences with Ladybug.

    Example:

    ```python
    import synalinks

    class Person(synalinks.Entity):
        label: str = synalinks.Field(default="Person", description="The entity label")
        name: str = synalinks.Field(description="Person name")

    class City(synalinks.Entity):
        label: str = synalinks.Field(default="City", description="The entity label")
        name: str = synalinks.Field(description="City name")

    class LivesIn(synalinks.Relation):
        label: str = synalinks.Field(default="LivesIn", description="The relation label")
        subj: Person
        obj: City

    knowledge_base = synalinks.KnowledgeBase(
        graph_uri="relate://./graph.relate",  # "relate://:memory:" for tests
        entity_models=[Person, City],
        relation_models=[LivesIn],
        embedding_model=synalinks.EmbeddingModel(model="ollama/mxbai-embed-large"),
    )

    # Writing a relation upserts its endpoints, deduplicated by embedding
    await knowledge_base.update_relations(
        [
            LivesIn(subj=Person(name="Alice"), obj=City(name="Paris")),
            LivesIn(subj=Person(name="Bob"), obj=City(name="London")),
        ]
    )

    # Entity search: BM25 full-text and vector similarity, fused by reciprocal rank
    people = await knowledge_base.entity_hybrid_fts_search(
        "Alice", label="Person", k=5
    )

    # Or Cypher (read-only by default)
    rows = await knowledge_base.cypher(
        "MATCH (p:Person)-[:LivesIn]->(c:City {name: 'Paris'}) RETURN p.name AS name"
    )
    ```

    Without an embedding model, entities are only merged on an identical
    key (no semantic deduplication), similarity search needs explicit
    vectors (`vector_or_vectors=`) and hybrid search uses full-text only.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        embedding_model: Optional[Any] = None,
        entity_models: Optional[List[Any]] = None,
        relation_models: Optional[List[Any]] = None,
        metric: str = "cosine",
        vector_dim: Optional[int] = None,
        dedup_threshold: float = 0.85,
        wipe_on_start: bool = False,
        name: Optional[str] = None,
        analyzer: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Relate adapter.

        Args:
            uri: ``relate://<path>`` (file-backed) or ``relate://:memory:``
                (in-memory). When omitted the adapter resolves to
                ``{synalinks_home()}/{name or 'database'}.relate``.
            embedding_model: Required for entity dedup and vector search,
                as for `LadybugAdapter`.
            entity_models: ``Entity`` subclasses; each one is a node label.
            relation_models: ``Relation`` subclasses; each one is an edge
                type between the labels of its ``subj`` / ``obj``.
            metric: ``"cosine"`` (default), ``"l2"`` or ``"l2sq"``.
            vector_dim: Embedding dimension. Inferred from the first
                embedding (or a reopened store's vector index) when omitted.
            dedup_threshold: Similarity above which an incoming entity is a
                duplicate of its nearest neighbour, as for `LadybugAdapter`.
            wipe_on_start: Whether to clear the database on init.
            name: Optional adapter name.
            analyzer: Lucene analyzer of the full-text indexes (e.g.
                ``"english"`` for stemming and stop words). ``None`` keeps
                Relate's default (``"standard-no-stop-words"``).
        """
        try:
            import networkx
            import relate
        except ImportError as e:
            raise ImportError(
                "RelateAdapter needs the `relate-db` and `networkx` packages: "
                "pip install 'synalinks[relate]'"
            ) from e
        if metric not in METRICS:
            raise ValueError(
                f"`metric` must be one of {sorted(METRICS)} for Relate "
                f"(it has no dot-product index), got {metric!r}"
            )
        self._relate = relate
        self._networkx = networkx
        self.uri = resolve_db_path(uri, scheme="relate", extension="relate", name=name)
        self.embedding_model = _get_em(embedding_model)
        self.entity_models = entity_models or []
        self.relation_models = relation_models or []
        self.metric = metric
        self.dedup_threshold = dedup_threshold
        self.name = name
        self.analyzer = analyzer
        self.vector_dim = vector_dim or 0

        # Relate keeps no schema of its own: these registries are the
        # adapter's view of it, filled as labels are declared or first seen.
        self._node_labels: set = set()
        self._rel_endpoints: Dict[str, tuple] = {}
        self._label_schemas: Dict[str, Dict[str, Any]] = {}
        self._rel_schemas: Dict[str, Dict[str, Any]] = {}
        self._json_properties: Dict[str, set] = {}

        self._db = relate.connect(self.uri)
        self._con = _RelateConnection(self)
        self._read_procedures = {
            row[0]
            for row in self._con.execute(
                "SHOW PROCEDURES YIELD name, mode WHERE mode = 'READ' RETURN name"
            ).get_all()
        }

        if wipe_on_start:
            self.wipe_database()
        pk_keys, fts_columns = self._stored_schema()
        self._defer_table_creation = bool(self.embedding_model and not self.vector_dim)
        self._setup_schema()
        for label, key in pk_keys.items():
            self._pk_keys.setdefault(label, key)
            self._node_labels.add(label)
        for label, columns in fts_columns.items():
            self._fts_columns.setdefault(label, columns)

    def _stored_schema(self) -> tuple:
        """What a reopened store already holds: ``(pk_keys, fts_columns)`` from
        its ``<label>_pk`` constraints and ``<label>_fts`` indexes, and the
        vector dimension from a ``<label>_vec`` index when not given."""
        pk_keys: Dict[str, str] = {}
        for name, labels, props in self._con.execute(
            "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties"
        ).get_all():
            if name.endswith("_pk") and labels and props:
                pk_keys[labels[0]] = props[0]
        fts_columns: Dict[str, List[str]] = {}
        for name, kind, labels, props, options in self._con.execute(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options"
        ).get_all():
            if not labels:
                continue
            if kind == "FULLTEXT" and name.endswith("_fts"):
                fts_columns[labels[0]] = list(props)
            if kind == "VECTOR" and name.endswith("_vec") and not self.vector_dim:
                config = (options or {}).get("indexConfig") or {}
                self.vector_dim = int(config.get("vector.dimensions") or 0)
        return pk_keys, fts_columns

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _register_json_properties(self, label: str, columns: List[str]) -> None:
        """Remember which properties of ``label`` are stored as JSON strings
        (``columns`` are ``"<name> <Ladybug type>"``), to decode them on read."""
        self._json_properties[label] = {
            column.split(" ", 1)[0] for column in columns if column.endswith(" JSON")
        }

    def _create_node_table(self, label: str, schema: Dict[str, Any]) -> None:
        """Register ``label``: its schema, a uniqueness constraint on its
        primary key and, with an embedding model, its vector index.
        Idempotent."""
        self._node_labels.add(label)
        self._label_schemas[label] = schema
        self._register_json_properties(label, self._json_schema_to_cypher_columns(schema))
        self._con.execute(
            f"CREATE CONSTRAINT {label.lower()}_pk IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{self._pk_keys[label]} IS UNIQUE"
        )
        if self.embedding_model and self.vector_dim:
            self._create_vector_index(label)

    def _create_vector_index(self, label: str) -> None:
        self._con.execute(
            f"CREATE VECTOR INDEX {label.lower()}_vec IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.embedding) OPTIONS {{indexConfig: {{"
            f"`vector.dimensions`: {int(self.vector_dim)}, "
            f"`vector.similarity_function`: '{METRICS[self.metric]}', "
            # Exact vectors, as Ladybug's HNSW index keeps them: Relate's
            # default binary quantization can miss true nearest neighbours.
            f"`vector.quantization.enabled`: false}}}}"
        )

    def _create_fts_index(self, label: str, columns: List[str]) -> None:
        options = ""
        if self.analyzer is not None:
            analyzer = self.analyzer.replace("'", "")
            options = f" OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}}"
        fields = ", ".join(f"n.{column}" for column in columns)
        try:
            self._con.execute(
                f"CREATE FULLTEXT INDEX {label.lower()}_fts IF NOT EXISTS "
                f"FOR (n:{label}) ON EACH [{fields}]{options}"
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Failed to create FTS index for {label!r}: {e}. "
                f"Full-text search on {label!r} will return no results."
            )

    def _drop_fts_index(self, label: str) -> None:
        self._con.execute(f"DROP INDEX {label.lower()}_fts IF EXISTS")

    def _create_rel_table(self, label: str, schema: Dict[str, Any]) -> None:
        """Register relation ``label`` between the labels its ``subj`` /
        ``obj`` reference."""
        defs = schema.get("$defs", {})
        props = schema.get("properties", {})

        def endpoint(field_name: str) -> str:
            ref = props.get(field_name, {}).get("$ref", "")
            target = ref.rsplit("/", 1)[-1] if ref else None
            if not target or target not in defs:
                raise ValueError(
                    f"Relation schema {schema.get('title')!r} field "
                    f"{field_name!r} must reference a node entity via $ref"
                )
            return sanitize_label(defs[target]["title"])

        self._register_relation(label, endpoint("subj"), endpoint("obj"), schema)

    def _ensure_rel_table(self, rel_label, subj_label, obj_label, relation) -> None:
        if rel_label in self._rel_endpoints:
            return
        try:
            schema = relation.get_schema()
        except Exception:  # noqa: BLE001
            schema = {}
        self._register_relation(rel_label, subj_label, obj_label, schema)

    def _register_relation(self, label, subj_label, obj_label, schema) -> None:
        self._rel_endpoints[label] = (subj_label, obj_label)
        self._rel_schemas[label] = schema
        try:
            columns = self._json_schema_to_cypher_columns(
                schema, skip=("subj", "obj", "label")
            )
        except Exception:  # noqa: BLE001
            columns = []
        self._register_json_properties(label, columns)

    def _ensure_node_table(self, label: str, entity: Any) -> None:
        """As `LadybugAdapter._ensure_node_table`, but registering ``label``
        even when nodes already carry it: on Relate a label exists with its
        first node, constraint and indexes or not."""
        if label in self._pk_keys:
            return
        schema = entity.get_schema()
        self._pk_keys[label] = self._get_id_key(schema)
        self._create_node_table(label, schema)
        string_cols = self._string_columns_from_schema(schema)
        if string_cols:
            self._fts_columns[label] = string_cols
            self._create_fts_index(label, string_cols)

    def _existing_tables(self, kind: str) -> set:
        """The labels (``"NODE"``) or relation types (``"REL"``) the adapter
        registered, plus those the store holds at least one of."""
        if kind == "NODE":
            names = self._con.execute("CALL db.labels() YIELD label RETURN label")
            registered, probe = self._node_labels, "MATCH (n:{0}) RETURN 1 LIMIT 1"
        else:
            names = self._con.execute(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType"
            )
            registered, probe = (
                set(self._rel_endpoints),
                "MATCH ()-[r:{0}]->() RETURN 1 LIMIT 1",
            )
        stored = {
            name
            for (name,) in names.get_all()
            if IDENTIFIER_RE.match(name)
            and self._con.execute(probe.format(name)).get_all()
        }
        return stored | registered

    def _resolve_endpoint_labels(self, rel_label: str) -> tuple:
        """``(subj_label, obj_label)`` of relation ``rel_label``: as registered,
        else read off a stored edge."""
        if rel_label in self._rel_endpoints:
            return self._rel_endpoints[rel_label]
        if IDENTIFIER_RE.match(rel_label):
            rows = self._con.execute(
                f"MATCH (s)-[r:{rel_label}]->(o) "
                f"RETURN labels(s)[0] AS s, labels(o)[0] AS o LIMIT 1"
            ).get_all()
            if rows:
                self._rel_endpoints[rel_label] = tuple(rows[0])
                return self._rel_endpoints[rel_label]
        raise ValueError(
            f"Relation label {rel_label!r} is not a known relation: pass it via "
            f"`relation_models` at construction time, or store one of its edges."
        )

    def wipe_database(self) -> None:
        """Delete every node and edge, constraint and index, and forget the
        registered schema."""
        self._con.execute("MATCH (n) DETACH DELETE n")
        for (name,) in self._con.execute(
            "SHOW CONSTRAINTS YIELD name RETURN name"
        ).get_all():
            self._con.execute(f"DROP CONSTRAINT `{name}` IF EXISTS")
        for name, kind in self._con.execute(
            "SHOW INDEXES YIELD name, type RETURN name, type"
        ).get_all():
            if kind != "LOOKUP":
                self._con.execute(f"DROP INDEX `{name}` IF EXISTS")
        for registry in (
            "_pk_keys",
            "_fts_columns",
            "_synth_models",
            "_node_labels",
            "_rel_endpoints",
            "_label_schemas",
            "_rel_schemas",
            "_json_properties",
        ):
            getattr(self, registry, {}).clear()

    def _stored_properties(self, procedure: str, key: str, value: str) -> List[str]:
        """``"<name> <Ladybug type>"`` per property Relate holds on the label
        or relation type whose ``key`` column is ``value``."""
        columns = []
        for row in self._con.execute(f"CALL {procedure}()").get_records():
            if row[key] == value and row["propertyName"]:
                types = row["propertyTypes"] or ["STRING"]
                columns.append(f"{row['propertyName']} {_relate_type(types[0])}")
        return columns

    def _node_table_to_json_schema(
        self, label: str, remove_embedding: bool = True
    ) -> Dict[str, Any]:
        """The JSON schema of ``label``: from its registered schema, mapped
        through the Ladybug column types so both adapters describe a model
        alike, else from the properties Relate holds on it."""
        if label in self._label_schemas:
            columns = self._json_schema_to_cypher_columns(self._label_schemas[label])
            if not remove_embedding and self.embedding_model and self.vector_dim:
                columns.append(f"embedding FLOAT[{self.vector_dim}]")
        elif label in self._existing_tables("NODE"):
            columns = self._stored_properties(
                "db.schema.nodeTypeProperties", "nodeType", f":{label}"
            )
        else:
            raise ValueError(f"No node label {label!r} in the graph store.")
        hidden = {"label", self._COMMUNITY_COLUMN, self._RANK_COLUMN}
        if remove_embedding:
            hidden.add("embedding")
        properties: Dict[str, Any] = {
            "label": {
                "const": label,
                "default": label,
                "title": "Label",
                "type": "string",
            }
        }
        for column in columns:
            name, column_type = column.split(" ", 1)
            if name not in hidden:
                properties[name] = _cypher_type_to_json_property(column_type, name)
        return {
            "title": label,
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties.keys()),
        }

    def _rel_table_to_json_schema(
        self, label: str, remove_embedding: bool = True
    ) -> Dict[str, Any]:
        src_label, dst_label = self._resolve_endpoint_labels(label)
        defs = {
            src_label: self._node_table_to_json_schema(src_label, remove_embedding),
        }
        if dst_label != src_label:
            defs[dst_label] = self._node_table_to_json_schema(dst_label, remove_embedding)
        if self._rel_schemas.get(label):
            columns = self._json_schema_to_cypher_columns(
                self._rel_schemas[label], skip=("subj", "obj", "label")
            )
        else:
            columns = self._stored_properties(
                "db.schema.relTypeProperties", "relType", f"[:{label}]"
            )
        properties: Dict[str, Any] = {
            "label": {
                "const": label,
                "default": label,
                "title": "Label",
                "type": "string",
            },
            "subj": {"$ref": f"#/$defs/{src_label}"},
            "obj": {"$ref": f"#/$defs/{dst_label}"},
        }
        for column in columns:
            name, column_type = column.split(" ", 1)
            properties[name] = _cypher_type_to_json_property(column_type, name)
        return {
            "title": label,
            "type": "object",
            "additionalProperties": False,
            "$defs": defs,
            "properties": properties,
            "required": list(properties.keys()),
        }

    def _rename_table(self, kind: str, old_label: str, new_label: str) -> None:
        """Relabel the nodes (or retype the edges) of ``old_label``, moving its
        constraint, indexes and registries to ``new_label``."""
        if kind == "NODE":
            self._con.execute(f"DROP CONSTRAINT {old_label.lower()}_pk IF EXISTS")
            self._con.execute(f"DROP INDEX {old_label.lower()}_vec IF EXISTS")
            self._drop_fts_index(old_label)
            self._con.execute(
                f"MATCH (n:{old_label}) SET n:{new_label} REMOVE n:{old_label}"
            )
            self._node_labels.discard(old_label)
            self._pk_keys[new_label] = self._pk_keys.pop(old_label)
            self._synth_models.pop(old_label, None)
            self._json_properties[new_label] = self._json_properties.pop(old_label, set())
            schema = self._label_schemas.pop(old_label, None)
            if schema is not None:
                self._create_node_table(new_label, {**schema, "title": new_label})
            else:
                self._node_labels.add(new_label)
            self._rel_endpoints = {
                rel: tuple(new_label if end == old_label else end for end in ends)
                for rel, ends in self._rel_endpoints.items()
            }
            self._rebuild_indices(old_label, new_label)
        else:
            self._con.execute(
                f"MATCH (s)-[r:{old_label}]->(o) CREATE (s)-[n:{new_label}]->(o) "
                f"SET n = properties(r) DELETE r"
            )
            self._rel_endpoints[new_label] = self._rel_endpoints.pop(old_label)
            self._rel_schemas[new_label] = self._rel_schemas.pop(old_label, {})
            self._json_properties[new_label] = self._json_properties.pop(old_label, set())

    # ------------------------------------------------------------------
    # Index calls (the hooks every search and upsert is built on)
    # ------------------------------------------------------------------

    def _vector_call(
        self,
        label: str,
        vector: str,
        k: str,
        kwargs_fragment: str = "",
        node: str = "node",
        distance: str = "distance",
    ) -> str:
        """Relate's vector index call, its score turned into Ladybug's distance.

        Relate scores like Neo4j: ``(1 + cos) / 2`` for cosine and
        ``1 / (1 + d²)`` for Euclidean. ``kwargs_fragment`` (Ladybug's HNSW
        ``efs``) has no Relate equivalent and is dropped."""
        score = f"__score_{node}"
        if self.metric == "cosine":
            to_distance = f"2.0 - 2.0 * {score}"
        elif self.metric == "l2sq":
            to_distance = f"1.0 / {score} - 1.0"
        else:
            to_distance = f"sqrt(1.0 / {score} - 1.0)"
        return (
            f"CALL db.index.vector.queryNodes('{label.lower()}_vec', {k}, {vector}) "
            f"YIELD node AS {node}, score AS {score} "
            f"WITH *, {to_distance} AS {distance}"
        )

    def _fulltext_call(
        self, label: str, query: str, k: str, kwargs_fragment: str = ""
    ) -> str:
        """Relate's full-text index call, best ``k`` first. ``kwargs_fragment``
        (Ladybug's ``conjunctive`` / ``b``) is dropped: ``conjunctive`` is in
        the query text (`_fulltext_query_text`), BM25 ``b`` is fixed."""
        return (
            f"CALL db.index.fulltext.queryNodes('{label.lower()}_fts', {query}) "
            f"YIELD node, score WITH node, score ORDER BY score DESC LIMIT {k}"
        )

    def _fulltext_query_text(self, text: str, conjunctive: bool) -> str:
        """``text`` as a Lucene query of plain terms: syntax escaped, and the
        operator words lower-cased so they are terms too. ``conjunctive``
        requires every term."""
        terms = [
            (
                term.lower()
                if term in ("AND", "OR", "NOT")
                else _LUCENE_SPECIAL_RE.sub(r"\\\1", term)
            )
            for term in text.split()
        ]
        return (" AND " if conjunctive else " ").join(terms)

    # ------------------------------------------------------------------
    # Cypher
    # ------------------------------------------------------------------

    async def cypher(
        self,
        query: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        read_only: bool = True,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], str]:
        """Execute a Cypher query (see `LadybugAdapter.cypher`). With
        ``read_only=True`` (default) a ``CALL`` must also name a procedure
        Relate runs in ``READ`` mode. ``params`` are bound as given."""
        if read_only:
            _assert_read_only_cypher(query)
            for procedure in _CALL_RE.findall(query):
                if procedure not in self._read_procedures:
                    raise ValueError(
                        f"read_only=True rejected Cypher query: procedure "
                        f"{procedure!r} is not a read-only Relate procedure."
                    )
        result = self._con.execute(query, params, encode=False)
        records = [dict(zip(result.get_column_names(), row)) for row in result.get_all()]
        return format_search_results(records, output_format)

    # ------------------------------------------------------------------
    # Graph algorithms (none in Relate: networkx over the projection)
    # ------------------------------------------------------------------

    def _projection(self, node_labels: List[str], rel_labels: List[str]) -> tuple:
        """``({node id: node}, [(source id, target id), ...])`` over the nodes
        of ``node_labels`` and the edges of ``rel_labels`` between them."""
        nodes: Dict[int, Dict[str, Any]] = {}
        for label in node_labels:
            for (node,) in self._con.execute(f"MATCH (n:{label}) RETURN n").get_all():
                nodes[node["_ID"]["offset"]] = node
        edges = []
        for rel_label in rel_labels:
            for source, target in self._con.execute(
                f"MATCH (s)-[r:{rel_label}]->(o) RETURN id(s), id(o)"
            ).get_all():
                if source in nodes and target in nodes:
                    edges.append((source, target))
        return nodes, edges

    def _community_rows(self, algorithm, node_labels, rel_labels, max_iterations):
        """As Ladybug's ``algo`` extension reports them: Louvain over the
        undirected graph (``max_iterations`` caps its levels), an isolated
        node in community ``-1``; components numbered by their first node."""
        nx = self._networkx
        nodes, edges = self._projection(node_labels, rel_labels)
        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(nodes))
        graph.add_edges_from(edges)
        if algorithm == "louvain":
            undirected = nx.Graph()
            undirected.add_nodes_from(graph)
            for source, target in edges:  # parallel edges weigh more
                weight = undirected.get_edge_data(source, target, {}).get("weight", 0)
                undirected.add_edge(source, target, weight=weight + 1)
            groups = nx.community.louvain_communities(
                undirected, max_level=max_iterations, seed=0
            )
        elif algorithm == "weakly_connected_components":
            groups = nx.weakly_connected_components(graph)
        else:
            groups = nx.strongly_connected_components(graph)
        community = {}
        for number, group in enumerate(sorted(groups, key=min)):
            for node_id in group:
                community[node_id] = number
        if algorithm == "louvain":
            community.update({n: -1 for n in graph if graph.degree(n) == 0})
        return [(nodes[node_id], community[node_id]) for node_id in sorted(nodes)]

    def _pagerank_rows(
        self,
        node_labels,
        rel_labels,
        *,
        damping_factor,
        max_iterations,
        tolerance,
        normalize_initial,
    ):
        """networkx PageRank; ranks sum to 1 whatever ``normalize_initial``.
        ``tolerance`` bounds the L1 change between iterations, as in Ladybug.
        When ``max_iterations`` ends before convergence, the ranks are those
        of a single iteration, with a warning."""
        nx = self._networkx
        nodes, edges = self._projection(node_labels, rel_labels)
        if not nodes:
            return []
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(sorted(nodes))
        graph.add_edges_from(edges)
        # networkx stops when the L1 change is below n * tol.
        tol = 1e-6 if tolerance is None else tolerance / len(nodes)
        try:
            rank = nx.pagerank(
                graph, alpha=damping_factor, max_iter=max_iterations, tol=tol
            )
        except nx.PowerIterationFailedConvergence:
            warnings.warn(
                f"PageRank did not converge within {max_iterations} iterations; "
                f"returning the ranks of one iteration."
            )
            rank = nx.pagerank(graph, alpha=damping_factor, max_iter=1, tol=float("inf"))
        return [(nodes[node_id], rank[node_id]) for node_id in sorted(nodes)]

    def _node_columns(self, label: str) -> set:
        return {
            key
            for (key,) in self._con.execute(
                f"MATCH (n:{label}) UNWIND keys(n) AS key RETURN DISTINCT key"
            ).get_all()
        }

    def _add_stamp_columns(self, label: str) -> None:
        """Nothing to add: Relate properties need no column."""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Relate connection. Idempotent."""
        try:
            self._db.close()
        except Exception:  # noqa: BLE001
            pass

    def __repr__(self):
        return f"<RelateAdapter uri={self.uri}>"


class _RelateConnection:
    """Relate's connection, shaped like Ladybug's so the inherited code runs
    unchanged: ``execute()`` returns a result with ``get_all()`` /
    ``get_column_names()``, nodes and edges come back as Ladybug's structs
    (properties plus ``_LABEL`` / ``_ID``, and ``_SRC`` / ``_DST`` for edges),
    and JSON-typed properties are encoded on write, decoded on read."""

    def __init__(self, adapter: RelateAdapter):
        self._adapter = adapter

    def execute(
        self, query: str, params: Optional[Dict[str, Any]] = None, *, encode=True
    ):
        bound = dict(params or {})
        if encode:
            bound = {key: _encoded(value) for key, value in bound.items()}
        cursor = self._adapter._db.execute(query, bound)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        records = cursor.fetchall() if columns else []
        return _RelateResult(columns, records, self._value)

    def close(self) -> None:
        self._adapter._db.close()

    def _value(self, value: Any) -> Any:
        relate = self._adapter._relate
        if isinstance(value, relate.Node):
            label = value.labels[0] if value.labels else None
            return {
                **self._decoded(label, value.properties),
                "_LABEL": label,
                "_ID": {"offset": value.id, "table": 0},
            }
        if isinstance(value, relate.Relationship):
            return {
                **self._decoded(value.type, value.properties),
                "_LABEL": value.type,
                "_ID": {"offset": value.id, "table": 1},
                "_SRC": {"offset": value.start_node, "table": 0},
                "_DST": {"offset": value.end_node, "table": 0},
            }
        if isinstance(value, list):
            return [self._value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._value(item) for key, item in value.items()}
        return value

    def _decoded(
        self, label: Optional[str], properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        json_properties = self._adapter._json_properties.get(label, ())
        decoded = dict(properties)
        for key in json_properties:
            if isinstance(decoded.get(key), str):
                try:
                    decoded[key] = json.loads(decoded[key])
                except ValueError:
                    pass
        return decoded


class _RelateResult:
    """A fetched result, with Ladybug's ``QueryResult`` accessors."""

    def __init__(self, columns: List[str], records: List[Any], convert):
        self._columns = columns
        self._records = records
        self._convert = convert

    def get_column_names(self) -> List[str]:
        return list(self._columns)

    def get_all(self) -> List[List[Any]]:
        return [
            [self._convert(record[c]) for c in self._columns] for record in self._records
        ]

    def get_records(self) -> List[Dict[str, Any]]:
        return [
            {c: self._convert(record[c]) for c in self._columns}
            for record in self._records
        ]

    def close(self) -> None:
        pass


def _encoded(value: Any) -> Any:
    """A parameter value Relate can store: a map, or a list holding maps or
    lists, as its JSON text (Relate stores neither)."""
    if isinstance(value, dict) or (
        isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value)
    ):
        return json.dumps(value)
    return value
