# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""LadybugDB graph adapter.

First-cut implementation covering the methods needed for a working
GraphRAG agent: ``update_entities`` and ``update_relations`` with
embedding-based deduplication, raw ``cypher()`` execution with a
read-only enforcement layer, and ``entity_similarity_search`` over the
vector index. The remaining ``GraphDatabaseAdapter`` methods (get /
delete / fulltext) fall through to the base's ``NotImplementedError``.

Security model mirrors `DuckDBAdapter`:

  * ``cypher(query, ..., read_only=True)`` (default) rejects any query
    whose tokens include a Cypher write/admin keyword
    (``CREATE`` / ``MERGE`` / ``DELETE`` / ``DETACH`` / ``SET`` /
    ``REMOVE`` / ``DROP`` / ``ALTER`` / ``COPY`` / ``INSTALL`` /
    ``LOAD``). This is the parser-layer analog of DuckDB's
    ``stmt.type == SELECT`` check — it blocks both write injection
    and ``COPY ... TO '/path'`` file exfiltration through an
    otherwise-legitimate read query.

  * Labels and property names are passed through
    `sanitize_label` / `sanitize_properties` before being
    interpolated into Cypher (Cypher doesn't support binding identifiers
    as parameters). Values always go through ``$``-parameter binding,
    never string interpolation.

Deduplication model:

  * ``update_entities`` looks up nearest-neighbour entities by
    embedding via ``QUERY_VECTOR_INDEX``. If the best match is closer
    than ``1 - threshold`` (cosine), the existing node id is returned
    and no INSERT is performed.

  * ``update_relations`` resolves ``subj`` and ``obj`` by the same
    nearest-neighbour rule, then ``MERGE``-s the edge so the same
    (s, label, o) triple is never inserted twice.

References:
    * LadybugDB documentation: https://docs.ladybugdb.com
    * ``algo`` extension (e.g. Louvain community detection):
      https://docs.ladybugdb.com/extensions/algo/louvain/
"""

import re
import uuid
import warnings
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import ladybug as lb
from pydantic import create_model

from synalinks.src.backend import EmbeddingRequest
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_entity
from synalinks.src.backend import is_relation
from synalinks.src.backend.pydantic.knowledge import Entity
from synalinks.src.backend.pydantic.knowledge import KnowledgeGraph
from synalinks.src.backend.pydantic.knowledge import KnowledgeGraphs
from synalinks.src.backend.pydantic.knowledge import Relation
from synalinks.src.knowledge_bases.adapters_utils import align_keywords
from synalinks.src.knowledge_bases.adapters_utils import format_search_results
from synalinks.src.knowledge_bases.adapters_utils import resolve_db_path
from synalinks.src.knowledge_bases.adapters_utils import to_pascal_identifier
from synalinks.src.knowledge_bases.adapters_utils import to_snake_identifier
from synalinks.src.knowledge_bases.graph_database_adapters.graph_database_adapter import (
    GraphDatabaseAdapter,
)
from synalinks.src.modules.embedding_models import get as _get_em

METRICS = ("cosine", "l2sq", "l2", "dotproduct")

# Word-boundary match of any Cypher write / admin keyword. Conservative
# on purpose — it also fires inside string literals, but read_only
# queries never legitimately need these words anyway, and the false
# positive is a clearer error message than a successful write.
_CYPHER_WRITE_RE = re.compile(
    r"\b(?:CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|COPY|INSTALL|LOAD"
    r"|CALL\s+CREATE_|CALL\s+DROP_)\b",
    re.IGNORECASE,
)

# Strip Cypher line and block comments before scanning for write
# keywords — otherwise `// SET foo` would falsely look like a write.
_CYPHER_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)

# Strip Cypher string literals before scanning for write keywords —
# otherwise `WHERE n.text = 'SET clause'` would falsely look like a
# write. Handles both single- and double-quoted strings with embedded
# backslash-escaped quotes. Cypher doesn't support raw or triple-quoted
# strings so this two-variant pattern is sufficient.
_CYPHER_STRING_RE = re.compile(
    r"'(?:\\.|[^'\\])*'" r"|" r'"(?:\\.|[^"\\])*"',
    re.DOTALL,
)

# Labels and property names must look like identifiers because we
# interpolate them into Cypher. Verified against Ladybug 0.16.1:
#
#   * Pattern-position labels (``MATCH (n:$label)``, ``[r:$rel]``)
#     fail at parse time — the grammar requires a literal token there.
#   * Catalog procs (``CALL TABLE_INFO``, ``SHOW_CONNECTION``,
#     ``CREATE_VECTOR_INDEX``) reject ``$``-parameters at the binder
#     with "PARAMETER but LITERAL was expected".
#   * Only ``WHERE label(n) = $l`` accepts a bound label, and that's
#     a runtime filter — useless for ``CREATE NODE TABLE`` / index DDL
#     / catalog introspection where we actually need labels.
#
# So interpolation through the shared identifier helpers is mandatory
# in every code path that touches a label or property name. The
# Cypher-flavored aliases below name the gate at each call site.
sanitize_label = to_pascal_identifier


def sanitize_property_name(name: str) -> Optional[str]:
    """Warn-and-drop variant of `to_snake_identifier` for property names.

    Property names arrive from external entity/relation payloads with
    arbitrary keys, so the graph adapter needs a non-raising path: an
    un-normalizable key is dropped at the call site rather than aborting
    the whole upsert. SQL columns use the raising variant directly.
    """
    return to_snake_identifier(name, strict=False)


def sanitize_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """Filter and normalize a property dict.

    Property *values* are bound via ``$``-parameters and don't need
    sanitization, but property *names* are interpolated into Cypher
    so they go through `sanitize_property_name`
    (``snake_case`` normalization + identifier validation). The
    ``label`` / ``subj`` / ``obj`` reserved fields are dropped —
    ``label`` is already captured by the table name; ``subj`` / ``obj``
    are how a ``Relation`` encodes its endpoints, not edge properties.
    """
    out: Dict[str, Any] = {}
    for key, value in props.items():
        if key in ("label", "subj", "obj"):
            continue
        normalized = sanitize_property_name(key)
        if normalized is None:
            warnings.warn(f"Dropping property with un-normalizable name {key!r}")
            continue
        out[normalized] = value
    return out


def _strip_comments_and_strings(query: str) -> str:
    """Remove comments and string literals from a Cypher query.

    Order matters: we strip strings first, then comments. Doing the
    reverse would let a quoted ``// CREATE`` inside a string sneak
    through after the comment strip. We replace each match with a
    single space so token boundaries don't accidentally fuse — e.g.
    ``WHERE'CREATE'AND`` after a naive empty-string replacement
    would look like ``WHERECREATAND`` rather than three separate
    tokens. The whitespace keeps the keyword regex's ``\\b``
    boundaries honest.
    """
    no_strings = _CYPHER_STRING_RE.sub(" ", query)
    return _CYPHER_COMMENT_RE.sub(" ", no_strings)


def _assert_read_only_cypher(query: str) -> None:
    """Reject a Cypher string that contains any write/admin keyword.

    The parser-layer analog of DuckDB's ``stmt.type != SELECT`` check.
    Cypher doesn't expose a typed AST through Ladybug's Python bindings,
    so this is a token-level scan with two pre-strips:

      * String literals (``'...'`` / ``"..."``) — without this, a
        legitimate ``WHERE n.label = 'CREATE'`` would falsely look
        like a write.
      * Line and block comments — without this, ``// SET foo`` would
        falsely fire.

    Raises:
        ValueError: when a write keyword is present.
    """
    body = _strip_comments_and_strings(query)
    match = _CYPHER_WRITE_RE.search(body)
    if match:
        raise ValueError(
            f"read_only=True rejected Cypher query: contains write/admin "
            f"keyword {match.group(0)!r}. Pass read_only=False to allow "
            f"mutations (only from trusted call sites)."
        )


def _resolve_ref(prop_spec: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    """Inline a ``$ref`` into the property spec.

    Pydantic v2 emits ``{"$ref": "#/$defs/Foo"}`` for nested models /
    enums; the schema-to-column converter needs the resolved object,
    not the indirection. Overrides on the referring spec (e.g.
    ``default``) take precedence over the resolved definition.
    """
    ref = prop_spec.get("$ref")
    if not ref:
        return prop_spec
    target = ref.rsplit("/", 1)[-1]
    resolved = dict(defs.get(target, {}))
    resolved.update({k: v for k, v in prop_spec.items() if k != "$ref"})
    return resolved


def _pick_anyof_variant(
    prop_spec: Dict[str, Any], defs: Dict[str, Any]
) -> Dict[str, Any]:
    """Pick the first non-null variant from an ``anyOf`` union.

    Optional fields in Pydantic v2 render as
    ``anyOf: [{"type": "..."}, {"type": "null"}]``; the column
    converter needs the non-null half. Enum-only variants get a
    synthetic ``"type": "string"``.
    """
    for variant in prop_spec.get("anyOf", []):
        variant = _resolve_ref(variant, defs)
        vtype = variant.get("type")
        if vtype and vtype != "null":
            return variant
        if "enum" in variant:
            return {**variant, "type": "string"}
    return prop_spec


def _map_prop_to_cypher(prop_type: str, prop_spec: Dict[str, Any]) -> str:
    """Map a resolved JSON-schema property to a Ladybug column type.

    Type vocabulary picked from a probe of Ladybug 0.16:

      * ``STRING``, ``INT64``, ``DOUBLE``, ``BOOL`` — scalars.
      * ``DATE`` / ``TIMESTAMP`` — string properties with the matching
        ``format`` annotation. (Ladybug exposes ``INTERVAL`` but no
        ``TIME``, so ``format=time`` falls back to ``STRING``.)
      * ``STRING[]`` / ``DOUBLE[]`` / ``INT64[]`` / ``BOOL[]`` —
        variable-length lists of primitives. The embedding vector is
        handled separately (fixed-size ``FLOAT[<dim>]``) outside the
        schema loop, so a generic list of numbers maps to
        ``DOUBLE[]``.
      * ``JSON`` — fallback for nested objects and lists of objects.
    """
    if prop_type == "array":
        item_spec = prop_spec.get("items") or {}
        item_type = item_spec.get("type")
        if item_type == "string":
            return "STRING[]"
        if item_type == "number":
            return "DOUBLE[]"
        if item_type == "integer":
            return "INT64[]"
        if item_type == "boolean":
            return "BOOL[]"
        # Lists of objects, mixed/unspecified items → opaque JSON.
        return "JSON"
    if prop_type == "object":
        return "JSON"
    if prop_type == "string":
        fmt = prop_spec.get("format")
        if fmt == "date":
            return "DATE"
        if fmt == "date-time":
            return "TIMESTAMP"
        return "STRING"
    if prop_type == "number":
        return "DOUBLE"
    if prop_type == "integer":
        return "INT64"
    if prop_type == "boolean":
        return "BOOL"
    raise ValueError(f"Unsupported JSON schema type: {prop_type!r}")


# Matches ``ELEM[N]`` (fixed-size, e.g. ``FLOAT[4]``) and ``ELEM[]``
# (variable-length, e.g. ``STRING[]``). The element name is captured so
# the reverse mapper can pick the JSON ``items`` shape.
_ARRAY_TYPE_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)\[(\d*)\]$")

# Ladybug exposes a wider integer family than the forward mapper emits;
# accept all of them on the reverse path so a manually-CREATEd table
# (or a future schema bump) still round-trips through the converter.
_INT_TYPES = {
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "INT128",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "SERIAL",
}
_FLOAT_TYPES = {"FLOAT", "DOUBLE"}


def _cypher_type_to_json_property(cypher_type: str, name: str) -> Dict[str, Any]:
    """Reverse of `_map_prop_to_cypher`.

    Mirror of DuckDB's ``_duckdb_table_to_json_schema`` per-column
    branch: maps a Ladybug type string back to a JSON-schema property
    spec, with the column name title-cased into the ``title`` field
    (same convention DuckDB uses). Fixed-size arrays (``FLOAT[N]``)
    collapse to a plain ``array`` of the element type — JSON Schema
    has no length annotation on ``array`` items, and downstream
    consumers don't need one.

    Raises:
        NotImplementedError: when the type isn't one of the supported
            Ladybug primitives / arrays.
    """
    title = name.title()

    array_match = _ARRAY_TYPE_RE.match(cypher_type)
    if array_match:
        elem = array_match.group(1)
        if elem == "STRING":
            return {"title": title, "type": "array", "items": {"type": "string"}}
        if elem in _FLOAT_TYPES:
            return {"title": title, "type": "array", "items": {"type": "number"}}
        if elem in _INT_TYPES:
            return {"title": title, "type": "array", "items": {"type": "integer"}}
        if elem == "BOOL":
            return {"title": title, "type": "array", "items": {"type": "boolean"}}
        raise NotImplementedError(
            f"Unsupported Ladybug array element {elem!r} in {cypher_type!r}"
        )

    if cypher_type == "STRING":
        return {"title": title, "type": "string"}
    if cypher_type == "DATE":
        return {"title": title, "type": "string", "format": "date"}
    if cypher_type == "TIMESTAMP":
        return {"title": title, "type": "string", "format": "date-time"}
    if cypher_type == "BOOL":
        return {"title": title, "type": "boolean"}
    if cypher_type == "JSON":
        return {"title": title, "type": "object"}
    if cypher_type in _INT_TYPES:
        return {"title": title, "type": "integer"}
    if cypher_type in _FLOAT_TYPES:
        return {"title": title, "type": "number"}
    raise NotImplementedError(
        f"Unsupported Ladybug type {cypher_type!r} for property {name!r}"
    )


class LadybugAdapter(GraphDatabaseAdapter):
    """LadybugDB-backed implementation of ``GraphDatabaseAdapter``."""

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
        mu: Optional[int] = None,
        ml: Optional[int] = None,
        pu: Optional[float] = None,
        efc: Optional[int] = None,
        stemmer: Optional[str] = None,
        stopwords: Optional[str] = None,
        tokenizer: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Ladybug adapter.

        Args:
            uri: ``ladybug://<path>`` (file-backed) or
                ``ladybug://:memory:`` (in-memory). When omitted the
                adapter resolves to ``{synalinks_home()}/{name or
                'database'}.lb`` so a no-arg ``KnowledgeBase()``
                persists the graph store next to the DuckDB file.
            embedding_model: Required for entity dedup and vector
                search. Without it ``update_entities`` falls back to
                always-insert (no semantic dedup) and
                ``entity_similarity_search`` raises.
            entity_models: Pydantic ``Entity`` subclasses. Each one
                becomes a ``NODE TABLE`` whose name is the schema
                ``title``.
            relation_models: Pydantic ``Relation`` subclasses. Each
                one becomes a ``REL TABLE`` whose endpoints are the
                node tables corresponding to ``subj`` / ``obj``.
            metric: Vector index distance metric. One of
                ``"cosine"``, ``"l2sq"``, ``"l2"``, ``"dotproduct"``.
            vector_dim: Embedding dimension. Inferred from the
                embedding model when omitted.
            dedup_threshold: Cosine-similarity threshold above which an
                incoming entity is considered a duplicate of the best
                vector-search match. ``1.0`` disables dedup;
                ``0.0`` collapses all entities of a label into one.
            wipe_on_start: Whether to clear the database on init.
            name: Optional adapter name.
            mu: HNSW upper-layer max degree (Ladybug's
                ``CREATE_VECTOR_INDEX`` ``mu`` arg). ``None`` defers
                to the engine default.
            ml: HNSW lower-layer max degree (Ladybug's ``ml`` arg).
            pu: HNSW upper-layer probability — controls how often a
                node lands in the sparser upper layer (Ladybug's
                ``pu`` arg).
            efc: HNSW build-time candidate-list depth (Ladybug's
                ``efc`` arg — the construction-time analogue of
                ``ef_search``). Higher = better index quality at
                slower build time.
            stemmer: FTS index stemmer (Ladybug's ``stemmer`` arg —
                e.g. ``"porter"``, ``"english"``, ``"none"``).
            stopwords: FTS index stopwords source (Ladybug's
                ``stopwords`` arg — accepts a node-table name or a
                file path with one stopword per line).
            tokenizer: FTS index tokenizer (Ladybug's ``tokenizer``
                arg — ``"simple"`` (default) or ``"jieba"`` for
                Chinese).
        """
        if metric not in METRICS:
            raise ValueError(f"`metric` must be one of {METRICS}, got {metric!r}")

        # Resolve URI → backing path. ``None`` defaults to
        # ``{synalinks_home()}/{name or 'database'}.lb`` so a no-args
        # KnowledgeBase() persists the graph store next to the SQL
        # one. Explicit ``ladybug://:memory:`` still selects the
        # in-memory backend for tests / ephemeral use.
        self.uri = resolve_db_path(uri, scheme="ladybug", extension="lb", name=name)

        self.embedding_model = _get_em(embedding_model)
        self.entity_models = entity_models or []
        self.relation_models = relation_models or []
        self.metric = metric
        self.dedup_threshold = dedup_threshold
        self.name = name
        # HNSW build params — only the ones the user supplied get
        # forwarded into ``CREATE_VECTOR_INDEX``; ``None`` defers to
        # Ladybug's own default. Kept as attributes (not a dict) so
        # the surface matches the per-method param style and the
        # values survive ``get_config`` round-trips.
        self.mu = mu
        self.ml = ml
        self.pu = pu
        self.efc = efc
        # FTS build params — same shape as the HNSW ones above.
        self.stemmer = stemmer
        self.stopwords = stopwords
        self.tokenizer = tokenizer

        # Vector dimension: resolved lazily on the main loop at first write
        # (see `_ensure_vector_dim`) when an embedding model is configured
        # without an explicit `vector_dim`. The old code probed the dimension
        # here via `run_maybe_nested(self.embedding_model(...))`, which ran the
        # embedding on a transient thread-loop and bound litellm's process-global
        # httpx client to a loop closed moments later — poisoning that client for
        # every subsequent main-loop embedding ("Event loop is closed" noise).
        self.vector_dim = vector_dim or 0
        # The `FLOAT[dim]` embedding column needs the dimension up front, so when
        # we don't yet know it, defer creating the declared tables until the
        # first write (where `_ensure_vector_dim` learns the dim on the main loop
        # and `_ensure_node_table` / `_ensure_rel_table` build them on demand).
        # Only this exact case is deferred; with no embedding model or an
        # explicit `vector_dim`, tables are still created eagerly below.
        self._defer_table_creation = bool(self.embedding_model and not vector_dim)

        self._db = lb.Database(self.uri)
        self._con = lb.Connection(self._db)

        # Install / load extensions used by the indices below. Failures
        # surface as a warning so the adapter still opens for users who
        # only need bare Cypher.
        for ext in ("vector", "fts", "algo"):
            try:
                self._con.execute(f"INSTALL {ext}")
                self._con.execute(f"LOAD {ext}")
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Ladybug extension {ext!r} unavailable: {e}. "
                    f"{ext} queries will fail until the extension is loaded."
                )

        if wipe_on_start:
            self.wipe_database()

        self._setup_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _setup_schema(self) -> None:
        """Create NODE / REL tables and their indices from the model lists."""
        existing_node = self._existing_tables("NODE")
        existing_rel = self._existing_tables("REL")

        # Per-label record of which columns the FTS index covers, so
        # ``_rebuild_fts_index`` (called after every ``update_entities``
        # batch) can drop + recreate the index against the same
        # columns. Ladybug's FTS index is a snapshot built at
        # CREATE_FTS_INDEX time, so it must be rebuilt after inserts
        # — same pattern DuckDB uses with PRAGMA create_fts_index +
        # overwrite=1 at the end of every ``update``.
        self._fts_columns: Dict[str, List[str]] = {}

        # Per-label record of the primary-key column name. ``label`` is
        # implicit (it's the table name) so it never lands in the
        # column list; the PK is the first remaining schema property,
        # same convention DuckDB uses via ``_get_id_key``. Cached at
        # schema-setup time so ``_upsert_entity`` doesn't have to
        # rediscover it on every call.
        self._pk_keys: Dict[str, str] = {}

        # Cache of Pydantic models synthesized for free-form labels that
        # have no matching ``entity_models`` class, so read-back keeps
        # their properties instead of collapsing to the bare ``Entity``.
        self._synth_models: Dict[str, Any] = {}

        if not self._defer_table_creation:
            for model in self.entity_models:
                schema = model.get_schema()
                label = sanitize_label(schema["title"])
                self._pk_keys[label] = self._get_id_key(schema)
                if label not in existing_node:
                    self._create_node_table(label, schema)
                string_cols = self._string_columns_from_schema(schema)
                if string_cols:
                    self._fts_columns[label] = string_cols
                    self._create_fts_index(label, string_cols)

            for model in self.relation_models:
                schema = model.get_schema()
                label = sanitize_label(schema["title"])
                if label not in existing_rel:
                    self._create_rel_table(label, schema)

    def _get_id_key(self, schema: Dict[str, Any]) -> str:
        """Return the schema's first non-``label`` property as PK.

        Mirror of `DuckDBAdapter._get_id_key` with one extra
        rule: the ``label`` field is metadata used to choose the
        node/rel table name and is never stored as a column, so it
        doesn't count when picking the PK. The next property in
        declaration order wins, after running through
        `sanitize_property_name` so the returned name matches
        the actual table column. Inputs that normalize away (empty
        / leading digit / nothing but separators) are skipped — they
        wouldn't make a valid identifier either.
        """
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not properties:
            raise ValueError(
                f"Cannot determine primary key: schema "
                f"{schema.get('title')!r} has no `properties`."
            )
        for name in properties:
            # ``label`` is the table name and ``embedding`` is the
            # adapter-managed vector column — neither is a PK candidate.
            # Skipping ``embedding`` matters when the schema is inferred
            # from an already-embedded entity instance (dynamic table
            # creation), where it can appear before the natural key.
            if name in ("label", "embedding"):
                continue
            normalized = sanitize_property_name(name)
            if normalized is None:
                continue
            return normalized
        raise ValueError(
            f"Cannot determine primary key: schema {schema.get('title')!r} "
            f"has no non-`label` property usable as an identifier."
        )

    def _string_columns_from_schema(self, schema: Dict[str, Any]) -> List[str]:
        """Return the normalized names of string-typed properties.

        These are the columns covered by the FTS index. ``label`` is
        metadata (it's the table name) and ``embedding`` is the vector
        column — both are excluded. The PK column IS included when
        it's STRING-typed, unlike DuckDB which excludes it: in a
        graph entity the PK is often the natural name of the thing
        (``name="Alice"``) and users expect to be able to FTS on it.
        """
        cols: List[str] = []
        for field_name, field_schema in schema.get("properties", {}).items():
            if field_name in ("label", "embedding"):
                continue
            if field_schema.get("type") != "string":
                continue
            normalized = sanitize_property_name(field_name)
            if normalized is None:
                continue
            cols.append(normalized)
        return cols

    @staticmethod
    def _render_ddl_option_value(value: Any) -> str:
        """Render a Python value as a Cypher literal for DDL options.

        ``CREATE_VECTOR_INDEX`` / ``CREATE_FTS_INDEX`` accept their
        optional kwargs as inline literals, not bound parameters
        (Ladybug DDL doesn't substitute ``$`` parameters at DDL
        time). We render bool / numeric directly and quote-escape
        strings — the keys themselves are gated by the
        ``_*_SUPPORTED_PARAMS`` allowlists, so the only thing we
        have to defend on the value side is single-quote escaping
        inside strings.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        raise ValueError(
            f"Unsupported DDL option value of type {type(value).__name__}: {value!r}"
        )

    def _render_ddl_options(self, options: Dict[str, Any]) -> str:
        """Render an options dict as the ``, k1 := <lit>, ...`` fragment
        appended to a Ladybug DDL ``CALL`` (empty string when the
        dict is empty so the caller can interpolate unconditionally).
        """
        if not options:
            return ""
        return ", " + ", ".join(
            f"{k} := {self._render_ddl_option_value(v)}" for k, v in options.items()
        )

    def _create_fts_index(self, label: str, columns: List[str]) -> None:
        """Build the FTS index for a label over the given columns.

        Ladybug's FTS index is a snapshot of the table at creation
        time — inserts after this point don't show up until the
        index is rebuilt. `update_entities` triggers that
        rebuild via `_rebuild_fts_index` at write time, so
        search paths assume the index is current.

        Optional build params (``stemmer`` / ``stopwords`` /
        ``tokenizer``) come from the matching ``self.<name>``
        attributes set at construction time — same options applied
        to every label's index so the surface stays uniform across
        the graph.
        """
        col_list = ", ".join(f"'{c}'" for c in columns)
        options_fragment = self._render_ddl_options(
            {
                k: v
                for k, v in (
                    ("stemmer", self.stemmer),
                    ("stopwords", self.stopwords),
                    ("tokenizer", self.tokenizer),
                )
                if v is not None
            }
        )
        try:
            self._con.execute(
                f"CALL CREATE_FTS_INDEX('{label}', '{label.lower()}_fts', "
                f"[{col_list}]{options_fragment})"
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"Failed to create FTS index for {label!r}: {e}. "
                f"Full-text search on {label!r} will return no results."
            )

    def _drop_fts_index(self, label: str) -> None:
        """Drop the FTS index for a label. Idempotent."""
        try:
            self._con.execute(f"CALL DROP_FTS_INDEX('{label}', '{label.lower()}_fts')")
        except Exception:
            pass

    def _rebuild_fts_index(self, label: str) -> None:
        """Drop and recreate the FTS index for a label.

        Called from the write paths (`update_entities`) after the
        underlying inserts have committed. Search paths assume the
        index is current — they never rebuild at query time.
        """
        if label not in self._fts_columns:
            return
        self._drop_fts_index(label)
        self._create_fts_index(label, self._fts_columns[label])

    def _existing_tables(self, kind: str) -> set:
        """Return the set of existing tables of the given kind ('NODE'/'REL')."""
        try:
            rows = self._con.execute(
                f"CALL SHOW_TABLES() WHERE type = '{kind}' RETURN name"
            ).get_all()
        except Exception:
            return set()
        return {row[0] for row in rows}

    def _json_schema_to_cypher_columns(
        self,
        json_schema: Dict[str, Any],
        skip: tuple = ("label", "embedding"),
    ) -> List[str]:
        """Convert a JSON schema's ``properties`` to Ladybug column defs.

        Mirror of DuckDB's ``_json_schema_to_duckdb_columns``: walks
        the property map, resolves ``$ref`` references against the
        schema's ``$defs``, picks the non-null half of an ``anyOf``
        union, and maps each resolved type via `_map_prop_to_cypher`.
        The reserved fields in ``skip`` are left to the caller —
        ``label`` is implicit (table name), ``embedding`` is added
        back later as ``FLOAT[<dim>]``, and rel tables additionally
        strip ``subj`` / ``obj``.
        """
        properties = json_schema.get("properties", {})
        defs = json_schema.get("$defs", {})
        cols: List[str] = []

        for prop_name, prop_spec in properties.items():
            if prop_name in skip:
                continue
            normalized = sanitize_property_name(prop_name)
            if normalized is None:
                continue

            prop_spec = _resolve_ref(prop_spec, defs)
            prop_type = prop_spec.get("type")
            if not prop_type and "anyOf" in prop_spec:
                prop_spec = _pick_anyof_variant(prop_spec, defs)
                prop_type = prop_spec.get("type")
            if not prop_type and "enum" in prop_spec:
                prop_type = "string"
            if not prop_type:
                raise ValueError(f"Malformed JSON schema: missing type for {prop_name!r}")

            cols.append(f"{normalized} {_map_prop_to_cypher(prop_type, prop_spec)}")
        return cols

    def _create_node_table(self, label: str, schema: Dict[str, Any]) -> None:
        """Issue ``CREATE NODE TABLE`` from a JSON schema.

        The schema's first non-``label`` property becomes the PRIMARY
        KEY — same convention DuckDB uses (the SQL adapter promotes
        the first column to PK in ``_json_schema_to_duckdb_columns``).
        ``label`` is never stored as a column; it's recoverable from
        the table name. When an embedding model is configured an
        ``embedding FLOAT[<dim>]`` column is added and a vector index
        is built over it. Every other schema field flows through
        `_json_schema_to_cypher_columns`.
        """
        pk_col = self._pk_keys[label]
        cols: List[str] = list(self._json_schema_to_cypher_columns(schema))
        if self.embedding_model and self.vector_dim:
            cols.append(f"embedding FLOAT[{self.vector_dim}]")
        cols.append(f"PRIMARY KEY({pk_col})")
        self._con.execute(f"CREATE NODE TABLE {label}({', '.join(cols)})")

        if self.embedding_model and self.vector_dim:
            # ``metric`` is fixed at construction time because it also
            # drives downstream conversion helpers (e.g.
            # `_max_distance_for_threshold`). HNSW build params
            # (``mu`` / ``ml`` / ``pu`` / ``efc``) come from their
            # matching ``self.<name>`` attributes and only get
            # forwarded when set — ``None`` falls back to Ladybug's
            # own defaults.
            ddl_options: Dict[str, Any] = {"metric": self.metric}
            for key in ("mu", "ml", "pu", "efc"):
                value = getattr(self, key)
                if value is not None:
                    ddl_options[key] = value
            options_fragment = self._render_ddl_options(ddl_options)
            self._con.execute(
                f"CALL CREATE_VECTOR_INDEX('{label}', "
                f"'{label.lower()}_vec', 'embedding'{options_fragment})"
            )

    def _create_rel_table(self, label: str, schema: Dict[str, Any]) -> None:
        """Issue ``CREATE REL TABLE``.

        The relation schema must reference its endpoints via nested
        ``subj`` / ``obj`` entities — Ladybug needs an explicit
        ``FROM <SrcLabel> TO <DstLabel>``. We resolve those labels from
        the ``$defs`` referenced by each field; the remaining schema
        properties (skipping ``subj`` / ``obj`` / ``label``) become
        edge attributes via the shared column converter.
        """
        defs = schema.get("$defs", {})
        props = schema.get("properties", {})

        def _endpoint_label(field_name: str) -> str:
            ref = props.get(field_name, {}).get("$ref", "")
            target = ref.rsplit("/", 1)[-1] if ref else None
            if not target or target not in defs:
                raise ValueError(
                    f"Relation schema {schema.get('title')!r} field "
                    f"{field_name!r} must reference a node entity via $ref"
                )
            return sanitize_label(defs[target]["title"])

        src_label = _endpoint_label("subj")
        dst_label = _endpoint_label("obj")

        extra_cols = self._json_schema_to_cypher_columns(
            schema, skip=("subj", "obj", "label")
        )
        clause = f"FROM {src_label} TO {dst_label}"
        if extra_cols:
            clause += ", " + ", ".join(extra_cols)
        self._con.execute(f"CREATE REL TABLE {label}({clause})")

    async def _ensure_vector_dim(self, sample_vector=None):
        """Resolve the embedding dimension lazily, on the current event loop.

        Prefers the length of an embedding vector already in hand — entities
        arrive pre-embedded from ``EmbedKnowledge`` — and only falls back to a
        probe, awaited on *this* loop, when a table must be created before any
        embedded entity is available. Never uses ``run_maybe_nested``: that runs
        the embedding on a transient loop and poisons litellm's global client.
        ``0`` is the "not yet resolved" sentinel. No-op once resolved or when no
        embedding model is configured.
        """
        if self.vector_dim or not self.embedding_model:
            return
        if sample_vector:
            self.vector_dim = len(sample_vector)
            return
        probe = await self.embedding_model(EmbeddingRequest(texts=["text"]))
        embeddings = probe.get("embeddings") if probe is not None else None
        if not embeddings:
            raise ValueError(
                f"Embedding model {self.embedding_model} returned no embeddings "
                "while resolving the vector dimension for the graph store. This "
                "usually means the model name is wrong or unavailable for your "
                "provider/API key. Fix the embedding model, or pass an explicit "
                "`vector_dim=...`."
            )
        self.vector_dim = len(embeddings[0])

    def _ensure_node_table(self, label: str, entity: Any) -> None:
        """Create the NODE table for ``label`` on demand if it's unknown.

        ``entity_models`` declares tables eagerly at construction, but a
        free-form graph carries labels the LM invented — not known ahead
        of time. Rather than require every label up front, the schema is
        inferred from the entity instance the first time a label is seen,
        then cached in ``_pk_keys`` / ``_fts_columns`` exactly like a
        declared model. No-op once the label is registered, so the cost
        is paid once per new label. The data models passed to the
        generator remain the place schema is *constrained*; the store
        just materializes whatever arrives.
        """
        if label in self._pk_keys:
            return
        schema = entity.get_schema()
        self._pk_keys[label] = self._get_id_key(schema)
        if label not in self._existing_tables("NODE"):
            self._create_node_table(label, schema)
        string_cols = self._string_columns_from_schema(schema)
        if string_cols:
            self._fts_columns[label] = string_cols
            self._create_fts_index(label, string_cols)

    def _ensure_rel_table(
        self,
        rel_label: str,
        subj_label: str,
        obj_label: str,
        relation: Any,
    ) -> None:
        """Create the REL table for ``rel_label`` on demand if it's absent.

        The endpoint pair ``FROM subj_label TO obj_label`` comes from the
        relation's actual endpoints, and any non-``subj``/``obj``/``label``
        properties become edge attributes. Endpoint NODE tables must
        already exist — callers ensure them first.

        A Ladybug REL table fixes its endpoint pair at creation, so a
        free-form relation label that connects several distinct
        ``(subj_label, obj_label)`` pairs keeps only the first pair seen;
        edges of the same label between other type pairs won't match. In
        practice a relation type connects consistent endpoint types, so
        this is rarely a constraint.
        """
        if rel_label in self._existing_tables("REL"):
            return
        try:
            extra_cols = self._json_schema_to_cypher_columns(
                relation.get_schema(), skip=("subj", "obj", "label")
            )
        except Exception:  # noqa: BLE001
            extra_cols = []
        clause = f"FROM {subj_label} TO {obj_label}"
        if extra_cols:
            clause += ", " + ", ".join(extra_cols)
        self._con.execute(f"CREATE REL TABLE {rel_label}({clause})")

    def wipe_database(self) -> None:
        """Drop every node and rel table from the current database.

        Indices have to come down first — Ladybug refuses to ``DROP
        TABLE`` while an FTS or vector index references it. So the
        order is: every FTS / vector index, then every REL table (FK
        to nodes), then the NODE tables themselves.
        """
        for table, name, idx_type in self._existing_indices():
            try:
                if idx_type == "FTS":
                    self._con.execute(f"CALL DROP_FTS_INDEX('{table}', '{name}')")
                elif idx_type == "HNSW":
                    self._con.execute(f"CALL DROP_VECTOR_INDEX('{table}', '{name}')")
                else:
                    warnings.warn(
                        f"Skipping {idx_type!r} index {name!r} on {table!r}: "
                        f"no known DROP procedure"
                    )
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Failed to drop {idx_type} index {name!r} on {table!r}: {e}"
                )
        for kind in ("REL", "NODE"):
            for name in self._existing_tables(kind):
                try:
                    self._con.execute(f"DROP TABLE {name}")
                except Exception as e:  # noqa: BLE001
                    warnings.warn(f"Failed to drop table {name!r}: {e}")

    def _existing_indices(self) -> List[tuple]:
        """Return ``[(table_name, index_name, index_type), ...]``."""
        try:
            rows = self._con.execute(
                "CALL SHOW_INDEXES() RETURN table_name, index_name, index_type"
            ).get_all()
        except Exception:
            return []
        return [(r[0], r[1], r[2]) for r in rows]

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _node_table_to_json_schema(
        self,
        label: str,
        remove_embedding: bool = True,
    ) -> Dict[str, Any]:
        """Build a JSON schema for a NODE table by introspection.

        Mirror of DuckDB's ``_duckdb_table_to_json_schema`` for the
        graph side. Walks ``CALL TABLE_INFO(<label>)`` row-by-row,
        maps each Ladybug type through
        `_cypher_type_to_json_property`, and adds a ``label``
        property pinned by ``const`` so the resulting schema acts as
        a discriminated-union member (same shape the user-written
        ``Entity`` subclasses emit via Pydantic v2).

        The ``embedding`` column is hidden by default — same default
        as DuckDB's ``get_symbolic_data_models`` — because the vector
        is internal to the adapter and rarely useful to downstream
        consumers (passing the schema as ``data_models`` to a search
        API, generating an LM prompt, etc.).
        """
        rows = self._con.execute(f"CALL TABLE_INFO('{label}') RETURN *").get_all()

        # const-pinned label discriminator (matches what
        # ``class Foo(Entity): label: Literal["Foo"]`` produces).
        properties: Dict[str, Any] = {
            "label": {
                "const": label,
                "default": label,
                "title": "Label",
                "type": "string",
            }
        }
        for _, col_name, col_type, _, _ in rows:
            if col_name == "label":
                continue
            if col_name == "embedding" and remove_embedding:
                continue
            # community / rank are adapter-internal (stamped by
            # build_communities); hide them like the embedding vector so
            # the schema keeps round-tripping through the user's model.
            if col_name in (self._COMMUNITY_COLUMN, self._RANK_COLUMN):
                continue
            properties[col_name] = _cypher_type_to_json_property(col_type, col_name)

        return {
            "title": label,
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": list(properties.keys()),
        }

    def _rel_table_to_json_schema(
        self,
        label: str,
        remove_embedding: bool = True,
    ) -> Dict[str, Any]:
        """Build a JSON schema for a REL table by introspection.

        Returns a schema with ``$defs`` containing the source /
        destination node schemas, ``subj`` / ``obj`` as ``$ref``
        pointers into those defs, a ``label`` const discriminator,
        and one property per edge attribute. Endpoint resolution
        uses ``CALL SHOW_CONNECTION(<label>)``; properties come
        from ``CALL TABLE_INFO(<label>)`` (which on REL tables
        already excludes the ``FROM``/``TO`` columns).
        """
        connection_rows = self._con.execute(
            f"CALL SHOW_CONNECTION('{label}') RETURN *"
        ).get_all()
        if not connection_rows:
            raise ValueError(
                f"REL table {label!r} has no SHOW_CONNECTION row — cannot "
                f"recover endpoint labels"
            )
        src_label, dst_label = connection_rows[0][0], connection_rows[0][1]

        info_rows = self._con.execute(f"CALL TABLE_INFO('{label}') RETURN *").get_all()

        defs: Dict[str, Any] = {
            src_label: self._node_table_to_json_schema(
                src_label, remove_embedding=remove_embedding
            ),
        }
        if dst_label != src_label:
            defs[dst_label] = self._node_table_to_json_schema(
                dst_label, remove_embedding=remove_embedding
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
        for _, col_name, col_type, _, _ in info_rows:
            if col_name in ("label", "subj", "obj"):
                continue
            properties[col_name] = _cypher_type_to_json_property(col_type, col_name)

        return {
            "title": label,
            "type": "object",
            "additionalProperties": False,
            "$defs": defs,
            "properties": properties,
            "required": list(properties.keys()),
        }

    def get_symbolic_entities(self) -> List[SymbolicDataModel]:
        """Return a ``SymbolicDataModel`` per existing NODE table.

        Graph-side counterpart of
        `DatabaseAdapter.get_symbolic_data_models`, split by
        graph role. Useful for introspection (``for m in
        kb.get_symbolic_entities(): print(m.get_schema())``) and for
        passing as ``data_models`` to search wrappers that take a
        symbolic-model list.
        """
        models: List[SymbolicDataModel] = []
        for label in sorted(self._existing_tables("NODE")):
            schema = self._node_table_to_json_schema(label)
            models.append(SymbolicDataModel(schema=schema, name=label))
        return models

    def get_symbolic_relations(self) -> List[SymbolicDataModel]:
        """Return a ``SymbolicDataModel`` per existing REL table.

        Each model's schema includes its endpoint node schemas under
        ``$defs`` and references them as ``subj`` / ``obj`` via
        ``$ref`` — same shape Pydantic v2 emits for a hand-written
        ``Relation`` subclass.
        """
        models: List[SymbolicDataModel] = []
        for label in sorted(self._existing_tables("REL")):
            schema = self._rel_table_to_json_schema(label)
            models.append(SymbolicDataModel(schema=schema, name=label))
        return models

    # ------------------------------------------------------------------
    # Cypher
    # ------------------------------------------------------------------

    @contextmanager
    def _execute(self, query: str, params: Optional[Dict[str, Any]] = None):
        """Yield a Ladybug ``QueryResult``, closing it on exit."""
        result = self._con.execute(query, params or {})
        try:
            yield result
        finally:
            try:
                result.close()
            except Exception:
                pass

    async def cypher(
        self,
        query: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        read_only: bool = True,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], str]:
        """Execute a Cypher query.

        Args:
            query: The Cypher query string.
            params: ``$``-parameters to bind into the query. Values are
                always bound (never interpolated).
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string for compact LM input).
            read_only: When ``True`` (default), rejects queries
                containing any write/admin keyword. Pass ``False``
                only from trusted call sites that genuinely need to
                mutate state.
        """
        if read_only:
            _assert_read_only_cypher(query)
        with self._execute(query, params) as result:
            rows = result.get_all()
            cols = result.get_column_names()
        records = [dict(zip(cols, row)) for row in rows]
        return format_search_results(records, output_format)

    # ------------------------------------------------------------------
    # Update (with dedup)
    # ------------------------------------------------------------------

    async def update_entities(
        self,
        entity_or_entities: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert entities with semantic dedup, one query per entity.

        Processes the batch sequentially — each entity gets its own
        ``CALL QUERY_VECTOR_INDEX`` round-trip that combines lookup +
        conditional CREATE into a single statement (see
        `_upsert_entity`). Sequential rather than bulk because
        Ladybug doesn't support the in-query primitives needed to
        share state across UNWIND rows: ``CALL QUERY_VECTOR_INDEX``
        only accepts a literal / parameter vector (not a struct-field
        reference from UNWIND), ``CALL { ... }`` subqueries aren't in
        the parser, and ``;``-separated multi-statement execute()
        with parameters errors out. Each per-entity query commits
        before the next runs, so within-batch dedup falls out for
        free — two near-duplicate inputs will collapse onto the same
        node id because the second query sees the first's CREATE.

        Tradeoff: N round-trips per batch (vs 1 for a hypothetical
        bulk path), but each round-trip uses the HNSW index
        (``O(log M)`` lookup) rather than a full scan, so total work
        scales better on large graphs.
        """
        scalar_in = not isinstance(entity_or_entities, list)
        items = [entity_or_entities] if scalar_in else list(entity_or_entities)
        if not items:
            return None if scalar_in else []

        ids = [await self._upsert_entity(e) for e in items]

        # FTS index rebuild is best-effort — data is already committed.
        # Same shape as the DuckDB adapter: pay the rebuild cost on
        # the write path so search paths can stay query-only.
        touched_labels = {sanitize_label(entity.get("label")) for entity in items}
        for label in touched_labels:
            if label not in self._fts_columns:
                continue
            try:
                self._rebuild_fts_index(label)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"FTS index rebuild failed for {label!r}; "
                    f"entity_fulltext_search results may be stale. ({e})"
                )

        return ids[0] if scalar_in else ids

    async def _upsert_entity(self, entity: Any) -> Optional[Any]:
        """Single-query dedup-or-insert for one entity.

        The PK value comes from the entity's first non-``label``
        property (the column the adapter promoted to PRIMARY KEY
        at table-creation time). Two-branch UNION on the dedup path:

          B1 (CREATE-if-not-a-duplicate):
            ``CALL QUERY_VECTOR_INDEX`` for the nearest neighbour;
            count hits within ``$__max_dist``; if the count is 0,
            ``UNWIND CASE … THEN [1] ELSE [] END`` runs CREATE with
            every property inline (Ladybug's HNSW index rejects
            ``SET`` on the indexed ``embedding`` column, so the
            value has to flow through the CREATE pattern). A PK
            collision in this branch surfaces as a runtime error
            — it means the user passed two entities with the same
            PK but far-apart embeddings, which is a real conflict.

          B2 (RETURN-existing-if-near-duplicate):
            ``CALL QUERY_VECTOR_INDEX`` again, filter by the same
            threshold, return the matched node's PK so the caller
            can resolve the new entity onto the existing one.

        The fast path (no embedding model or no embedding on the
        entity) uses ``MERGE (n:Label {pk_key: $pk}) ON CREATE SET
        … ON MATCH SET …`` so re-inserts of the same PK upsert
        rather than fail — there's no vector dedup safety net in
        this branch, so a second insert is the only way to update
        a node's non-embedding properties.
        """
        if not is_entity(entity):
            raise ValueError(
                "update_entities expects Entity-shaped data models "
                "(must satisfy is_entity, i.e. carry a `label` field)."
            )
        label = sanitize_label(entity.get("label"))
        # Learn the vector dimension from this entity's embedding (or an on-loop
        # probe) before the node table — and its FLOAT[dim] embedding column —
        # is created on first sight of the label.
        await self._ensure_vector_dim(
            entity.get("embedding") if hasattr(entity, "get") else None
        )
        self._ensure_node_table(label, entity)
        pk_key = self._pk_keys.get(label)
        if pk_key is None:
            raise ValueError(
                f"Label {label!r} has no usable primary key — its first "
                f"non-`label` property couldn't be resolved as an identifier."
            )
        props = sanitize_properties(entity.get_json())
        if pk_key not in props or props[pk_key] is None:
            raise ValueError(
                f"Entity of label {label!r} is missing required PK field "
                f"{pk_key!r}; the first non-`label` property carries the "
                f"node identity."
            )
        pk_value = props[pk_key]
        embedding = entity.get("embedding") if hasattr(entity, "get") else None

        # PK is always inline in the pattern. ``embedding`` (if any)
        # is inline too because Ladybug's HNSW index forbids ``SET``
        # on the indexed column. Everything else goes through ON
        # CREATE / ON MATCH SET on the fast path so re-inserts
        # upsert non-PK / non-embedding properties.
        bindings: Dict[str, Any] = {"__pk_value": pk_value}
        settable: List[tuple] = []
        for key, value in props.items():
            if key == pk_key or key == "embedding":
                continue
            settable.append((key, value))
            bindings[key] = value
        set_assigns = ", ".join(f"n.{k} = ${k}" for k, _ in settable)

        # Fast path: no embedding model or no embedding on the entity
        # → no vector dedup; MERGE on PK with conditional SET so
        # repeated updates compose correctly.
        if not embedding or not self.embedding_model:
            merge_set = ""
            if set_assigns:
                merge_set = f" ON CREATE SET {set_assigns} ON MATCH SET {set_assigns}"
            self._con.execute(
                f"MERGE (n:{label} {{{pk_key}: $__pk_value}}){merge_set}",
                bindings,
            )
            return pk_value

        bindings["__embedding"] = embedding
        bindings["__max_dist"] = self._max_distance_for_threshold()
        index_name = f"{label.lower()}_vec"

        # CREATE pattern with every property inline — PK, embedding,
        # and any SET-able prop alike. This is what Branch 1 fires
        # exactly once when the vector index says "no near-duplicate".
        create_inline = [f"{pk_key}: $__pk_value", "embedding: $__embedding"]
        for key, _ in settable:
            create_inline.append(f"{key}: ${key}")
        create_clause = f"CREATE (n:{label} {{{', '.join(create_inline)}}})"

        query = (
            f"CALL QUERY_VECTOR_INDEX('{label}', '{index_name}', "
            f"$__embedding, 1) "
            f"YIELD node, distance "
            f"WITH count(CASE WHEN distance <= $__max_dist THEN 1 END) AS hits "
            f"UNWIND CASE WHEN hits = 0 THEN [1] ELSE [] END AS _ "
            f"{create_clause} "
            f"RETURN n.{pk_key} AS id "
            f"UNION "
            f"CALL QUERY_VECTOR_INDEX('{label}', '{index_name}', "
            f"$__embedding, 1) "
            f"YIELD node, distance "
            f"WITH node, distance WHERE distance <= $__max_dist "
            f"RETURN node.{pk_key} AS id"
        )
        with self._execute(query, bindings) as result:
            rows = result.get_all()
        # Defensive: an empty result shouldn't happen because exactly
        # one branch always yields, but if it does we fall back to
        # the PK the caller supplied.
        return rows[0][0] if rows else pk_value

    def _max_distance_for_threshold(self) -> float:
        """Convert the similarity threshold into a max distance for
        the configured metric.

        For ``cosine`` (the default), similarity = 1 - distance, so
        ``max_dist = 1 - threshold``. For ``l2``/``l2sq``/
        ``dotproduct`` the threshold is interpreted directly as the
        maximum allowed raw distance.
        """
        if self.metric == "cosine":
            return max(0.0, 1.0 - self.dedup_threshold)
        return self.dedup_threshold

    async def update_relations(
        self,
        relation_or_relations: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert relations with endpoint dedup + edge MERGE, sequentially.

        For each relation in turn:
          1. Upsert ``subj`` via `_upsert_entity` (one query —
             dedup against existing nodes with HNSW vector lookup).
          2. Upsert ``obj`` via `_upsert_entity` (one query).
          3. ``MERGE`` the edge between the two resolved ids so the
             same ``(s, label, o)`` triple is never inserted twice.

        Sequential rather than bulk for the same reason
        `update_entities` is — within-batch dedup needs each
        write to be visible to the next read, which Ladybug's
        in-query primitives don't support.
        """
        scalar_in = not isinstance(relation_or_relations, list)
        items = [relation_or_relations] if scalar_in else list(relation_or_relations)
        if not items:
            return None if scalar_in else []

        ids = [await self._upsert_relation(r) for r in items]
        return ids[0] if scalar_in else ids

    async def _upsert_relation(self, relation: Any) -> Optional[str]:
        """One-query MERGE: vector-resolve both endpoints + MERGE edge.

        Same shape as the legacy MemGraph adapter — chain two
        ``CALL QUERY_VECTOR_INDEX`` calls (one per endpoint) into a
        single ``MERGE`` statement. Endpoints that don't match any
        existing node within ``dedup_threshold`` cause the WHERE
        to drop the row, so the MERGE silently no-ops; the caller
        is expected to have upserted the endpoint nodes first (which
        `update_knowledge_graph` does automatically by
        running `update_entities` before
        `update_relations`).

        Falls back to ``MATCH``-by-id + MERGE (3 queries total) when
        either endpoint lacks an embedding or the adapter has no
        embedding model — no vector index means no in-query
        endpoint resolution.
        """
        if not is_relation(relation):
            raise ValueError(
                "update_relations expects Relation-shaped data models "
                "(must satisfy is_relation: subj + label + obj fields)."
            )
        rel_label = sanitize_label(relation.get("label"))
        subj = relation.get_nested_entity("subj")
        obj = relation.get_nested_entity("obj")
        if subj is None or obj is None:
            raise ValueError(
                "Relation subj/obj must be Entity-shaped (a `label` field "
                "is required so the discriminator resolves)."
            )

        subj_label = sanitize_label(subj.get("label"))
        obj_label = sanitize_label(obj.get("label"))
        # Resolve the vector dimension from an endpoint's embedding (or an
        # on-loop probe) before any endpoint/rel table is created.
        await self._ensure_vector_dim(
            (subj.get("embedding") if hasattr(subj, "get") else None)
            or (obj.get("embedding") if hasattr(obj, "get") else None)
        )
        # Create endpoint NODE tables (if unseen) and then the REL table
        # on demand, so free-form labels store without being predeclared.
        self._ensure_node_table(subj_label, subj)
        self._ensure_node_table(obj_label, obj)
        self._ensure_rel_table(rel_label, subj_label, obj_label, relation)
        subj_emb = subj.get("embedding") if hasattr(subj, "get") else None
        obj_emb = obj.get("embedding") if hasattr(obj, "get") else None
        rel_props = sanitize_properties(relation.get_json())

        if self.embedding_model and subj_emb and obj_emb:
            return await self._merge_relation_by_vector(
                subj_label,
                rel_label,
                obj_label,
                subj_emb,
                obj_emb,
                rel_props,
            )

        # Fallback: no vector index to resolve endpoints, so upsert
        # them explicitly then MERGE by their declared PKs. Used
        # when no embeddings are configured at all.
        subj_id = await self._upsert_entity(subj)
        obj_id = await self._upsert_entity(obj)
        subj_pk = self._pk_keys[subj_label]
        obj_pk = self._pk_keys[obj_label]
        set_clauses: List[str] = []
        bindings: Dict[str, Any] = {"__subj_id": subj_id, "__obj_id": obj_id}
        for key, value in rel_props.items():
            set_clauses.append(f"r.{key} = ${key}")
            bindings[key] = value
        set_clause = " SET " + ", ".join(set_clauses) if set_clauses else ""
        self._con.execute(
            f"MATCH (s:{subj_label}), (o:{obj_label}) "
            f"WHERE s.{subj_pk} = $__subj_id AND o.{obj_pk} = $__obj_id "
            f"MERGE (s)-[r:{rel_label}]->(o){set_clause}",
            bindings,
        )
        return f"{subj_id}-{rel_label}->{obj_id}"

    async def _merge_relation_by_vector(
        self,
        subj_label: str,
        rel_label: str,
        obj_label: str,
        subj_emb: List[float],
        obj_emb: List[float],
        rel_props: Dict[str, Any],
    ) -> Optional[str]:
        """Single-query relation MERGE with vector-resolved endpoints."""
        max_dist = self._max_distance_for_threshold()
        subj_pk = self._pk_keys[subj_label]
        obj_pk = self._pk_keys[obj_label]
        bindings: Dict[str, Any] = {
            "__subj_emb": subj_emb,
            "__obj_emb": obj_emb,
            "__max_dist": max_dist,
        }
        set_clauses: List[str] = []
        for key, value in rel_props.items():
            set_clauses.append(f"r.{key} = ${key}")
            bindings[key] = value
        set_part = " SET " + ", ".join(set_clauses) if set_clauses else ""

        query = (
            f"CALL QUERY_VECTOR_INDEX('{subj_label}', "
            f"'{subj_label.lower()}_vec', $__subj_emb, 1) "
            f"YIELD node AS s, distance AS subj_dist "
            f"WITH s, subj_dist WHERE subj_dist <= $__max_dist "
            f"CALL QUERY_VECTOR_INDEX('{obj_label}', "
            f"'{obj_label.lower()}_vec', $__obj_emb, 1) "
            f"YIELD node AS o, distance AS obj_dist "
            f"WITH s, o, subj_dist, obj_dist "
            f"WHERE obj_dist <= $__max_dist "
            f"MERGE (s)-[r:{rel_label}]->(o){set_part} "
            f"RETURN s.{subj_pk} AS subj_id, o.{obj_pk} AS obj_id"
        )
        with self._execute(query, bindings) as result:
            rows = result.get_all()
        if not rows:
            # Either endpoint failed the similarity threshold → edge
            # silently dropped. Caller is responsible for having
            # upserted endpoints first if it wants the relation to
            # land.
            return None
        subj_id, obj_id = rows[0]
        return f"{subj_id}-{rel_label}->{obj_id}"

    async def update_knowledge_graph(self, knowledge_graph: Any) -> Any:
        """Bulk-insert a full KG: entities first, then relations."""
        entities = knowledge_graph.get_nested_entity_list("entities")
        relations = knowledge_graph.get_nested_entity_list("relations")
        ent_ids = await self.update_entities(entities) if entities else []
        rel_ids = await self.update_relations(relations) if relations else []
        return {"entities": ent_ids, "relations": rel_ids}

    # ------------------------------------------------------------------
    # Get / delete
    # ------------------------------------------------------------------

    def _node_to_json(
        self,
        node: Dict[str, Any],
        label: str,
        *,
        remove_embedding: bool = True,
    ) -> Dict[str, Any]:
        """Strip Ladybug's internal node fields and shape for JsonDataModel.

        ``CALL ... RETURN n`` yields a struct with ``_ID`` / ``_LABEL``
        metadata plus the user's properties; the consumer-facing JSON
        view drops those internals, optionally drops the embedding
        vector, and re-injects the ``label`` discriminator (since the
        table name is the only thing carrying it).
        """
        clean = {
            key: value
            for key, value in node.items()
            if not key.startswith("_")
            and key not in (self._COMMUNITY_COLUMN, self._RANK_COLUMN)
            and not (remove_embedding and key == "embedding")
        }
        clean["label"] = label
        return clean

    async def get_entity(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        label: str,
    ) -> Union[Optional[Any], List[Optional[Any]]]:
        """Retrieve entities by primary key. Scalar in / scalar out.

        Mirrors `DuckDBAdapter.get` on the SQL side: a single id
        returns the matched ``JsonDataModel`` (or ``None`` for a
        miss); a list returns a list in the same order with ``None``
        in the unmatched slots.
        """
        return_single = not isinstance(id_or_ids, list)
        ids = [id_or_ids] if return_single else list(id_or_ids)
        if not ids:
            return None if return_single else []

        label = sanitize_label(label)
        pk_key = self._pk_keys.get(label)
        if pk_key is None:
            raise ValueError(
                f"Label {label!r} has no registered primary key — pass it "
                f"in `entity_models` at construction time."
            )

        try:
            schema = self._node_table_to_json_schema(label)
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"get_entity(): cannot introspect table {label!r}; "
                f"returning misses. ({e})"
            )
            return None if return_single else [None] * len(ids)

        results: List[Optional[JsonDataModel]] = [None] * len(ids)
        with self._execute(
            f"MATCH (n:{label}) WHERE n.{pk_key} IN $ids RETURN n",
            {"ids": ids},
        ) as r:
            rows_by_id: Dict[Any, Dict[str, Any]] = {}
            for row in r.get_all():
                node = row[0]
                rows_by_id[node[pk_key]] = node

        for idx, id_val in enumerate(ids):
            node = rows_by_id.get(id_val)
            if node is None:
                continue
            results[idx] = JsonDataModel(
                json=self._node_to_json(node, label),
                schema=schema,
                name=str(id_val),
            )
        return results[0] if return_single else results

    async def delete_entity(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        label: str,
    ) -> int:
        """Delete entities by primary key. Cascades to incident edges.

        Uses Cypher's ``DETACH DELETE`` so any relations incident to
        the matched nodes are removed too (referentially consistent
        graph; no dangling edges). Returns the number of nodes
        actually deleted. The FTS index is rebuilt after the write
        so subsequent search calls don't return ghost rows — same
        write-time rebuild shape `update_entities` uses.
        """
        ids = [id_or_ids] if not isinstance(id_or_ids, list) else list(id_or_ids)
        if not ids:
            return 0

        label = sanitize_label(label)
        pk_key = self._pk_keys.get(label)
        if pk_key is None:
            raise ValueError(
                f"Label {label!r} has no registered primary key — pass it "
                f"in `entity_models` at construction time."
            )

        with self._execute(
            f"MATCH (n:{label}) WHERE n.{pk_key} IN $ids "
            f"DETACH DELETE n RETURN n.{pk_key} AS id",
            {"ids": ids},
        ) as r:
            deleted = len(r.get_all())

        # Best-effort FTS rebuild — failure here just means searches
        # may report stale hits until the next write triggers a rebuild.
        if label in self._fts_columns:
            try:
                self._rebuild_fts_index(label)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"FTS index rebuild failed for {label!r} after delete; "
                    f"entity_fulltext_search results may be stale. ({e})"
                )
        return deleted

    async def delete_relation(
        self,
        *,
        label: str,
        source_id: Any,
        target_id: Any,
    ) -> int:
        """Delete ``(source_id)-[:label]->(target_id)`` edges.

        Returns the number of edges actually removed (zero if no
        matching edge exists). Endpoints are matched by their declared
        PKs — same convention as `get_entity` / `update_entities`.
        """
        label = sanitize_label(label)
        subj_label, obj_label = self._resolve_endpoint_labels(label)
        subj_pk = self._pk_keys.get(subj_label)
        obj_pk = self._pk_keys.get(obj_label)
        if subj_pk is None or obj_pk is None:
            raise ValueError(
                f"Endpoint labels {subj_label!r}/{obj_label!r} have no "
                f"registered primary key — pass them via `entity_models`."
            )

        with self._execute(
            f"MATCH (s:{subj_label})-[r:{label}]->(o:{obj_label}) "
            f"WHERE s.{subj_pk} = $source_id AND o.{obj_pk} = $target_id "
            f"DELETE r RETURN s.{subj_pk} AS id",
            {"source_id": source_id, "target_id": target_id},
        ) as result:
            return len(result.get_all())

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    # Each algo returns its community id under a different column,
    # and the set of optional keyword args it accepts varies (probed
    # against Ladybug's algo extension at the time of writing).
    # Keeping the dispatch table-driven beats a chain of if/elif and
    # makes it obvious which keys are forwarded to which algorithm.
    _COMMUNITY_ALGORITHMS = {
        "louvain": {
            "column": "louvain_id",
            "supported_params": {"maxIterations"},
        },
        "weakly_connected_components": {
            "column": "group_id",
            "supported_params": {"maxIterations"},
        },
        "strongly_connected_components": {
            "column": "group_id",
            "supported_params": {"maxIterations"},
        },
    }

    def _entity_model_for_label(self, label: str) -> Optional[Any]:
        """Look up a registered entity model class by its label.

        Used by `detect_communities` to reconstruct typed
        Pydantic instances from raw node structs — the user passed
        ``Person`` / ``City`` etc. at construction time and we want
        the returned KnowledgeGraphs to round-trip through those
        types instead of collapsing to the base `Entity`.
        """
        for model in self.entity_models:
            try:
                if sanitize_label(model.get_schema()["title"]) == label:
                    return model
            except Exception:  # noqa: BLE001
                continue
        return None

    def _relation_model_for_label(self, label: str) -> Optional[Any]:
        """Look up a registered relation model class by its label."""
        for model in self.relation_models:
            try:
                if sanitize_label(model.get_schema()["title"]) == label:
                    return model
            except Exception:  # noqa: BLE001
                continue
        return None

    @staticmethod
    def _json_type_to_python(spec: Dict[str, Any]):
        """Map a JSON-schema property type to a Python type for model synth."""
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return mapping.get(spec.get("type"), Any)

    def _synth_entity_model(self, label: str):
        """Synthesize a generic ``Entity`` subclass for an unregistered label.

        Free-form graphs carry labels with no matching ``entity_models``
        class, so reconstructing them as the bare `Entity` would
        drop every property but ``label``. Instead, build a Pydantic
        model from the table's introspected columns (cached per label)
        so ``name`` and any other stored field survive read-back. Fields
        are optional with a ``None`` default so partial rows still
        validate. Returns ``None`` when the table can't be introspected.
        """
        if label in self._synth_models:
            return self._synth_models[label]
        try:
            schema = self._node_table_to_json_schema(label)
        except Exception:  # noqa: BLE001
            self._synth_models[label] = None
            return None
        fields: Dict[str, Any] = {}
        for name, spec in (schema.get("properties") or {}).items():
            if name == "label":
                continue  # provided by the Entity base
            fields[name] = (Optional[self._json_type_to_python(spec)], None)
        model = create_model(label, __base__=Entity, **fields)
        self._synth_models[label] = model
        return model

    def _build_entity_instance(self, label: str, node: Dict[str, Any]):
        """Construct an Entity-typed instance from a Ladybug node struct.

        Tries the user's registered subclass first (so ``Person``
        keeps its ``name`` / ``embedding`` fields); for a free-form
        label with no registered class, synthesizes a generic subclass
        from the table schema so its properties survive too. Falls back
        to the bare `Entity` (label only) only when neither
        resolves or validation fails.
        """
        clean = self._node_to_json(node, label)
        model = self._entity_model_for_label(label) or self._synth_entity_model(label)
        if model is not None:
            try:
                return model(**clean)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Could not reconstruct {label!r} from model {model!r}; "
                    f"falling back to base Entity. ({e})"
                )
        return Entity(label=label)

    def _build_relation_instance(
        self,
        rel_label: str,
        subj_label: str,
        obj_label: str,
        edge: Dict[str, Any],
        subj_node: Dict[str, Any],
        obj_node: Dict[str, Any],
    ):
        """Construct a Relation-typed instance from a Ladybug edge struct."""
        subj = self._build_entity_instance(subj_label, subj_node)
        obj = self._build_entity_instance(obj_label, obj_node)
        rel_model = self._relation_model_for_label(rel_label)
        if rel_model is not None:
            props = {
                key: value
                for key, value in edge.items()
                if not key.startswith("_") and key != "embedding"
            }
            props["label"] = rel_label
            props["subj"] = subj
            props["obj"] = obj
            try:
                return rel_model(**props)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Could not reconstruct {rel_label!r} from registered "
                    f"model {rel_model!r}; falling back to base Relation. "
                    f"({e})"
                )
        return Relation(label=rel_label, subj=subj, obj=obj)

    def _resolve_projection_labels(
        self,
        node_labels: Optional[List[str]],
        rel_labels: Optional[List[str]],
    ) -> tuple:
        """Normalize the (node_labels, rel_labels) arguments shared by
        every algo-extension method.

        Resolves ``None`` to "every existing table of this kind",
        sanitizes user-supplied label names, and drops anything that
        doesn't exist on the graph. Returns sorted lists so the
        downstream interpolation is deterministic (also makes test
        assertions stable).
        """
        existing_nodes = self._existing_tables("NODE")
        existing_rels = self._existing_tables("REL")
        if node_labels is None:
            sanitized_nodes = sorted(existing_nodes)
        else:
            sanitized_nodes = sorted(
                {sanitize_label(l) for l in node_labels} & existing_nodes
            )
        if rel_labels is None:
            sanitized_rels = sorted(existing_rels)
        else:
            sanitized_rels = sorted(
                {sanitize_label(l) for l in rel_labels} & existing_rels
            )
        return sanitized_nodes, sanitized_rels

    # Allowlists for the optional kwargs forwarded to Ladybug's
    # ``QUERY_FTS_INDEX`` / ``QUERY_VECTOR_INDEX`` procedures. Gated
    # through ``_build_algo_kwargs`` (same mechanism as the algo procs)
    # so unsupported keys are dropped rather than passed through blindly.
    # Keys mirror what ``_collect_fts_query_kwargs`` /
    # ``_collect_vector_query_kwargs`` emit.
    _FTS_QUERY_SUPPORTED_PARAMS = {
        "conjunctive",
        "b",
    }
    _VECTOR_QUERY_SUPPORTED_PARAMS = {
        "efs",
    }

    @staticmethod
    def _collect_fts_query_kwargs(
        conjunctive: bool,
        bm25_b: Optional[float],
    ) -> Dict[str, Any]:
        """Translate the Python-side ``conjunctive`` / ``bm25_b`` args
        into the camelCase keys Ladybug's ``QUERY_FTS_INDEX`` accepts.

        Kept separate from each search method so the four FTS-bearing
        surfaces (entity / entity-hybrid / relation-hybrid /
        chain-hybrid) share one translation rule. ``conjunctive`` is
        only forwarded when ``True`` so the default ``False`` matches
        Ladybug's own default behaviour without burning an explicit
        kwarg in the query.
        """
        kwargs: Dict[str, Any] = {}
        if conjunctive:
            kwargs["conjunctive"] = True
        if bm25_b is not None:
            kwargs["b"] = bm25_b
        return kwargs

    @staticmethod
    def _collect_vector_query_kwargs(
        ef_search: Optional[int],
    ) -> Dict[str, Any]:
        """Translate the Python-side ``ef_search`` arg into the
        camelCase ``efs`` key Ladybug's ``QUERY_VECTOR_INDEX``
        accepts. Mirrors `_collect_fts_query_kwargs` so the
        per-call wiring stays uniform across the search methods.
        """
        if ef_search is None:
            return {}
        return {"efs": ef_search}

    def _build_algo_kwargs(
        self,
        kwargs: Dict[str, Any],
        supported: set,
    ) -> tuple:
        """Build the optional-keyword fragment forwarded to a Ladybug
        algo procedure call.

        Ladybug algo procs (``LOUVAIN``, ``PAGE_RANK``, ...) accept
        optional camelCase kwargs after the projection name; the
        accepted set varies per procedure. This helper:

          * filters ``kwargs`` to keys the procedure actually
            supports (silently drops the rest — keeps the cross-algo
            Python signature stable),
          * renders the fragment ``, k1 := $k1, k2 := $k2`` (empty
            string when nothing is forwarded so the caller can
            interpolate unconditionally),
          * returns the matching ``$``-parameter bind dict.

        Returns ``(fragment: str, bind: Dict[str, Any])``.
        """
        forwarded = {k: v for k, v in kwargs.items() if k in supported}
        if not forwarded:
            return "", {}
        fragment = ", " + ", ".join(f"{k} := ${k}" for k in forwarded)
        return fragment, forwarded

    @contextmanager
    def _projected_graph(
        self,
        node_labels: List[str],
        rel_labels: List[str],
        *,
        purpose: str = "algorithm",
    ):
        """Context-manage a Ladybug ``algo``-extension graph projection.

        Most algo procedures (``LOUVAIN``, ``WEAKLY_CONNECTED_COMPONENTS``,
        ``PAGE_RANK``, ...) take the *name* of a projected graph
        rather than working off the live tables, so callers have to:
        ``CALL PROJECT_GRAPH(name, [...], [...])`` → run algo → ``CALL
        DROP_PROJECTED_GRAPH(name)``. Forgetting the drop leaves a
        stale projection that breaks subsequent calls.

        This helper:

          * builds a unique projection name with a uuid suffix so two
            concurrent calls don't collide,
          * interpolates the label lists into the ``PROJECT_GRAPH``
            call (Ladybug doesn't accept ``$``-parameter binding for
            them — but the labels went through ``sanitize_label``
            upstream, so they're injection-safe),
          * yields the projection name to the caller,
          * always drops the projection on exit, even if the algo
            call raised.
        """
        node_list_lit = ", ".join(f"'{n}'" for n in node_labels)
        rel_list_lit = ", ".join(f"'{r}'" for r in rel_labels)
        proj_name = f"_proj_{uuid.uuid4().hex[:12]}"
        try:
            self._con.execute(
                f"CALL PROJECT_GRAPH('{proj_name}', [{node_list_lit}], [{rel_list_lit}])"
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to project graph for {purpose}: {e}") from e
        try:
            yield proj_name
        finally:
            try:
                self._con.execute(f"CALL DROP_PROJECTED_GRAPH('{proj_name}')")
            except Exception:  # noqa: BLE001
                pass

    async def detect_communities(
        self,
        *,
        algorithm: str = "louvain",
        node_labels: Optional[List[str]] = None,
        rel_labels: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
    ) -> KnowledgeGraphs:
        """Run a community-detection algorithm via Ladybug's ``algo``
        extension and return one `KnowledgeGraph` per
        community.

        Algorithms:

          * ``"louvain"`` — modularity-based clustering. Ladybug's
            implementation only supports a single node label at a
            time; if ``node_labels`` resolves to more than one this
            method raises. Nodes with degree 0 are assigned community
            ``-1`` by Louvain; each such isolated node lands in its
            own singleton `KnowledgeGraph` so the output
            stays unambiguous.
          * ``"weakly_connected_components"`` — disconnected-piece
            clustering across any number of node labels, ignoring
            edge direction. Useful when "communities" really means
            "graph islands".
          * ``"strongly_connected_components"`` — same as WCC but
            requires directed reachability in both directions. Edges
            that are only traversable one way don't merge their
            endpoints into the same component.

        Cross-community edges are dropped: a `KnowledgeGraph`
        for community ``c`` contains an edge only when both endpoints
        belong to ``c``. Edges that straddle communities don't belong
        to any single subgraph by definition.

        Args:
            algorithm: One of ``"louvain"``,
                ``"weakly_connected_components"``, or
                ``"strongly_connected_components"``.
            node_labels: Optional whitelist of NODE tables to
                project. Defaults to every existing one.
            rel_labels: Optional whitelist of REL tables to project.
                Defaults to every existing one.
            max_iterations: Optional upper bound on the algorithm's
                iteration count. Forwarded to Ladybug's
                ``maxIterations`` keyword arg. ``None`` defers to the
                engine default.

        Returns:
            `KnowledgeGraphs` — one
            `KnowledgeGraph` per detected community, sorted
            by community id for deterministic ordering.
        """
        if algorithm not in self._COMMUNITY_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}; expected one of "
                f"{sorted(self._COMMUNITY_ALGORITHMS)}"
            )

        sanitized_nodes, sanitized_rels = self._resolve_projection_labels(
            node_labels, rel_labels
        )

        if not sanitized_nodes:
            return KnowledgeGraphs(knowledge_graphs=[])

        if algorithm == "louvain" and len(sanitized_nodes) != 1:
            raise ValueError(
                f"Louvain only supports a single node label at a time, "
                f"got {sanitized_nodes!r}. Pass `node_labels=[<one>]` or "
                f"switch to algorithm='weakly_connected_components' for "
                f"multi-label clustering."
            )

        algo_spec = self._COMMUNITY_ALGORITHMS[algorithm]
        community_column = algo_spec["column"]

        # Build the keyword-arg fragment from the user's params, but
        # only forward keys the algorithm actually accepts. This keeps
        # the cross-algorithm signature stable on our side while
        # quietly dropping params Ladybug doesn't know about for the
        # selected algo.
        algo_kwargs: Dict[str, Any] = {}
        if max_iterations is not None:
            algo_kwargs["maxIterations"] = max_iterations
        kwargs_fragment, kwargs_bind = self._build_algo_kwargs(
            algo_kwargs, algo_spec["supported_params"]
        )

        with self._projected_graph(
            sanitized_nodes, sanitized_rels, purpose="community detection"
        ) as proj_name:
            # node._LABEL / node._ID are Ladybug internals; we return
            # the whole struct and unpack in Python so we can also
            # look up the per-label PK via ``self._pk_keys``.
            rows = self._con.execute(
                f"CALL {algorithm.upper()}('{proj_name}'"
                f"{kwargs_fragment}) "
                f"RETURN node, {community_column} AS comm_id",
                kwargs_bind,
            ).get_all()

        # Assign each node to a community. Louvain marks isolated
        # nodes (degree 0) with ``-1``; we promote them to their own
        # singleton community keyed by ``(label, pk_value)`` so the
        # output is a clean partition.
        from collections import defaultdict

        comm_to_nodes: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        comm_lookup: Dict[tuple, Any] = {}
        for node, comm_id in rows:
            label = node["_LABEL"]
            pk_key = self._pk_keys.get(label, "id")
            pk_value = node.get(pk_key)
            if comm_id == -1:
                # One singleton per isolated node — same partition
                # semantics WCC would give them.
                comm_id = ("__isolated__", label, pk_value)
            comm_to_nodes[comm_id].append(node)
            comm_lookup[(label, pk_value)] = comm_id

        # Pull edges in one pass per rel table; group by the shared
        # community of their two endpoints.
        comm_to_edges: Dict[Any, List[tuple]] = defaultdict(list)
        for rel_label in sanitized_rels:
            try:
                subj_label, obj_label = self._resolve_endpoint_labels(rel_label)
            except Exception:  # noqa: BLE001
                continue
            if subj_label not in sanitized_nodes or obj_label not in sanitized_nodes:
                # Endpoint table wasn't part of this projection — its
                # edges don't have communities to land in.
                continue
            subj_pk = self._pk_keys.get(subj_label)
            obj_pk = self._pk_keys.get(obj_label)
            if subj_pk is None or obj_pk is None:
                continue
            edge_rows = self._con.execute(
                f"MATCH (s:{subj_label})-[r:{rel_label}]->(o:{obj_label}) RETURN s, r, o"
            ).get_all()
            for subj_node, edge, obj_node in edge_rows:
                s_comm = comm_lookup.get((subj_label, subj_node[subj_pk]))
                o_comm = comm_lookup.get((obj_label, obj_node[obj_pk]))
                # Cross-community edges have nowhere to live.
                if s_comm is None or s_comm != o_comm:
                    continue
                comm_to_edges[s_comm].append(
                    (rel_label, subj_label, obj_label, edge, subj_node, obj_node)
                )

        # Build the KnowledgeGraph instances. Integer community ids
        # sort before tuple-keyed singletons; both go through
        # ``str()`` for the sort key so mixed types don't blow up.
        kgs: List[KnowledgeGraph] = []
        for comm_id in sorted(comm_to_nodes.keys(), key=str):
            entities = [
                self._build_entity_instance(node["_LABEL"], node)
                for node in comm_to_nodes[comm_id]
            ]
            relations = [
                self._build_relation_instance(
                    rel_label,
                    subj_label,
                    obj_label,
                    edge,
                    subj_node,
                    obj_node,
                )
                for (
                    rel_label,
                    subj_label,
                    obj_label,
                    edge,
                    subj_node,
                    obj_node,
                ) in comm_to_edges.get(comm_id, [])
            ]
            kgs.append(KnowledgeGraph(entities=entities, relations=relations))
        return KnowledgeGraphs(knowledge_graphs=kgs)

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    # Keyword args Ladybug's ``PAGE_RANK`` procedure accepts. Probed
    # at the time of writing — Ladybug rejects unknown keys with a
    # binder exception, so the API surface is gated through
    # ``_build_algo_kwargs`` rather than passing user input through
    # blindly.
    _PAGERANK_SUPPORTED_PARAMS = {
        "dampingFactor",
        "maxIterations",
        "tolerance",
        "normalizeInitial",
    }

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
        """Rank entities by PageRank importance via Ladybug's ``algo``
        extension.

        Returns rows shaped like
        ``{<pk_column>: <pk_value>, "label": <label>, "node": <full node>,
        "rank": <float>}`` sorted by ``rank`` descending. The per-label
        PK column name is kept verbatim (e.g. ``name`` for ``Person``,
        ``isbn`` for ``Book``) instead of being aliased to ``id`` — same
        convention as `entity_similarity_search` — so callers
        carrying multiple labels in one result set can still tell them
        apart.

        Args:
            node_labels: Optional whitelist of NODE tables. ``None``
                projects every existing one.
            rel_labels: Optional whitelist of REL tables. ``None``
                projects every existing one.
            damping_factor: Ladybug's ``dampingFactor`` arg; 0.85 is
                the standard textbook value.
            max_iterations: Ladybug's ``maxIterations`` arg.
            tolerance: Ladybug's ``tolerance`` arg — convergence
                threshold for the L1 difference between iterations.
                ``None`` defers to Ladybug's default.
            normalize_initial: Ladybug's ``normalizeInitial`` arg —
                whether to normalize the starting rank vector.
                ``None`` defers to Ladybug's default.
            k: Optional cap on returned rows. ``None`` returns every
                ranked entity.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        sanitized_nodes, sanitized_rels = self._resolve_projection_labels(
            node_labels, rel_labels
        )
        if not sanitized_nodes:
            return format_search_results([], output_format)

        algo_kwargs: Dict[str, Any] = {
            "dampingFactor": damping_factor,
            "maxIterations": max_iterations,
        }
        if tolerance is not None:
            algo_kwargs["tolerance"] = tolerance
        if normalize_initial is not None:
            algo_kwargs["normalizeInitial"] = normalize_initial
        kwargs_fragment, kwargs_bind = self._build_algo_kwargs(
            algo_kwargs, self._PAGERANK_SUPPORTED_PARAMS
        )

        with self._projected_graph(
            sanitized_nodes, sanitized_rels, purpose="pagerank"
        ) as proj_name:
            rows = self._con.execute(
                f"CALL PAGE_RANK('{proj_name}'{kwargs_fragment}) RETURN node, rank",
                kwargs_bind,
            ).get_all()

        # Shape rows into [{<pk>: <pk_value>, "label": ..., "node": ...,
        # "rank": ...}]. The pk column name varies per label, mirroring
        # the convention used by entity_similarity_search.
        records: List[Dict[str, Any]] = []
        for node, rank in rows:
            label = node["_LABEL"]
            pk_key = self._pk_keys.get(label, "id")
            pk_value = node.get(pk_key)
            records.append(
                {
                    pk_key: pk_value,
                    "label": label,
                    "node": self._node_to_json(node, label),
                    "rank": rank,
                }
            )
        records.sort(key=lambda r: r["rank"], reverse=True)
        if k is not None:
            records = records[:k]
        return format_search_results(records, output_format)

    # ------------------------------------------------------------------
    # GraphRAG-style local / global search
    # ------------------------------------------------------------------

    # Reserved columns ``build_communities`` stamps onto node tables so
    # ``global_graph_search`` can aggregate without re-running clustering.
    _COMMUNITY_COLUMN = "community"
    _RANK_COLUMN = "rank"

    @staticmethod
    def _internal_id_key(internal_id: Optional[Dict[str, Any]]) -> Optional[tuple]:
        """Hashable key for a Ladybug ``_ID`` / ``_SRC`` / ``_DST`` struct.

        Ladybug surfaces node ids and edge endpoints as
        ``{"offset": int, "table": int}`` dicts, which are unhashable
        and so can't index a dict directly. Collapse to an
        ``(offset, table)`` tuple so a path's edges can be matched
        back to their endpoint nodes by identity.
        """
        if not internal_id:
            return None
        return (internal_id.get("offset"), internal_id.get("table"))

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
    ) -> KnowledgeGraph:
        """GraphRAG-style *local* search: seed by vector, then expand.

        One fused round-trip per query embedding:

          1. ``CALL QUERY_VECTOR_INDEX`` finds the ``k`` entities of
             ``label`` closest to the query — the seeds.
          2. ``WITH`` carries them through (with an optional
             distance-threshold filter).
          3. ``OPTIONAL MATCH p = (seed)-[*1..max_hops]-(n)`` collects
             every node and edge within ``max_hops``, *undirected* so
             both in- and out-neighbours land in the context.
             ``OPTIONAL`` keeps edge-less seeds in the result.

        The returned `KnowledgeGraph` is the deduped union of
        every seed's neighbourhood — the local context subgraph you'd
        hand a generator alongside the question. Relations are rebuilt
        from each path edge's stored ``_SRC`` / ``_DST`` (not the
        traversal direction), so ``subj`` / ``obj`` stay schema-correct
        even though the walk is undirected.

        Args:
            text_or_texts: Query text (or list). Each is embedded and
                expanded; the neighbourhoods merge into one graph.
            label: Entity label whose vector index seeds the search.
            max_hops: Neighbourhood radius in edges (>= 1, default 2).
            k: Number of seed entities per query text.
            threshold: Optional seed vector-distance ceiling — seeds
                beyond it are dropped before expansion.
            rel_label: Optional rel-label constraint applied to every
                hop. ``None`` (default) traverses any edge type.
            ef_search: Optional HNSW ``efs`` for the seed lookup.

        Returns:
            A `KnowledgeGraph` of the deduped neighbourhood
            entities and relations. Empty when no seed matches.
        """
        if not self.embedding_model:
            raise ValueError(
                "local_graph_search requires an embedding_model on the LadybugAdapter."
            )
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}.")

        label = sanitize_label(label)
        rel = sanitize_label(rel_label) if rel_label is not None else None
        if not text_or_texts:
            return KnowledgeGraph(entities=[], relations=[])

        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts
        vectors = (await self.embedding_model(EmbeddingRequest(texts=texts)))[
            "embeddings"
        ]

        vec_index = f"{label.lower()}_vec"
        vec_kwargs_frag, vec_kwargs_bind = self._build_algo_kwargs(
            self._collect_vector_query_kwargs(ef_search),
            self._VECTOR_QUERY_SUPPORTED_PARAMS,
        )

        # ``WHERE TRUE`` keeps the query shape stable whether or not a
        # threshold is supplied (same no-op trick path_similarity_search
        # uses to avoid string-template branching).
        seed_filter = "dist <= $max" if threshold is not None else "TRUE"
        hop_pattern = (
            f"-[:{rel}*1..{max_hops}]-" if rel is not None else f"-[*1..{max_hops}]-"
        )

        query = (
            f"CALL QUERY_VECTOR_INDEX('{label}', '{vec_index}', "
            f"$vec, $k{vec_kwargs_frag}) "
            f"YIELD node AS seed, distance AS dist "
            f"WITH seed, dist WHERE {seed_filter} "
            f"OPTIONAL MATCH p = (seed){hop_pattern}(n) "
            f"RETURN seed, nodes(p) AS nodes, rels(p) AS rels"
        )

        # Pass 1: collect every node struct (indexed by internal id so
        # edges can find their endpoints) and dedup entities by
        # (label, pk). Raw edge structs are stashed for pass 2.
        nodes_by_iid: Dict[tuple, Dict[str, Any]] = {}
        entities: Dict[tuple, Any] = {}
        raw_edges: Dict[tuple, Dict[str, Any]] = {}

        def _collect_node(node: Optional[Dict[str, Any]]) -> None:
            if not node:
                return
            iid = self._internal_id_key(node.get("_ID"))
            if iid is not None and iid not in nodes_by_iid:
                nodes_by_iid[iid] = node
            node_label = node.get("_LABEL")
            pk_key = self._pk_keys.get(node_label, "id")
            ent_key = (node_label, node.get(pk_key))
            if ent_key not in entities:
                entities[ent_key] = self._build_entity_instance(node_label, node)

        for vec in vectors:
            params = {"vec": vec, "k": k, **vec_kwargs_bind}
            if threshold is not None:
                params["max"] = threshold
            with self._execute(query, params) as r:
                cols = r.get_column_names()
                for row in r.get_all():
                    rec = dict(zip(cols, row))
                    _collect_node(rec.get("seed"))
                    for node in rec.get("nodes") or []:
                        _collect_node(node)
                    for edge in rec.get("rels") or []:
                        edge_key = self._internal_id_key(edge.get("_ID"))
                        if edge_key is not None and edge_key not in raw_edges:
                            raw_edges[edge_key] = edge

        # Pass 2: rebuild relations. The edge's _SRC/_DST point at the
        # stored subj/obj nodes regardless of which way we walked the
        # (undirected) path, so subj/obj stay schema-correct.
        relations: List[Any] = []
        for edge in raw_edges.values():
            subj_node = nodes_by_iid.get(self._internal_id_key(edge.get("_SRC")))
            obj_node = nodes_by_iid.get(self._internal_id_key(edge.get("_DST")))
            if subj_node is None or obj_node is None:
                # Endpoint fell outside the collected neighbourhood —
                # shouldn't happen (path nodes include both ends), but
                # skip rather than emit a half-built relation.
                continue
            relations.append(
                self._build_relation_instance(
                    edge.get("_LABEL"),
                    subj_node.get("_LABEL"),
                    obj_node.get("_LABEL"),
                    edge,
                    subj_node,
                    obj_node,
                )
            )

        return KnowledgeGraph(
            entities=list(entities.values()),
            relations=relations,
        )

    def _node_columns(self, label: str) -> set:
        """Return the set of column names on NODE table ``label``."""
        rows = self._con.execute(f"CALL TABLE_INFO('{label}') RETURN *").get_all()
        return {col_name for _, col_name, _, _, _ in rows}

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

        The index-time half of GraphRAG-global: clustering and ranking
        are precomputed once and persisted as the ``community`` /
        ``rank`` properties on each node, so query-time
        `global_graph_search` is a single aggregation read rather
        than a project→cluster→drop sequence on every call.

        Runs `detect_communities` to partition the graph, then
        (when ``with_pagerank``) `pagerank` to score importance,
        lazily ``ALTER``-adds the two reserved columns to each touched
        node table, and writes the values back keyed by primary key.
        Idempotent — re-running overwrites the previous stamping.

        Args:
            algorithm: Community-detection algorithm; see
                `detect_communities`.
            node_labels: Optional NODE-table whitelist. ``None`` =
                every existing one.
            rel_labels: Optional REL-table whitelist. ``None`` =
                every existing one.
            max_iterations: Optional iteration cap forwarded to the
                clustering algorithm.
            with_pagerank: When ``True`` (default), also stamp a
                ``rank`` column from PageRank; communities then sort
                by aggregate importance in global search. When
                ``False``, ``rank`` is left at 0.0 and global search
                effectively orders by size.
            damping_factor: PageRank damping; ignored when
                ``with_pagerank`` is ``False``.

        Returns:
            The number of nodes stamped with a community id.
        """
        communities = await self.detect_communities(
            algorithm=algorithm,
            node_labels=node_labels,
            rel_labels=rel_labels,
            max_iterations=max_iterations,
        )

        # Map each (label, pk) -> integer community id. detect_communities
        # returns one KnowledgeGraph per community in deterministic order,
        # so the enumeration index is a stable community id.
        comm_by_entity: Dict[tuple, int] = {}
        touched_labels: set = set()
        for comm_id, kg in enumerate(communities.knowledge_graphs):
            for entity in kg.entities:
                ent_label = sanitize_label(entity.label)
                pk_key = self._pk_keys.get(ent_label, "id")
                pk_value = getattr(entity, pk_key, None)
                if pk_value is None:
                    continue
                comm_by_entity[(ent_label, pk_value)] = comm_id
                touched_labels.add(ent_label)

        # Optional PageRank pass, keyed the same way.
        rank_by_entity: Dict[tuple, float] = {}
        if with_pagerank:
            pr_rows = await self.pagerank(
                node_labels=node_labels,
                rel_labels=rel_labels,
                damping_factor=damping_factor,
                output_format="json",
            )
            for row in pr_rows:
                ent_label = row["label"]
                pk_key = self._pk_keys.get(ent_label, "id")
                rank_by_entity[(ent_label, row.get(pk_key))] = row["rank"]

        # Ensure the reserved columns exist on each touched table. Guard
        # first against a user model that legitimately declares one of
        # the reserved names — stamping would clobber that field, and
        # the strip in _node_to_json would hide it, so fail loudly.
        for label in touched_labels:
            model = self._entity_model_for_label(label)
            if model is not None:
                try:
                    declared = set(model.get_schema().get("properties", {}))
                except Exception:  # noqa: BLE001
                    declared = set()
                clash = declared & {self._COMMUNITY_COLUMN, self._RANK_COLUMN}
                if clash:
                    raise ValueError(
                        f"Entity model for label {label!r} declares reserved "
                        f"field(s) {sorted(clash)}; build_communities needs "
                        f"{self._COMMUNITY_COLUMN!r}/{self._RANK_COLUMN!r} as "
                        f"internal columns. Rename the conflicting field(s)."
                    )
            existing = self._node_columns(label)
            if self._COMMUNITY_COLUMN not in existing:
                self._con.execute(
                    f"ALTER TABLE {label} ADD {self._COMMUNITY_COLUMN} INT64"
                )
            if self._RANK_COLUMN not in existing:
                self._con.execute(f"ALTER TABLE {label} ADD {self._RANK_COLUMN} DOUBLE")

        # Write community + rank back, one node at a time (Ladybug binds
        # scalar params per statement; same per-node write shape as the
        # upsert path).
        stamped = 0
        for (label, pk_value), comm_id in comm_by_entity.items():
            pk_key = self._pk_keys.get(label, "id")
            self._con.execute(
                f"MATCH (n:{label}) WHERE n.{pk_key} = $pk "
                f"SET n.{self._COMMUNITY_COLUMN} = $comm, "
                f"n.{self._RANK_COLUMN} = $rank",
                {
                    "pk": pk_value,
                    "comm": comm_id,
                    "rank": rank_by_entity.get((label, pk_value), 0.0),
                },
            )
            stamped += 1
        return stamped

    async def global_graph_search(
        self,
        *,
        node_labels: Optional[List[str]] = None,
        k: int = 10,
        members_per_community: int = 10,
        output_format: str = "json",
    ):
        """GraphRAG-style *global* search: per-community aggregates.

        The query-time half of GraphRAG-global. Reads the ``community``
        / ``rank`` properties `build_communities` stamped and
        rolls them up in a single multi-label aggregation query — no
        clustering at query time. Each returned row describes one
        community::

            {"community": <int>, "size": <int>, "total_rank": <float>,
             "members": [<node>, ...]}

        ordered by ``total_rank`` descending and capped at ``k``
        communities. ``members`` carries the community's nodes (best
        first by ``rank``) so a caller can run the LM map-reduce step
        — summarise each community, then combine — on top. That
        map-reduce stays above the adapter; Cypher only does the
        retrieval.

        Only node tables that `build_communities` has stamped
        participate; unstamped tables are skipped so the query never
        references a missing column. Returns an empty result when no
        table has been built.

        Args:
            node_labels: Optional NODE-table whitelist. ``None`` =
                every stamped table.
            k: Maximum number of communities to return.
            members_per_community: Cap on member nodes carried per
                community row (best first by ``rank``).
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        sanitized_nodes, _ = self._resolve_projection_labels(node_labels, None)
        # Restrict to tables that actually carry the reserved column, so
        # the multi-label match never references a column a table lacks.
        built = [
            label
            for label in sanitized_nodes
            if self._COMMUNITY_COLUMN in self._node_columns(label)
        ]
        if not built:
            return format_search_results([], output_format)

        label_pattern = "|".join(built)
        rows = self._con.execute(
            f"MATCH (n:{label_pattern}) "
            f"WHERE n.{self._COMMUNITY_COLUMN} IS NOT NULL "
            f"RETURN n.{self._COMMUNITY_COLUMN} AS community, "
            f"count(n) AS size, "
            f"sum(n.{self._RANK_COLUMN}) AS total_rank, "
            f"collect(n) AS members "
            f"ORDER BY total_rank DESC LIMIT $k",
            {"k": k},
        ).get_all()

        records: List[Dict[str, Any]] = []
        for community, size, total_rank, members in rows:
            # Members come back as raw node structs; order by stamped
            # rank (desc), strip internals, and cap per community.
            ordered = sorted(
                members,
                key=lambda m: m.get(self._RANK_COLUMN) or 0.0,
                reverse=True,
            )[:members_per_community]
            records.append(
                {
                    "community": community,
                    "size": size,
                    "total_rank": total_rank,
                    "members": [self._node_to_json(m, m.get("_LABEL")) for m in ordered],
                }
            )
        return format_search_results(records, output_format)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

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
        """Vector search over entities of the given label.

        Args:
            text_or_texts: Query text or list of query texts. Multiple
                queries are merged into a single ranked result set
                (best score per id kept).
            label: The entity label to search within.
            k: Maximum number of results.
            threshold: Optional maximum vector distance — rows beyond
                this are dropped.
            ef_search: Optional HNSW ``efs`` — search-time depth of
                the candidate list. Higher = better recall but slower.
                ``None`` defers to Ladybug's default.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        if not self.embedding_model:
            raise ValueError(
                "entity_similarity_search requires an embedding_model on the "
                "LadybugAdapter."
            )
        label = sanitize_label(label)
        pk_key = self._pk_keys.get(label, "id")
        if not text_or_texts:
            return format_search_results([], output_format)
        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts
        embeddings = await self.embedding_model(EmbeddingRequest(texts=texts))
        vectors = embeddings["embeddings"]

        vec_kwargs_frag, vec_kwargs_bind = self._build_algo_kwargs(
            self._collect_vector_query_kwargs(ef_search),
            self._VECTOR_QUERY_SUPPORTED_PARAMS,
        )

        # Best (lowest) distance per PK across all query vectors. The
        # PK column is returned under its own name (e.g. ``name``);
        # the Python dedup keys off it instead of a synthetic ``id``.
        best: Dict[Any, Dict[str, Any]] = {}
        for vec in vectors:
            with self._execute(
                f"CALL QUERY_VECTOR_INDEX('{label}', "
                f"'{label.lower()}_vec', $vec, $k{vec_kwargs_frag}) "
                f"RETURN node.{pk_key} AS {pk_key}, node, distance",
                {"vec": vec, "k": k, **vec_kwargs_bind},
            ) as r:
                cols = r.get_column_names()
                for row in r.get_all():
                    record = dict(zip(cols, row))
                    dist = record["distance"]
                    if threshold is not None and dist > threshold:
                        continue
                    node_id = record[pk_key]
                    if node_id not in best or dist < best[node_id]["distance"]:
                        best[node_id] = record

        ranked = sorted(best.values(), key=lambda r: r["distance"])[:k]
        return format_search_results(ranked, output_format)

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
        """BM25 fulltext search over entities of the given label.

        Uses Ladybug's FTS extension. The underlying index is a
        snapshot built from the table contents; rebuild happens at
        write time in `update_entities`, so search assumes the
        index is current.

        Args:
            conjunctive: When ``True``, the query requires every term
                to match (AND); the default (``False``) ORs them so
                any term match counts. Forwards to Ladybug's
                ``conjunctive`` kwarg.
            bm25_b: Optional override for BM25's ``b`` parameter —
                controls how aggressively document-length normalises
                the score. ``None`` keeps Ladybug's default (0.75).
        """
        label = sanitize_label(label)
        if label not in self._fts_columns:
            raise ValueError(
                f"No FTS index for label {label!r}: the entity model "
                f"declares no string-typed properties (or wasn't passed "
                f"in entity_models)."
            )
        if not text_or_texts:
            return format_search_results([], output_format)

        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts

        fts_kwargs_frag, fts_kwargs_bind = self._build_algo_kwargs(
            self._collect_fts_query_kwargs(conjunctive, bm25_b),
            self._FTS_QUERY_SUPPORTED_PARAMS,
        )

        # Best (highest) BM25 score per PK across all query texts. The
        # PK column is returned under its own name (no alias to
        # ``id``); the Python dedup keys off it directly.
        pk_key = self._pk_keys.get(label, "id")
        best: Dict[Any, Dict[str, Any]] = {}
        index_name = f"{label.lower()}_fts"
        for query_text in texts:
            with self._execute(
                f"CALL QUERY_FTS_INDEX('{label}', '{index_name}', $q, "
                f"top := $k{fts_kwargs_frag}) "
                f"RETURN node.{pk_key} AS {pk_key}, node, score",
                {"q": query_text, "k": k, **fts_kwargs_bind},
            ) as r:
                cols = r.get_column_names()
                for row in r.get_all():
                    record = dict(zip(cols, row))
                    score = record["score"]
                    if threshold is not None and score < threshold:
                        continue
                    node_id = record[pk_key]
                    if node_id not in best or score > best[node_id]["score"]:
                        best[node_id] = record

        ranked = sorted(best.values(), key=lambda r: -r["score"])[:k]
        return format_search_results(ranked, output_format)

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
        """Cypher-side regex match. Counterpart of DuckDB ``regex_search``.

        Uses Cypher's ``=~`` operator on each candidate string column;
        the column list comes from `_fts_columns` (the same
        derived list of string-typed properties FTS indexes) so the
        scoping rules stay consistent across both search families.
        Case-insensitive matching is handled inline via the ``(?i)``
        flag prefix — Ladybug forwards regexes to its underlying
        engine, so this is a regex flag, not a Cypher operator.

        Args:
            pattern: The regex pattern.
            label: The entity label to search within.
            fields: Optional whitelist of string fields to match
                against. Defaults to every indexed string field.
            case_sensitive: When ``False``, prefixes the pattern
                with ``(?i)`` for case-insensitive matching.
            k: Maximum number of rows returned.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        if not pattern:
            return format_search_results([], output_format)

        label = sanitize_label(label)
        pk_key = self._pk_keys.get(label, "id")
        string_cols = self._fts_columns.get(label, [])
        if fields is not None:
            requested = {
                sanitize_property_name(f)
                for f in fields
                if sanitize_property_name(f) is not None
            }
            cols = [c for c in string_cols if c in requested]
        else:
            cols = list(string_cols)
        if not cols:
            warnings.warn(
                f"Skipping regex search for {label!r}: no matching string fields."
            )
            return format_search_results([], output_format)

        effective_pattern = pattern if case_sensitive else f"(?i){pattern}"
        where = " OR ".join(f"n.{c} =~ $pattern" for c in cols)
        query = (
            f"MATCH (n:{label}) WHERE {where} "
            f"RETURN n.{pk_key} AS {pk_key}, n AS node LIMIT $k"
        )

        with self._execute(query, {"pattern": effective_pattern, "k": k}) as r:
            rows = r.get_all()
            cols_out = r.get_column_names()
        results = [dict(zip(cols_out, row)) for row in rows]
        return format_search_results(results, output_format)

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
        """RRF fusion of vector similarity + regex match.

        Sibling of `entity_hybrid_fts_search`. The regex side
        captures "exact textual shape" — useful when the user knows
        a substring / pattern that should appear, but the embedding
        alone doesn't surface it. Composition is RRF over per-source
        ranks, identical math to the FTS hybrid.

        Degenerates to plain `entity_regex_search` when no
        embedding model is configured (the vector half can't run),
        and to plain `entity_similarity_search` when no patterns
        are passed in.
        """
        label = sanitize_label(label)

        if not pattern_or_patterns:
            return await self.entity_similarity_search(
                text_or_texts,
                label=label,
                k=k,
                threshold=similarity_threshold,
                output_format=output_format,
            )

        patterns: List[str] = (
            [pattern_or_patterns]
            if isinstance(pattern_or_patterns, str)
            else list(pattern_or_patterns)
        )

        # Fallback: no embedding model → regex-only union across patterns.
        if not self.embedding_model:
            merged: Dict[Any, Dict[str, Any]] = {}
            for pattern in patterns:
                rows = await self.entity_regex_search(
                    pattern,
                    label=label,
                    fields=fields,
                    case_sensitive=case_sensitive,
                    k=k,
                    output_format="json",
                )
                for row in rows:
                    pk_value = row.get(self._pk_keys.get(label, "id"))
                    merged.setdefault(pk_value, row)
            results = list(merged.values())[:k]
            return format_search_results(results, output_format)

        if not text_or_texts:
            # Vector side has nothing to embed — fall back to a
            # union of regex matches across all patterns, same as
            # the no-embedding-model branch above.
            merged: Dict[Any, Dict[str, Any]] = {}
            for pattern in patterns:
                rows = await self.entity_regex_search(
                    pattern,
                    label=label,
                    fields=fields,
                    case_sensitive=case_sensitive,
                    k=k,
                    output_format="json",
                )
                for row in rows:
                    pk_value = row.get(self._pk_keys.get(label, "id"))
                    merged.setdefault(pk_value, row)
            return format_search_results(list(merged.values())[:k], output_format)

        texts: List[str] = (
            [text_or_texts] if isinstance(text_or_texts, str) else list(text_or_texts)
        )
        candidate_k = max(k * 5, k)
        pk_key = self._pk_keys.get(label, "id")

        # Vector side — one similarity_search call per query text,
        # widened to k*5 so the union has enough breadth for RRF.
        vec_best: Dict[Any, Dict[str, Any]] = {}
        for query_text in texts:
            try:
                rows = await self.entity_similarity_search(
                    query_text,
                    label=label,
                    k=candidate_k,
                    threshold=similarity_threshold,
                    output_format="json",
                )
            except Exception:
                rows = []
            for row in rows:
                pk_value = row[pk_key]
                # Keep the closest distance per id across query texts.
                prev = vec_best.get(pk_value)
                if prev is None or row["distance"] < prev["distance"]:
                    vec_best[pk_value] = row

        # Regex side — union over patterns, deduplicated by PK.
        rx_best: Dict[Any, Dict[str, Any]] = {}
        for pattern in patterns:
            try:
                rows = await self.entity_regex_search(
                    pattern,
                    label=label,
                    fields=fields,
                    case_sensitive=case_sensitive,
                    k=candidate_k,
                    output_format="json",
                )
            except Exception:
                rows = []
            for row in rows:
                rx_best.setdefault(row[pk_key], row)

        # RRF per-source ranks. Vector: ascending by distance (smaller
        # is better). Regex: there's no score, so rows are ranked in
        # the order the regex returned them — same shape DuckDB uses.
        vec_ranks = {
            pk: rank
            for rank, (pk, _) in enumerate(
                sorted(vec_best.items(), key=lambda x: x[1]["distance"]),
                start=1,
            )
        }
        rx_ranks = {pk: rank for rank, pk in enumerate(rx_best, start=1)}

        all_ids = set(vec_ranks) | set(rx_ranks)
        fused: List[Dict[str, Any]] = []
        for pk_value in all_ids:
            score = 0.0
            if pk_value in vec_ranks:
                score += 1.0 / (k_rank + vec_ranks[pk_value])
            if pk_value in rx_ranks:
                score += 1.0 / (k_rank + rx_ranks[pk_value])
            base = vec_best.get(pk_value) or rx_best.get(pk_value) or {}
            record: Dict[str, Any] = dict(base)
            record[pk_key] = pk_value
            record["rrf_score"] = score
            if pk_value in vec_best:
                record["distance"] = vec_best[pk_value]["distance"]
            fused.append(record)
        fused.sort(key=lambda r: -r["rrf_score"])
        return format_search_results(fused[:k], output_format)

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
        """Reciprocal-Rank-Fusion of vector + BM25 over entities of a label.

        Ladybug has no native ``QUERY_HYBRID_INDEX`` and rejects
        SQL-style window functions (``rank() OVER (...)``), so the
        fusion happens in Python. Per query text we issue a single
        ``UNION ALL``-ed Cypher statement that returns *both* the
        FTS-ranked rows and the vector-ranked rows tagged with a
        ``source`` column. Python then assigns per-source ranks and
        computes RRF — same end shape as DuckDB's
        ``hybrid_fts_search``, just with the fusion outside the DB.

        ``text_or_texts`` feeds the vector branch; ``keywords`` (when
        provided) feeds the BM25 branch instead — the two signals
        look for different things (semantic vs lexical) and the
        natural-language query that drives the vectors is usually
        not the keyword set you'd hand to BM25. When ``keywords`` is
        omitted, the text is reused for both branches so existing
        call sites keep working.

        Falls back to fulltext-only when no embedding model is
        configured (mirroring DuckDB). The fulltext branch in that
        fallback uses ``keywords`` when provided, ``text_or_texts``
        otherwise.

        Args:
            ef_search: Optional HNSW ``efs`` for the vector branch
                (search-time candidate depth — higher = better recall).
            conjunctive: AND vs OR mode for the BM25 branch.
            bm25_b: Optional override for BM25's ``b`` parameter.
        """
        label = sanitize_label(label)
        if label not in self._fts_columns:
            raise ValueError(
                f"No FTS index for label {label!r}: the entity model "
                f"declares no string-typed properties (or wasn't passed "
                f"in entity_models)."
            )
        if not text_or_texts:
            return format_search_results([], output_format)

        if not self.embedding_model:
            # Fulltext-only fallback. Prefer explicit keywords when
            # the caller passed them — that's what the BM25 branch
            # would have used in the full hybrid path anyway. Tag
            # each row with ``rrf_score`` (set to the BM25 score) so
            # the result shape matches the full-hybrid path — callers
            # can always read ``rrf_score`` without branching.
            fts_rows = await self.entity_fulltext_search(
                keywords if keywords is not None else text_or_texts,
                label=label,
                k=k,
                threshold=fulltext_threshold,
                conjunctive=conjunctive,
                bm25_b=bm25_b,
                output_format="json",
            )
            for row in fts_rows:
                row.setdefault("rrf_score", row.get("score", 0.0))
                row.setdefault("fulltext_score", row.get("score", 0.0))
            return format_search_results(fts_rows, output_format)

        texts, keyword_list = align_keywords(text_or_texts, keywords)
        embeddings = await self.embedding_model(EmbeddingRequest(texts=texts))
        vectors = embeddings["embeddings"]

        fts_best: Dict[str, float] = {}
        vec_best: Dict[str, float] = {}
        nodes: Dict[str, Any] = {}

        pk_key = self._pk_keys.get(label, "id")
        fts_index = f"{label.lower()}_fts"
        vec_index = f"{label.lower()}_vec"
        # Widen each branch's candidate pool to 5*k before RRF fusion.
        # If we asked each branch for only k results, the actual top-k
        # by combined score is often outside one branch's top-k —
        # standard RRF practice and what DuckDB's adapter does too.
        candidate_k = max(k * 5, k)
        vec_kwargs_frag, vec_kwargs_bind = self._build_algo_kwargs(
            self._collect_vector_query_kwargs(ef_search),
            self._VECTOR_QUERY_SUPPORTED_PARAMS,
        )
        fts_kwargs_frag, fts_kwargs_bind = self._build_algo_kwargs(
            self._collect_fts_query_kwargs(conjunctive, bm25_b),
            self._FTS_QUERY_SUPPORTED_PARAMS,
        )
        for keyword_text, vec in zip(keyword_list, vectors):
            query = (
                f"CALL QUERY_FTS_INDEX('{label}', '{fts_index}', $q, "
                f"top := $candidate_k{fts_kwargs_frag}) "
                f"RETURN node.{pk_key} AS {pk_key}, node AS node, "
                f"score AS metric, 'fts' AS source "
                f"UNION ALL "
                f"CALL QUERY_VECTOR_INDEX('{label}', '{vec_index}', "
                f"$vec, $candidate_k{vec_kwargs_frag}) "
                f"RETURN node.{pk_key} AS {pk_key}, node AS node, "
                f"distance AS metric, 'vec' AS source"
            )
            with self._execute(
                query,
                {
                    "q": keyword_text,
                    "vec": vec,
                    "candidate_k": candidate_k,
                    **vec_kwargs_bind,
                    **fts_kwargs_bind,
                },
            ) as r:
                cols = r.get_column_names()
                for row in r.get_all():
                    record = dict(zip(cols, row))
                    node_id = record[pk_key]
                    metric = record["metric"]
                    if record["source"] == "fts":
                        if fulltext_threshold is not None and metric < fulltext_threshold:
                            continue
                        # Higher BM25 is better → keep the max.
                        prev = fts_best.get(node_id)
                        if prev is None or metric > prev:
                            fts_best[node_id] = metric
                    else:
                        if (
                            similarity_threshold is not None
                            and metric > similarity_threshold
                        ):
                            continue
                        # Lower distance is better → keep the min.
                        prev = vec_best.get(node_id)
                        if prev is None or metric < prev:
                            vec_best[node_id] = metric
                    nodes[node_id] = record["node"]

        # Convert per-source best scores to per-source ranks.
        fts_ranks = {
            node_id: rank
            for rank, (node_id, _) in enumerate(
                sorted(fts_best.items(), key=lambda x: -x[1]), start=1
            )
        }
        vec_ranks = {
            node_id: rank
            for rank, (node_id, _) in enumerate(
                sorted(vec_best.items(), key=lambda x: x[1]), start=1
            )
        }

        all_ids = set(fts_ranks) | set(vec_ranks)
        rrf_scores: List[Dict[str, Any]] = []
        for node_id in all_ids:
            score = 0.0
            if node_id in fts_ranks:
                score += 1.0 / (k_rank + fts_ranks[node_id])
            if node_id in vec_ranks:
                score += 1.0 / (k_rank + vec_ranks[node_id])
            record: Dict[str, Any] = {
                pk_key: node_id,
                "node": nodes[node_id],
                "rrf_score": score,
            }
            if node_id in fts_best:
                record["fulltext_score"] = fts_best[node_id]
            if node_id in vec_best:
                record["distance"] = vec_best[node_id]
            rrf_scores.append(record)

        rrf_scores.sort(key=lambda r: -r["rrf_score"])
        return format_search_results(rrf_scores[:k], output_format)

    # ------------------------------------------------------------------
    # Relation-level search
    # ------------------------------------------------------------------

    def _resolve_endpoint_labels(self, rel_label: str) -> tuple:
        """Look up ``(subj_label, obj_label)`` for a REL table.

        Uses ``CALL SHOW_CONNECTION`` — same introspection path
        `_rel_table_to_json_schema` uses. Ladybug raises a
        binder-level error when the table doesn't exist or isn't a
        rel; this wrapper catches that and re-raises as a clear
        ``ValueError`` so call sites get an actionable message
        instead of a generic ``RuntimeError`` from the C++ layer.
        """
        try:
            rows = self._con.execute(
                f"CALL SHOW_CONNECTION('{rel_label}') RETURN *"
            ).get_all()
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Relation label {rel_label!r} couldn't be resolved via "
                f"SHOW_CONNECTION — is it registered as a REL table? "
                f"Pass it via `relation_models` at construction time. "
                f"(underlying error: {e})"
            ) from e
        if not rows:
            raise ValueError(
                f"Relation label {rel_label!r} has no SHOW_CONNECTION row "
                f"— is it registered as a REL table? Pass it via "
                f"`relation_models` at construction time."
            )
        return rows[0][0], rows[0][1]

    def _flatten_endpoint_pks(
        self,
        subj_pk: str,
        obj_pk: str,
        subj_value: Any,
        obj_value: Any,
    ) -> Dict[str, Any]:
        """Surface subj/obj PK values as flat columns on a result row.

        When the two endpoints share a PK column name (e.g. both
        Person and City use ``name``), the object's PK lands under
        ``{name}_1`` to avoid clobbering the subject's. Otherwise
        each PK occupies its own real column name. Mirrors what
        DuckDB's ``SELECT *`` from a JOIN would do via the engine's
        own column-deduplication.
        """
        out: Dict[str, Any] = {subj_pk: subj_value}
        if obj_pk == subj_pk:
            out[f"{obj_pk}_1"] = obj_value
        else:
            out[obj_pk] = obj_value
        return out

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

        For each query vector we issue a single ``UNION ALL`` Cypher
        statement: one branch vector-searches the subject side,
        joins through to its outgoing edges of ``label``, and returns
        the ``(subj, rel, obj)`` triple tagged with ``matched_on =
        'subj'``; the second branch does the symmetric thing on the
        object side with ``matched_on = 'obj'``. Python then
        deduplicates by ``(subj_pk, obj_pk)``, keeping the smaller
        distance and tagging ``matched_on = 'both'`` when an edge
        surfaced on both branches — useful for downstream ranking
        (edges where both endpoints look relevant tend to be the
        most interesting).

        Args:
            text_or_texts: Query text or list of query texts.
            label: The relation label (edge type) to search within.
            k: Maximum number of results.
            threshold: Optional vector-distance threshold applied to
                each endpoint search before the union.
            ef_search: Optional HNSW ``efs`` — search-time candidate
                depth applied to both endpoint vector searches.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
        """
        if not self.embedding_model:
            raise ValueError(
                "relation_similarity_search requires an embedding_model "
                "on the LadybugAdapter."
            )
        label = sanitize_label(label)
        subj_label, obj_label = self._resolve_endpoint_labels(label)
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        if not text_or_texts:
            return format_search_results([], output_format)
        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts
        embeddings = await self.embedding_model(EmbeddingRequest(texts=texts))
        vectors = embeddings["embeddings"]

        subj_vec_index = f"{subj_label.lower()}_vec"
        obj_vec_index = f"{obj_label.lower()}_vec"

        vec_kwargs_frag, vec_kwargs_bind = self._build_algo_kwargs(
            self._collect_vector_query_kwargs(ef_search),
            self._VECTOR_QUERY_SUPPORTED_PARAMS,
        )

        # Aliasing the PK twice would clash when subj_pk == obj_pk
        # (both endpoints use ``name``). The unique column names
        # ``__subj_pk`` / ``__obj_pk`` decouple the Cypher result
        # from the user-facing PK column names — Python re-flattens
        # them with collision handling in `_flatten_endpoint_pks`.
        query = (
            f"CALL QUERY_VECTOR_INDEX('{subj_label}', '{subj_vec_index}', "
            f"$vec, $k{vec_kwargs_frag}) "
            f"YIELD node AS s, distance AS d "
            f"MATCH (s)-[r:{label}]->(o:{obj_label}) "
            f"RETURN s AS subj, r AS rel, o AS obj, d AS distance, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk, "
            f"'subj' AS matched_on "
            f"UNION ALL "
            f"CALL QUERY_VECTOR_INDEX('{obj_label}', '{obj_vec_index}', "
            f"$vec, $k{vec_kwargs_frag}) "
            f"YIELD node AS o, distance AS d "
            f"MATCH (s:{subj_label})-[r:{label}]->(o) "
            f"RETURN s AS subj, r AS rel, o AS obj, d AS distance, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk, "
            f"'obj' AS matched_on"
        )

        # Per-edge best distance + which side(s) fired across all
        # query vectors. ``matched_sides`` is a set so re-runs from
        # different queries accumulate the "both"-ness correctly.
        best: Dict[tuple, Dict[str, Any]] = {}
        sides: Dict[tuple, set] = {}
        for vec in vectors:
            with self._execute(query, {"vec": vec, "k": k, **vec_kwargs_bind}) as r:
                cols = r.get_column_names()
                for row in r.get_all():
                    record = dict(zip(cols, row))
                    dist = record["distance"]
                    if threshold is not None and dist > threshold:
                        continue
                    key = (record["__subj_pk"], record["__obj_pk"])
                    sides.setdefault(key, set()).add(record["matched_on"])
                    if key not in best or dist < best[key]["distance"]:
                        best[key] = record

        ranked = sorted(best.values(), key=lambda r: r["distance"])[:k]
        results: List[Dict[str, Any]] = []
        for record in ranked:
            key = (record["__subj_pk"], record["__obj_pk"])
            side_set = sides[key]
            matched_on = "both" if len(side_set) > 1 else next(iter(side_set))
            row_out: Dict[str, Any] = {
                "subj": record["subj"],
                "rel": record["rel"],
                "obj": record["obj"],
                "distance": record["distance"],
                "matched_on": matched_on,
            }
            row_out.update(
                self._flatten_endpoint_pks(
                    subj_pk, obj_pk, record["__subj_pk"], record["__obj_pk"]
                )
            )
            results.append(row_out)
        return format_search_results(results, output_format)

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

        Composed via `entity_fulltext_search` on each endpoint
        side. Per matched edge, the final ``score`` is the sum of the
        subject-side and object-side BM25 scores. Either-endpoint
        union: an edge surfaces if either endpoint matched.

        Args:
            text_or_texts: Query text or list of query texts.
            label: The relation label (edge type) to search within.
            k: Maximum number of results.
            threshold: Optional minimum BM25 threshold applied per
                endpoint before the union.
            conjunctive: AND-mode query (every term must match).
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        label = sanitize_label(label)
        subj_label, obj_label = self._resolve_endpoint_labels(label)
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        if not text_or_texts:
            return format_search_results([], output_format)

        subj_hits = await self.entity_fulltext_search(
            text_or_texts,
            label=subj_label,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )
        obj_hits = await self.entity_fulltext_search(
            text_or_texts,
            label=obj_label,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )

        subj_score: Dict[Any, float] = {
            hit[subj_pk]: hit.get("score", 0.0) for hit in subj_hits
        }
        obj_score: Dict[Any, float] = {
            hit[obj_pk]: hit.get("score", 0.0) for hit in obj_hits
        }

        if not subj_score and not obj_score:
            return format_search_results([], output_format)

        rows = self._con.execute(
            f"MATCH (s:{subj_label})-[r:{label}]->(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks OR o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, r AS rel, o AS obj, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_score.keys()),
                "obj_pks": list(obj_score.keys()),
            },
        ).get_all()

        results: List[Dict[str, Any]] = []
        for row in rows:
            s_node, r_edge, o_node, spk_val, opk_val = row
            s_hit = spk_val in subj_score
            o_hit = opk_val in obj_score
            subj_contribution = subj_score.get(spk_val, 0.0) if s_hit else 0.0
            obj_contribution = obj_score.get(opk_val, 0.0) if o_hit else 0.0
            row_out: Dict[str, Any] = {
                "subj": s_node,
                "rel": r_edge,
                "obj": o_node,
                "score": subj_contribution + obj_contribution,
                "subj_score": subj_contribution,
                "obj_score": obj_contribution,
                "matched_on": (
                    "both" if s_hit and o_hit else ("subj" if s_hit else "obj")
                ),
            }
            row_out.update(self._flatten_endpoint_pks(subj_pk, obj_pk, spk_val, opk_val))
            results.append(row_out)

        results.sort(key=lambda r: -r["score"])
        return format_search_results(results[:k], output_format)

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

        Composed via `entity_regex_search` on each endpoint side.
        Regex hits have no continuous score; ranking is binary
        (matched or not) with a slight bias toward edges that hit on
        both endpoints over edges that hit on only one — exposed via
        ``score`` (2.0 for both, 1.0 for one) and ``matched_on``.

        Args:
            pattern: The regex pattern.
            label: The relation label (edge type) to search within.
            fields: Optional whitelist of fields, applied to both
                endpoints.
            case_sensitive: When ``False``, matches case-insensitively.
            k: Maximum number of rows.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        if not pattern:
            return format_search_results([], output_format)

        label = sanitize_label(label)
        subj_label, obj_label = self._resolve_endpoint_labels(label)
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        subj_hits = await self.entity_regex_search(
            pattern,
            label=subj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
        )
        obj_hits = await self.entity_regex_search(
            pattern,
            label=obj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
        )

        subj_pks = {hit[subj_pk] for hit in subj_hits}
        obj_pks = {hit[obj_pk] for hit in obj_hits}

        if not subj_pks and not obj_pks:
            return format_search_results([], output_format)

        rows = self._con.execute(
            f"MATCH (s:{subj_label})-[r:{label}]->(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks OR o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, r AS rel, o AS obj, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_pks),
                "obj_pks": list(obj_pks),
            },
        ).get_all()

        results: List[Dict[str, Any]] = []
        for row in rows:
            s_node, r_edge, o_node, spk_val, opk_val = row
            s_hit = spk_val in subj_pks
            o_hit = opk_val in obj_pks
            row_out: Dict[str, Any] = {
                "subj": s_node,
                "rel": r_edge,
                "obj": o_node,
                "score": float(int(s_hit) + int(o_hit)),
                "matched_on": (
                    "both" if s_hit and o_hit else ("subj" if s_hit else "obj")
                ),
            }
            row_out.update(self._flatten_endpoint_pks(subj_pk, obj_pk, spk_val, opk_val))
            results.append(row_out)

        results.sort(key=lambda r: -r["score"])
        return format_search_results(results[:k], output_format)

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

        Composed via `entity_hybrid_regex_search` on each
        endpoint side. Per matched edge, the final ``rrf_score`` is
        the sum of the subject's and the object's hybrid scores —
        same 4-source-RRF reduction as `relation_hybrid_fts_search`.

        Falls through to `relation_similarity_search` when no
        patterns are supplied.

        Args:
            text_or_texts: Query text or list of query texts for the
                vector branch.
            pattern_or_patterns: Regex pattern (or list) for the
                regex branch. ``None`` skips the regex side.
            label: The relation label (edge type).
            fields: Forwarded to `entity_regex_search`.
            case_sensitive: Forwarded to `entity_regex_search`.
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        label = sanitize_label(label)

        if not pattern_or_patterns:
            return await self.relation_similarity_search(
                text_or_texts,
                label=label,
                k=k,
                threshold=similarity_threshold,
                output_format=output_format,
            )

        subj_label, obj_label = self._resolve_endpoint_labels(label)
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        subj_hits = await self.entity_hybrid_regex_search(
            text_or_texts=text_or_texts,
            pattern_or_patterns=pattern_or_patterns,
            label=subj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
        )
        obj_hits = await self.entity_hybrid_regex_search(
            text_or_texts=text_or_texts,
            pattern_or_patterns=pattern_or_patterns,
            label=obj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
        )

        subj_score: Dict[Any, float] = {
            hit[subj_pk]: self._hybrid_hit_score(hit) for hit in subj_hits
        }
        obj_score: Dict[Any, float] = {
            hit[obj_pk]: self._hybrid_hit_score(hit) for hit in obj_hits
        }

        if not subj_score and not obj_score:
            return format_search_results([], output_format)

        rows = self._con.execute(
            f"MATCH (s:{subj_label})-[r:{label}]->(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks OR o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, r AS rel, o AS obj, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_score.keys()),
                "obj_pks": list(obj_score.keys()),
            },
        ).get_all()

        results: List[Dict[str, Any]] = []
        for row in rows:
            s_node, r_edge, o_node, spk_val, opk_val = row
            s_hit = spk_val in subj_score
            o_hit = opk_val in obj_score
            subj_contribution = subj_score.get(spk_val, 0.0) if s_hit else 0.0
            obj_contribution = obj_score.get(opk_val, 0.0) if o_hit else 0.0
            row_out: Dict[str, Any] = {
                "subj": s_node,
                "rel": r_edge,
                "obj": o_node,
                "rrf_score": subj_contribution + obj_contribution,
                "subj_rrf_score": subj_contribution,
                "obj_rrf_score": obj_contribution,
                "matched_on": (
                    "both" if s_hit and o_hit else ("subj" if s_hit else "obj")
                ),
            }
            row_out.update(self._flatten_endpoint_pks(subj_pk, obj_pk, spk_val, opk_val))
            results.append(row_out)

        results.sort(key=lambda r: -r["rrf_score"])
        return format_search_results(results[:k], output_format)

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
        """Variable-length-path counterpart to a single-hop AND-search.

        For each ``(subj_vec, obj_vec)`` query pair we issue one
        chained-CALL Cypher statement:

          1. Vector-search the subject label → ``s``.
          2. ``WITH`` carry through (with optional threshold filter).
          3. Vector-search the object label → ``o``.
          4. ``WITH`` carry through (with optional threshold filter).
          5. ``MATCH p = (s)-[*min..max]->(o)`` — any path of
             valid hop count. When ``label`` is set, the pattern
             tightens to ``[:label*min..max]`` so every hop must be
             of that type.
          6. Return ``nodes(p)`` / ``rels(p)`` / ``length(p)`` along
             with the two distances.

        Same chained-CALL shape `_merge_relation_by_vector`
        uses for upserts — Ladybug parses chained CALLs as long as
        each one is separated by a ``WITH``. The variable-length
        pattern is Cypher's standard ``*<min>..<max>`` form,
        verified against Ladybug 0.16.

        Args:
            subj_text_or_texts: Query text (or list) for the subject.
            obj_text_or_texts: Query text (or list) for the object.
            subj_label: Entity label for the subject endpoint
                (selects its vector index).
            obj_label: Entity label for the object endpoint.
            label: Optional rel-label constraint applied to every
                hop. ``None`` (default) accepts any edge type.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            subj_threshold: Vector-distance ceiling for the subject.
            obj_threshold: Vector-distance ceiling for the object.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        if not self.embedding_model:
            raise ValueError(
                "path_similarity_search requires an embedding_model "
                "on the LadybugAdapter."
            )
        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                f"Invalid hop range: min_hops={min_hops}, "
                f"max_hops={max_hops}. Require 1 <= min_hops <= max_hops."
            )

        subj_label = sanitize_label(subj_label)
        obj_label = sanitize_label(obj_label)
        rel_label = sanitize_label(label) if label is not None else None
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        if not subj_text_or_texts or not obj_text_or_texts:
            return format_search_results([], output_format)

        subj_texts = (
            [subj_text_or_texts]
            if not isinstance(subj_text_or_texts, list)
            else subj_text_or_texts
        )
        obj_texts = (
            [obj_text_or_texts]
            if not isinstance(obj_text_or_texts, list)
            else obj_text_or_texts
        )
        if len(subj_texts) != len(obj_texts):
            raise ValueError(
                f"`subj_text_or_texts` and `obj_text_or_texts` must be "
                f"the same length (got {len(subj_texts)} vs "
                f"{len(obj_texts)}). Each pair represents one "
                f"(subj, obj) query."
            )

        subj_emb = (await self.embedding_model(EmbeddingRequest(texts=subj_texts)))[
            "embeddings"
        ]
        obj_emb = (await self.embedding_model(EmbeddingRequest(texts=obj_texts)))[
            "embeddings"
        ]

        subj_vec_index = f"{subj_label.lower()}_vec"
        obj_vec_index = f"{obj_label.lower()}_vec"

        vec_kwargs_frag, vec_kwargs_bind = self._build_algo_kwargs(
            self._collect_vector_query_kwargs(ef_search),
            self._VECTOR_QUERY_SUPPORTED_PARAMS,
        )

        # ``WHERE TRUE`` is a no-op the parser accepts and lets the
        # query shape stay stable across thresholded / unthresholded
        # call sites (no string-template branching needed).
        subj_filter = "subj_dist <= $subj_max" if subj_threshold is not None else "TRUE"
        obj_filter = "obj_dist <= $obj_max" if obj_threshold is not None else "TRUE"

        # Variable-length pattern. Labeled form constrains every hop
        # in the path to the given rel label; unlabeled accepts any.
        if rel_label is not None:
            path_pattern = f"-[:{rel_label}*{min_hops}..{max_hops}]->"
        else:
            path_pattern = f"-[*{min_hops}..{max_hops}]->"

        query = (
            f"CALL QUERY_VECTOR_INDEX('{subj_label}', '{subj_vec_index}', "
            f"$subj_vec, $k{vec_kwargs_frag}) "
            f"YIELD node AS s, distance AS subj_dist "
            f"WITH s, subj_dist WHERE {subj_filter} "
            f"CALL QUERY_VECTOR_INDEX('{obj_label}', '{obj_vec_index}', "
            f"$obj_vec, $k{vec_kwargs_frag}) "
            f"YIELD node AS o, distance AS obj_dist "
            f"WITH s, o, subj_dist, obj_dist WHERE {obj_filter} "
            f"MATCH p = (s){path_pattern}(o) "
            f"RETURN s AS subj, o AS obj, "
            f"nodes(p) AS nodes, rels(p) AS rels, length(p) AS length, "
            f"subj_dist AS subj_distance, obj_dist AS obj_distance, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk"
        )

        # Per-path dedup key: (subj_pk, obj_pk, length). Same
        # endpoints can be connected by paths of different lengths;
        # we surface each length once with its best (smallest) sum
        # of endpoint distances across query pairs.
        best: Dict[tuple, Dict[str, Any]] = {}
        for sv, ov in zip(subj_emb, obj_emb):
            params = {
                "subj_vec": sv,
                "obj_vec": ov,
                "k": k,
                **vec_kwargs_bind,
            }
            if subj_threshold is not None:
                params["subj_max"] = subj_threshold
            if obj_threshold is not None:
                params["obj_max"] = obj_threshold
            with self._execute(query, params) as r:
                cols = r.get_column_names()
                for row in r.get_all():
                    record = dict(zip(cols, row))
                    key = (
                        record["__subj_pk"],
                        record["__obj_pk"],
                        record["length"],
                    )
                    score = record["subj_distance"] + record["obj_distance"]
                    if key not in best or score < (
                        best[key]["subj_distance"] + best[key]["obj_distance"]
                    ):
                        best[key] = record

        ranked = sorted(
            best.values(),
            key=lambda r: (
                r["subj_distance"] + r["obj_distance"],
                r["length"],
            ),
        )[:k]
        results: List[Dict[str, Any]] = []
        for record in ranked:
            row_out: Dict[str, Any] = {
                "subj": record["subj"],
                "obj": record["obj"],
                "nodes": record["nodes"],
                "rels": record["rels"],
                "length": record["length"],
                "subj_distance": record["subj_distance"],
                "obj_distance": record["obj_distance"],
            }
            row_out.update(
                self._flatten_endpoint_pks(
                    subj_pk, obj_pk, record["__subj_pk"], record["__obj_pk"]
                )
            )
            results.append(row_out)
        return format_search_results(results, output_format)

    # ------------------------------------------------------------------
    # Relation-level hybrid search (vec + BM25 fusion)
    # ------------------------------------------------------------------

    @staticmethod
    def _hybrid_hit_score(hit: Dict[str, Any]) -> float:
        """Extract the per-entity hybrid score from a hit row.

        `entity_hybrid_fts_search` returns ``"rrf_score"`` when
        an embedding model is configured; on the FTS-only fallback
        path it returns ``"score"`` (the raw BM25). This helper
        picks whichever is present so call sites stay shape-agnostic
        across both modes.
        """
        if "rrf_score" in hit:
            return hit["rrf_score"]
        return hit.get("score", 0.0)

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
        """RRF of vector + BM25 fulltext over relations.

        Composed via `entity_hybrid_fts_search` on each
        endpoint side. Per matched edge, the final ``rrf_score`` is
        the sum of the subject's and the object's hybrid scores —
        equivalent to a single 4-source RRF over the four underlying
        rankings, because each endpoint participates in exactly one
        subj-side rank and one obj-side rank.

        ``text_or_texts`` drives the vector branch; ``keywords``
        (when provided) drives the BM25 branch on both endpoints.
        When ``keywords`` is omitted, the text is reused for both.

        Endpoint resolution uses `_resolve_endpoint_labels` so
        a single ``label`` argument suffices, mirroring
        `relation_similarity_search`. Falls back to fulltext-
        only when no embedding model is configured.
        """
        label = sanitize_label(label)
        subj_label, obj_label = self._resolve_endpoint_labels(label)
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        if not text_or_texts:
            return format_search_results([], output_format)

        # Composed hybrid on each side. ``entity_hybrid_fts_search``
        # raises when the endpoint has no FTS index — that's the
        # right error to surface upstream too, so let it propagate.
        subj_hits = await self.entity_hybrid_fts_search(
            text_or_texts=text_or_texts,
            keywords=keywords,
            label=subj_label,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )
        obj_hits = await self.entity_hybrid_fts_search(
            text_or_texts=text_or_texts,
            keywords=keywords,
            label=obj_label,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )

        subj_score: Dict[Any, float] = {
            hit[subj_pk]: self._hybrid_hit_score(hit) for hit in subj_hits
        }
        obj_score: Dict[Any, float] = {
            hit[obj_pk]: self._hybrid_hit_score(hit) for hit in obj_hits
        }

        # No candidates on either side → nothing to score. Bail out
        # early so the IN-clause below doesn't run with an empty list
        # (Cypher would still evaluate it correctly, but the trip is
        # wasted).
        if not subj_score and not obj_score:
            return format_search_results([], output_format)

        # Pull all edges whose subj is among the hits OR whose obj
        # is among the hits — single Cypher round-trip, with the
        # full subj/obj/edge structs so the result row matches the
        # similarity-version's shape. ``OR`` is the union semantics
        # (matches `relation_similarity_search`).
        rows = self._con.execute(
            f"MATCH (s:{subj_label})-[r:{label}]->(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks OR o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, r AS rel, o AS obj, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_score.keys()),
                "obj_pks": list(obj_score.keys()),
            },
        ).get_all()

        results: List[Dict[str, Any]] = []
        for row in rows:
            s_node, r_edge, o_node, spk_val, opk_val = row
            s_hit = spk_val in subj_score
            o_hit = opk_val in obj_score
            subj_contribution = subj_score.get(spk_val, 0.0) if s_hit else 0.0
            obj_contribution = obj_score.get(opk_val, 0.0) if o_hit else 0.0
            row_out: Dict[str, Any] = {
                "subj": s_node,
                "rel": r_edge,
                "obj": o_node,
                "rrf_score": subj_contribution + obj_contribution,
                "subj_rrf_score": subj_contribution,
                "obj_rrf_score": obj_contribution,
                "matched_on": (
                    "both" if s_hit and o_hit else ("subj" if s_hit else "obj")
                ),
            }
            row_out.update(self._flatten_endpoint_pks(subj_pk, obj_pk, spk_val, opk_val))
            results.append(row_out)

        results.sort(key=lambda r: -r["rrf_score"])
        return format_search_results(results[:k], output_format)

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
        """Hybrid (vec + BM25) variable-length path search, AND semantics.

        Each side is hybrid-searched independently; the chain's
        ``rrf_score`` is ``subj_rrf_score + obj_rrf_score`` (the
        4-source RRF identity). Paths are scoped by ``min_hops`` /
        ``max_hops`` and optionally constrained to a single rel
        label per hop. Falls back to fulltext-only when no
        embedding model is configured.

        ``subj_text_or_texts`` / ``obj_text_or_texts`` drive the
        vector branches; ``subj_keywords`` / ``obj_keywords`` (when
        provided) drive the BM25 branches on their respective
        endpoints. Per-side keyword inputs are paired with their
        corresponding text input by position; when omitted, the
        text is reused for both branches on that side.
        """
        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                f"Invalid hop range: min_hops={min_hops}, "
                f"max_hops={max_hops}. Require 1 <= min_hops <= max_hops."
            )
        subj_label = sanitize_label(subj_label)
        obj_label = sanitize_label(obj_label)
        rel_label = sanitize_label(label) if label is not None else None
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        if not subj_text_or_texts or not obj_text_or_texts:
            return format_search_results([], output_format)

        subj_hits = await self.entity_hybrid_fts_search(
            text_or_texts=subj_text_or_texts,
            keywords=subj_keywords,
            label=subj_label,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )
        obj_hits = await self.entity_hybrid_fts_search(
            text_or_texts=obj_text_or_texts,
            keywords=obj_keywords,
            label=obj_label,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
            ef_search=ef_search,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )

        subj_score: Dict[Any, float] = {
            hit[subj_pk]: self._hybrid_hit_score(hit) for hit in subj_hits
        }
        obj_score: Dict[Any, float] = {
            hit[obj_pk]: self._hybrid_hit_score(hit) for hit in obj_hits
        }
        # AND-semantics: both sides must produce candidates.
        if not subj_score or not obj_score:
            return format_search_results([], output_format)

        if rel_label is not None:
            path_pattern = f"-[:{rel_label}*{min_hops}..{max_hops}]->"
        else:
            path_pattern = f"-[*{min_hops}..{max_hops}]->"

        rows = self._con.execute(
            f"MATCH p = (s:{subj_label}){path_pattern}(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks AND o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, o AS obj, "
            f"nodes(p) AS nodes, rels(p) AS rels, length(p) AS length, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_score.keys()),
                "obj_pks": list(obj_score.keys()),
            },
        ).get_all()

        # Dedup key includes length: same endpoints connected by
        # different-length paths surface as distinct rows, same as
        # `path_similarity_search`.
        best: Dict[tuple, Dict[str, Any]] = {}
        for row in rows:
            s_node, o_node, ns, rs, length, spk_val, opk_val = row
            sub_c = subj_score[spk_val]
            obj_c = obj_score[opk_val]
            rrf = sub_c + obj_c
            key = (spk_val, opk_val, length)
            if key not in best or rrf > best[key]["rrf_score"]:
                best[key] = {
                    "subj": s_node,
                    "obj": o_node,
                    "nodes": ns,
                    "rels": rs,
                    "length": length,
                    "subj_rrf_score": sub_c,
                    "obj_rrf_score": obj_c,
                    "rrf_score": rrf,
                    "__subj_pk": spk_val,
                    "__obj_pk": opk_val,
                }

        ranked = sorted(
            best.values(),
            key=lambda r: (-r["rrf_score"], r["length"]),
        )[:k]
        results: List[Dict[str, Any]] = []
        for entry in ranked:
            row_out: Dict[str, Any] = {
                "subj": entry["subj"],
                "obj": entry["obj"],
                "nodes": entry["nodes"],
                "rels": entry["rels"],
                "length": entry["length"],
                "rrf_score": entry["rrf_score"],
                "subj_rrf_score": entry["subj_rrf_score"],
                "obj_rrf_score": entry["obj_rrf_score"],
            }
            row_out.update(
                self._flatten_endpoint_pks(
                    subj_pk, obj_pk, entry["__subj_pk"], entry["__obj_pk"]
                )
            )
            results.append(row_out)
        return format_search_results(results, output_format)

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

        Same shape as `path_similarity_search` but driven by
        BM25 fulltext on each endpoint instead of vector similarity.
        Per matched path the ``score`` is the sum of the
        subject-side and object-side BM25 scores.

        Args:
            subj_text_or_texts: Keyword query (or list) for the
                subject endpoint.
            obj_text_or_texts: Keyword query (or list) for the
                object endpoint.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint applied to every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            threshold: Optional minimum BM25 threshold applied per endpoint.
            conjunctive: AND-mode query (every term must match).
            bm25_b: Optional override for BM25's ``b`` parameter.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                f"Invalid hop range: min_hops={min_hops}, "
                f"max_hops={max_hops}. Require 1 <= min_hops <= max_hops."
            )
        subj_label = sanitize_label(subj_label)
        obj_label = sanitize_label(obj_label)
        rel_label = sanitize_label(label) if label is not None else None
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        if not subj_text_or_texts or not obj_text_or_texts:
            return format_search_results([], output_format)

        subj_hits = await self.entity_fulltext_search(
            subj_text_or_texts,
            label=subj_label,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )
        obj_hits = await self.entity_fulltext_search(
            obj_text_or_texts,
            label=obj_label,
            k=k,
            threshold=threshold,
            conjunctive=conjunctive,
            bm25_b=bm25_b,
        )

        subj_score: Dict[Any, float] = {
            hit[subj_pk]: hit.get("score", 0.0) for hit in subj_hits
        }
        obj_score: Dict[Any, float] = {
            hit[obj_pk]: hit.get("score", 0.0) for hit in obj_hits
        }
        # AND-semantics: both sides must produce candidates.
        if not subj_score or not obj_score:
            return format_search_results([], output_format)

        if rel_label is not None:
            path_pattern = f"-[:{rel_label}*{min_hops}..{max_hops}]->"
        else:
            path_pattern = f"-[*{min_hops}..{max_hops}]->"

        rows = self._con.execute(
            f"MATCH p = (s:{subj_label}){path_pattern}(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks AND o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, o AS obj, "
            f"nodes(p) AS nodes, rels(p) AS rels, length(p) AS length, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_score.keys()),
                "obj_pks": list(obj_score.keys()),
            },
        ).get_all()

        best: Dict[tuple, Dict[str, Any]] = {}
        for row in rows:
            s_node, o_node, ns, rs, length, spk_val, opk_val = row
            sub_c = subj_score[spk_val]
            obj_c = obj_score[opk_val]
            total = sub_c + obj_c
            key = (spk_val, opk_val, length)
            if key not in best or total > best[key]["score"]:
                best[key] = {
                    "subj": s_node,
                    "obj": o_node,
                    "nodes": ns,
                    "rels": rs,
                    "length": length,
                    "subj_score": sub_c,
                    "obj_score": obj_c,
                    "score": total,
                    "__subj_pk": spk_val,
                    "__obj_pk": opk_val,
                }

        ranked = sorted(
            best.values(),
            key=lambda r: (-r["score"], r["length"]),
        )[:k]
        results: List[Dict[str, Any]] = []
        for entry in ranked:
            row_out: Dict[str, Any] = {
                "subj": entry["subj"],
                "obj": entry["obj"],
                "nodes": entry["nodes"],
                "rels": entry["rels"],
                "length": entry["length"],
                "score": entry["score"],
                "subj_score": entry["subj_score"],
                "obj_score": entry["obj_score"],
            }
            row_out.update(
                self._flatten_endpoint_pks(
                    subj_pk, obj_pk, entry["__subj_pk"], entry["__obj_pk"]
                )
            )
            results.append(row_out)
        return format_search_results(results, output_format)

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

        Both endpoints must match their respective regex pattern;
        rows surface only when at least one path of valid length
        connects a subject hit to an object hit. Regex uses RE2 — no
        catastrophic-backtracking exposure.

        Args:
            subj_pattern: Regex pattern for the subject endpoint.
            obj_pattern: Regex pattern for the object endpoint.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint applied to every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            fields: Optional whitelist of fields, applied to both endpoints.
            case_sensitive: When ``False``, matches case-insensitively.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                f"Invalid hop range: min_hops={min_hops}, "
                f"max_hops={max_hops}. Require 1 <= min_hops <= max_hops."
            )
        if not subj_pattern or not obj_pattern:
            return format_search_results([], output_format)

        subj_label = sanitize_label(subj_label)
        obj_label = sanitize_label(obj_label)
        rel_label = sanitize_label(label) if label is not None else None
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        subj_hits = await self.entity_regex_search(
            subj_pattern,
            label=subj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
        )
        obj_hits = await self.entity_regex_search(
            obj_pattern,
            label=obj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
        )

        subj_pks = {hit[subj_pk] for hit in subj_hits}
        obj_pks = {hit[obj_pk] for hit in obj_hits}

        if not subj_pks or not obj_pks:
            return format_search_results([], output_format)

        if rel_label is not None:
            path_pattern = f"-[:{rel_label}*{min_hops}..{max_hops}]->"
        else:
            path_pattern = f"-[*{min_hops}..{max_hops}]->"

        rows = self._con.execute(
            f"MATCH p = (s:{subj_label}){path_pattern}(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks AND o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, o AS obj, "
            f"nodes(p) AS nodes, rels(p) AS rels, length(p) AS length, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_pks),
                "obj_pks": list(obj_pks),
            },
        ).get_all()

        # Regex is binary — same path-length distinct dedup as the
        # vector path search, ranked by length (shorter first).
        seen: set = set()
        results: List[Dict[str, Any]] = []
        for row in rows:
            s_node, o_node, ns, rs, length, spk_val, opk_val = row
            key = (spk_val, opk_val, length)
            if key in seen:
                continue
            seen.add(key)
            row_out: Dict[str, Any] = {
                "subj": s_node,
                "obj": o_node,
                "nodes": ns,
                "rels": rs,
                "length": length,
            }
            row_out.update(self._flatten_endpoint_pks(subj_pk, obj_pk, spk_val, opk_val))
            results.append(row_out)

        results.sort(key=lambda r: r["length"])
        return format_search_results(results[:k], output_format)

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

        Each side is hybrid-searched (vec + regex) independently;
        the path's combined ``rrf_score`` is the sum of the two
        endpoint hybrid scores — the 4-source RRF identity. Falls
        through to `path_similarity_search` when no patterns
        are supplied.

        Args:
            subj_text_or_texts: Query text (or list) for the subject
                vector branch.
            obj_text_or_texts: Query text (or list) for the object
                vector branch.
            subj_pattern_or_patterns: Regex pattern (or list) for the
                subject regex branch.
            obj_pattern_or_patterns: Regex pattern (or list) for the
                object regex branch.
            subj_label: Entity label of the subject endpoint.
            obj_label: Entity label of the object endpoint.
            label: Optional rel-label constraint applied to every hop.
            min_hops: Minimum hop count, inclusive (default: 1).
            max_hops: Maximum hop count, inclusive (default: 3).
            k: Maximum number of results.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional vector-distance threshold.
            fields: Forwarded to the regex branch.
            case_sensitive: Forwarded to the regex branch.
            output_format: ``"json"`` (default) or ``"csv"``.
        """
        if not subj_pattern_or_patterns and not obj_pattern_or_patterns:
            return await self.path_similarity_search(
                subj_text_or_texts,
                obj_text_or_texts,
                subj_label=subj_label,
                obj_label=obj_label,
                label=label,
                min_hops=min_hops,
                max_hops=max_hops,
                k=k,
                subj_threshold=similarity_threshold,
                obj_threshold=similarity_threshold,
                output_format=output_format,
            )

        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                f"Invalid hop range: min_hops={min_hops}, "
                f"max_hops={max_hops}. Require 1 <= min_hops <= max_hops."
            )
        subj_label = sanitize_label(subj_label)
        obj_label = sanitize_label(obj_label)
        rel_label = sanitize_label(label) if label is not None else None
        subj_pk = self._pk_keys.get(subj_label, "id")
        obj_pk = self._pk_keys.get(obj_label, "id")

        if not subj_text_or_texts or not obj_text_or_texts:
            return format_search_results([], output_format)

        subj_hits = await self.entity_hybrid_regex_search(
            text_or_texts=subj_text_or_texts,
            pattern_or_patterns=subj_pattern_or_patterns,
            label=subj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
        )
        obj_hits = await self.entity_hybrid_regex_search(
            text_or_texts=obj_text_or_texts,
            pattern_or_patterns=obj_pattern_or_patterns,
            label=obj_label,
            fields=fields,
            case_sensitive=case_sensitive,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
        )

        subj_score: Dict[Any, float] = {
            hit[subj_pk]: self._hybrid_hit_score(hit) for hit in subj_hits
        }
        obj_score: Dict[Any, float] = {
            hit[obj_pk]: self._hybrid_hit_score(hit) for hit in obj_hits
        }
        if not subj_score or not obj_score:
            return format_search_results([], output_format)

        if rel_label is not None:
            path_pattern = f"-[:{rel_label}*{min_hops}..{max_hops}]->"
        else:
            path_pattern = f"-[*{min_hops}..{max_hops}]->"

        rows = self._con.execute(
            f"MATCH p = (s:{subj_label}){path_pattern}(o:{obj_label}) "
            f"WHERE s.{subj_pk} IN $subj_pks AND o.{obj_pk} IN $obj_pks "
            f"RETURN s AS subj, o AS obj, "
            f"nodes(p) AS nodes, rels(p) AS rels, length(p) AS length, "
            f"s.{subj_pk} AS __subj_pk, o.{obj_pk} AS __obj_pk",
            {
                "subj_pks": list(subj_score.keys()),
                "obj_pks": list(obj_score.keys()),
            },
        ).get_all()

        best: Dict[tuple, Dict[str, Any]] = {}
        for row in rows:
            s_node, o_node, ns, rs, length, spk_val, opk_val = row
            sub_c = subj_score[spk_val]
            obj_c = obj_score[opk_val]
            rrf = sub_c + obj_c
            key = (spk_val, opk_val, length)
            if key not in best or rrf > best[key]["rrf_score"]:
                best[key] = {
                    "subj": s_node,
                    "obj": o_node,
                    "nodes": ns,
                    "rels": rs,
                    "length": length,
                    "subj_rrf_score": sub_c,
                    "obj_rrf_score": obj_c,
                    "rrf_score": rrf,
                    "__subj_pk": spk_val,
                    "__obj_pk": opk_val,
                }

        ranked = sorted(
            best.values(),
            key=lambda r: (-r["rrf_score"], r["length"]),
        )[:k]
        results: List[Dict[str, Any]] = []
        for entry in ranked:
            row_out: Dict[str, Any] = {
                "subj": entry["subj"],
                "obj": entry["obj"],
                "nodes": entry["nodes"],
                "rels": entry["rels"],
                "length": entry["length"],
                "rrf_score": entry["rrf_score"],
                "subj_rrf_score": entry["subj_rrf_score"],
                "obj_rrf_score": entry["obj_rrf_score"],
            }
            row_out.update(
                self._flatten_endpoint_pks(
                    subj_pk, obj_pk, entry["__subj_pk"], entry["__obj_pk"]
                )
            )
            results.append(row_out)
        return format_search_results(results, output_format)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying Ladybug connection. Idempotent."""
        try:
            self._con.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"<LadybugAdapter uri={self.uri}>"
