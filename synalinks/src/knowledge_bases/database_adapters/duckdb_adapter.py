import heapq
import os
import warnings
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import duckdb
import duckdb.sqltypes
import orjson

from synalinks.src.backend import EmbeddingRequest
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.knowledge_bases.adapters_utils import format_search_results
from synalinks.src.knowledge_bases.adapters_utils import minmax_normalize_scores
from synalinks.src.knowledge_bases.adapters_utils import resolve_db_path
from synalinks.src.knowledge_bases.adapters_utils import sanitize_identifier
from synalinks.src.knowledge_bases.adapters_utils import to_pascal_identifier
from synalinks.src.knowledge_bases.adapters_utils import to_snake_identifier
from synalinks.src.knowledge_bases.database_adapters.database_adapter import (
    DatabaseAdapter,
)
from synalinks.src.modules.embedding_models import get as _get_em
from synalinks.src.utils.naming import to_pascal_case

VSS_KEY = "embedding"

_DATE_LIKE_FORMATS = frozenset({"date", "date-time", "time"})

STEMMERS = [
    "arabic",
    "basque",
    "catalan",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hindi",
    "hungarian",
    "indonesian",
    "irish",
    "italian",
    "lithuanian",
    "nepali",
    "norwegian",
    "porter",
    "portuguese",
    "romanian",
    "russian",
    "serbian",
    "spanish",
    "swedish",
    "tamil",
    "turkish",
    "none",
]

METRICS = [
    "l2sq",
    "cosine",
    "ip",
]

# DuckDB VSS distance function for each canonical metric. All three return a
# value where *smaller means closer*, so ``ORDER BY score ASC`` ranks correctly
# for every metric. The HNSW index is only used when the query's distance
# function matches the one the index was built with (see ``_hnsw_with_clause``).
_VSS_DISTANCE_FN = {
    "l2sq": "array_distance",
    "cosine": "array_cosine_distance",
    "ip": "array_negative_inner_product",
}

MAIN_TABLE = "main"


def _get_json_columns_from_schema(schema: dict) -> set:
    """Get column names that are JSON type from a JSON schema."""
    json_columns = set()
    properties = schema.get("properties", {})
    for prop_name, prop_spec in properties.items():
        prop_type = prop_spec.get("type")
        if prop_type == "object":
            json_columns.add(prop_name)
        elif prop_type == "array":
            item_spec = prop_spec.get("items", {})
            if item_spec.get("type") == "object":
                json_columns.add(prop_name)
    return json_columns


def _parse_json_columns(row: dict, json_columns: set) -> dict:
    """Parse JSON string columns to Python dicts based on known JSON columns."""
    result = dict(row)
    for col in json_columns:
        if col in result and isinstance(result[col], str):
            try:
                result[col] = orjson.loads(result[col])
            except (orjson.JSONDecodeError, TypeError):
                pass
    return result


# SQL-flavored aliases over the shared identifier helpers in
# ``adapters_utils``. Kept under their existing names so call sites
# read naturally for a SQL backend.
table_identifier = to_pascal_identifier
column_identifier = to_snake_identifier


def sanitize_properties(properties: dict):
    """Sanitize and snake_case-normalize dict keys before SQL interpolation."""
    return {column_identifier(k): v for k, v in properties.items()}


_ATTACH_ALIAS = "kb"


class DuckDBAdapter(DatabaseAdapter):
    def __init__(
        self,
        uri=None,
        embedding_model=None,
        data_models=None,
        stemmer="porter",
        stopwords=None,
        ignore=None,
        strip_accents=None,
        lower=None,
        metric="cosine",
        ef_construction=None,
        ef_search=None,
        M=None,
        M0=None,
        vss_key=VSS_KEY,
        main_table=MAIN_TABLE,
        vector_dim=None,
        wipe_on_start=False,
        name=None,
        encryption_key=None,
        **kwargs,
    ):
        # Resolve URI → backing path. ``None`` defaults to
        # ``{synalinks_home()}/{name or 'database'}.db`` so a no-args
        # KnowledgeBase() persists across processes. The helper also
        # enforces the path-traversal gate on ``name``.
        self.uri = resolve_db_path(uri, scheme="duckdb", extension="db", name=name)

        # Bound as a `?` param to ATTACH; deliberately excluded from get_config().
        self._encryption_key: Optional[str] = encryption_key

        self.embedding_model = _get_em(embedding_model)

        # Vector dimension resolved lazily on the main loop the first time a
        # table is created (see `_ensure_vector_dim`), never via a probe here.
        # The old probe used `run_maybe_nested`, running the embedding on a
        # transient thread-loop and binding litellm's process-global httpx
        # client to a loop closed moments later — poisoning it for every later
        # main-loop embedding ("Event loop is closed"). `vector_dim=` short-
        # circuits.
        self.vector_dim = vector_dim
        # The `FLOAT[dim]` embedding column needs the dimension up front, so when
        # an embedding model is set without an explicit `vector_dim`, defer
        # creating the declared tables until the first `update` (where the dim is
        # learned on the main loop). Only this case is deferred.
        self._defer_table_creation = bool(self.embedding_model and not vector_dim)

        if stemmer not in STEMMERS:
            raise ValueError(f"`stemmer` parameter should be one of {STEMMERS}")
        self.stemmer = stemmer

        if metric not in METRICS:
            raise ValueError(f"`metric` parameter should be one of {METRICS}")
        self.metric = metric

        # FTS-index build params (forwarded to ``PRAGMA create_fts_index``).
        # ``None`` defers to the DuckDB default for that arg so we don't
        # have to track upstream defaults (``stopwords='english'``,
        # ``strip_accents=1``, ``lower=1``, etc.) — only user-supplied
        # values get serialized into the pragma.
        self.stopwords = stopwords
        self.ignore = ignore
        self.strip_accents = strip_accents
        self.lower = lower

        # HNSW build params (forwarded to ``CREATE INDEX ... WITH (...)``).
        # Same None-defers-to-engine convention as the FTS knobs and as the
        # Ladybug adapter's ``mu`` / ``ml`` / ``pu`` / ``efc``.
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.M = M
        self.M0 = M0

        self.vss_key = vss_key

        # Single persistent RW connection — DuckDB holds an exclusive file
        # lock for its lifetime, so this process is the only writer.
        self._con: Optional[duckdb.DuckDBPyConnection] = None

        self._install_extensions()

        # Encrypted DBs must be created through ATTACH-with-key, not bare connect.
        if not os.path.exists(self.uri):
            tmp = duckdb.connect(":memory:")
            try:
                self._attach_db(tmp)
                tmp.execute(f"DETACH {_ATTACH_ALIAS}")
            finally:
                tmp.close()

        self._con = self._open_main_connection()
        self._sandbox(self._con)

        if wipe_on_start:
            self.wipe_database()

        self.data_models: Dict[str, Any] = {}
        if data_models:
            for dm in data_models:
                title = dm.get_schema().get("title")
                if title is None:
                    raise ValueError(
                        "Each registered data model must carry a "
                        "schema `title`; got a schema with no title."
                    )
                self.data_models[table_identifier(title)] = dm
                if not self._defer_table_creation:
                    self._maybe_create_table(dm)
        else:
            for dm in self.get_symbolic_data_models():
                self.data_models[dm.get_schema().get("title")] = dm

    def _attach_db(self, con, read_only=False):
        """ATTACH the configured database file onto ``con``.

        The path is interpolated as a SQL string literal (with single-
        quote escaping as defence in depth — ``uri`` is documented as
        caller-trusted). The encryption key, if any, is bound as a
        ``?`` parameter so arbitrary key bytes can't break the SQL or
        end up in logs.
        """
        safe_uri = self.uri.replace("'", "''")
        options = []
        params: List[Any] = []
        if self._encryption_key is not None:
            options.append("ENCRYPTION_KEY ?")
            params.append(self._encryption_key)
        if read_only:
            options.append("READ_ONLY")
        opts_clause = f" ({', '.join(options)})" if options else ""
        con.execute(
            f"ATTACH '{safe_uri}' AS {_ATTACH_ALIAS}{opts_clause}",
            params,
        )

    def _open_main_connection(self):
        """Open the persistent connection used by every operation."""
        if self._encryption_key is not None:
            con = duckdb.connect(":memory:")
            self._attach_db(con)
            con.execute(f"USE {_ATTACH_ALIAS}")
            return con
        return duckdb.connect(self.uri, read_only=False)

    def _open_loose_connection(self):
        """Open a fresh, NON-sandboxed connection for native bulk loads.

        ``read_csv`` / ``read_parquet`` need ``enable_external_access=true``,
        which the sandboxed persistent connection has permanently disabled.
        """
        if self._encryption_key is not None:
            con = duckdb.connect(":memory:")
            self._attach_db(con)
            con.execute(f"USE {_ATTACH_ALIAS}")
        else:
            con = duckdb.connect(self.uri, read_only=False)
        try:
            con.execute("LOAD fts;")
            if self.embedding_model:
                con.execute("LOAD vss;")
                con.execute("SET hnsw_enable_experimental_persistence=true;")
        except duckdb.Error:
            pass
        return con

    @contextmanager
    def _connect(self, read_only=False):
        """Yield the adapter's cached connection.

        ``read_only`` only changes the bootstrap throwaway path; the cached
        connection is always RW (see ``__init__``). ``query(read_only=True)``
        enforces SELECT-only at the parser layer for untrusted SQL.
        """
        if self._con is None:
            if self._encryption_key is not None:
                con = duckdb.connect(":memory:")
                self._attach_db(con, read_only=read_only)
                con.execute(f"USE {_ATTACH_ALIAS}")
            else:
                con = duckdb.connect(self.uri, read_only=read_only)
            try:
                yield con
            finally:
                con.close()
        else:
            yield self._con

    def _sandbox(self, con):
        """Lock the persistent connection down against external I/O.

        Without ``enable_external_access=false``, a query like
        ``SELECT * FROM read_csv('/etc/passwd', ...)`` passes parser-level
        SELECT-only checks and exfiltrates host files. Extensions must be
        LOAD-ed before the switch flips — LOAD itself needs external access.
        """
        try:
            con.execute("LOAD fts;")
            if self.embedding_model:
                con.execute("LOAD vss;")
                con.execute("SET hnsw_enable_experimental_persistence=true;")
        except duckdb.Error as e:
            warnings.warn(
                f"DuckDB extension LOAD failed; FTS / VSS queries will fail "
                f"until the extensions install successfully. ({e})"
            )
        con.execute("SET enable_external_access=false;")

    def close(self):
        """Close the adapter's cached connection. Idempotent."""
        if self._con is not None:
            try:
                self._con.close()
            except duckdb.Error:
                pass
            self._con = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _install_extensions(self):
        """Install required extensions only if missing, via throwaway connections."""
        needed = ["fts"]
        if self.embedding_model:
            needed.append("vss")

        try:
            probe = duckdb.connect(":memory:")
            try:
                installed = {
                    row[0]
                    for row in probe.execute(
                        "SELECT extension_name FROM duckdb_extensions() WHERE installed"
                    ).fetchall()
                }
            finally:
                probe.close()
            missing = [ext for ext in needed if ext not in installed]
            if not missing:
                return
        except duckdb.PermissionException:
            # Sibling adapter already disabled external access process-wide.
            return
        except duckdb.Error:
            missing = needed

        installer = duckdb.connect(":memory:")
        try:
            for ext in missing:
                installer.execute(f"INSTALL {ext};")
        except duckdb.PermissionException:
            pass
        except duckdb.Error as e:
            warnings.warn(
                f"Failed to install DuckDB extensions {missing}; "
                f"FTS / VSS queries will not work until the extensions "
                f"are available. ({e})"
            )
        finally:
            installer.close()

    def wipe_database(self):
        with self._connect(read_only=False) as con:
            tables = con.execute(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='{MAIN_TABLE}';
            """).fetchall()

            for (table_name,) in tables:
                try:
                    con.execute(f"DROP TABLE IF EXISTS {table_name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to drop table {table_name}: {e}")

    def _get_id_key(self, schema: dict) -> str:
        """Return the first schema property (snake-cased, sanitized) as primary key."""
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not props:
            raise ValueError("Cannot determine primary key: schema has no `properties`.")
        return column_identifier(next(iter(props.keys())))

    def _has_indexable_text_columns(self, schema: dict) -> bool:
        """Return ``True`` if the schema has at least one VARCHAR column
        besides the primary key. Used to gate FTS index creation and
        BM25 queries: DuckDB's ``create_fts_index('*', ...)`` requires
        at least one indexable text column, and the id column is
        excluded by the pragma itself.
        """
        id_key = self._get_id_key(schema)
        for name, info in schema.get("properties", {}).items():
            if column_identifier(name) == id_key:
                continue
            if (
                info.get("type") == "string"
                and info.get("format") not in _DATE_LIKE_FORMATS
            ):
                return True
        return False

    def _duckdb_table_to_json_schema(
        self,
        table_name: str,
        remove_embedding: bool = True,
    ) -> dict:
        with self._connect(read_only=True) as con:
            info = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            props = {}
            for _, name, dtype, _, _, _ in info:
                if name == self.vss_key and remove_embedding:
                    continue
                elif dtype == duckdb.sqltypes.DuckDBPyType(str):
                    props[name] = {"title": name.title(), "type": "string"}
                elif dtype == duckdb.sqltypes.DuckDBPyType(float):
                    props[name] = {"title": name.title(), "type": "number"}
                elif dtype == duckdb.sqltypes.DuckDBPyType(int):
                    props[name] = {"title": name.title(), "type": "integer"}
                elif dtype == duckdb.sqltypes.DuckDBPyType(bool):
                    props[name] = {"title": name.title(), "type": "boolean"}
                elif dtype == duckdb.sqltypes.DuckDBPyType(list[Union[str]]):
                    props[name] = {
                        "title": name.title(),
                        "items": {"type": "string"},
                        "type": "array",
                    }
                elif dtype == duckdb.sqltypes.DuckDBPyType(list[Union[float]]):
                    props[name] = {
                        "title": name.title(),
                        "items": {"type": "number"},
                        "type": "array",
                    }
                elif dtype == duckdb.sqltypes.DuckDBPyType(list[Union[int]]):
                    props[name] = {
                        "title": name.title(),
                        "items": {"type": "integer"},
                        "type": "array",
                    }
                elif dtype == duckdb.sqltypes.DuckDBPyType(list[Union[bool]]):
                    props[name] = {
                        "title": name.title(),
                        "items": {"type": "boolean"},
                        "type": "array",
                    }
                elif dtype == duckdb.sqltypes.DATE:
                    props[name] = {
                        "title": name.title(),
                        "type": "string",
                        "format": "date",
                    }
                elif dtype in (duckdb.sqltypes.TIMESTAMP, duckdb.sqltypes.TIMESTAMP_TZ):
                    props[name] = {
                        "title": name.title(),
                        "type": "string",
                        "format": "date-time",
                    }
                elif dtype == duckdb.sqltypes.TIME:
                    props[name] = {
                        "title": name.title(),
                        "type": "string",
                        "format": "time",
                    }
                elif str(dtype) == "JSON":
                    props[name] = {
                        "title": name.title(),
                        "type": "object",
                    }
                else:
                    raise NotImplementedError(
                        f"Type '{dtype}' not supported by {self.__class__.__name__}"
                        " at the moment, please fill out an issue."
                    )

            return {
                "title": table_name,
                "type": "object",
                "additionalProperties": False,
                "required": list(props.keys()),
                "properties": props,
            }

    def _json_schema_to_duckdb_columns(self, json_schema: dict):
        """Convert JSON schema to DuckDB column definitions. First property is PK."""
        properties = json_schema.get("properties", {})
        defs = json_schema.get("$defs", {})
        out = []
        first_col = True

        for prop_name, prop_spec in properties.items():
            prop_name = column_identifier(prop_name)

            # Resolve Pydantic v2 $ref for enums / nested models.
            if "$ref" in prop_spec:
                ref_name = prop_spec["$ref"].rsplit("/", 1)[-1]
                if ref_name in defs:
                    resolved = dict(defs[ref_name])
                    resolved.update({k: v for k, v in prop_spec.items() if k != "$ref"})
                    prop_spec = resolved

            prop_type = prop_spec.get("type")

            if prop_name == self.vss_key:
                continue

            if not prop_type and "anyOf" in prop_spec:
                for variant in prop_spec["anyOf"]:
                    if "$ref" in variant:
                        ref_name = variant["$ref"].rsplit("/", 1)[-1]
                        if ref_name in defs:
                            variant = defs[ref_name]
                    vtype = variant.get("type")
                    if vtype and vtype != "null":
                        prop_type = vtype
                        prop_spec = variant
                        break
                    if "enum" in variant:
                        prop_type = "string"
                        break

            if not prop_type and "enum" in prop_spec:
                prop_type = "string"

            if not prop_type:
                raise ValueError(f"Malformed JSON schema: missing type for '{prop_name}'")

            col_def = None

            if prop_type == "array":
                item_spec = prop_spec.get("items")
                if not item_spec:
                    col_def = f"{prop_name} JSON"
                else:
                    item_type = item_spec.get("type")
                    if item_type == "string":
                        dtype = duckdb.sqltypes.DuckDBPyType(list[Union[str]])
                        col_def = f"{prop_name} {dtype}"
                    elif item_type == "number":
                        dtype = duckdb.sqltypes.DuckDBPyType(list[Union[float]])
                        col_def = f"{prop_name} {dtype}"
                    elif item_type == "integer":
                        dtype = duckdb.sqltypes.DuckDBPyType(list[Union[int]])
                        col_def = f"{prop_name} {dtype}"
                    elif item_type == "boolean":
                        dtype = duckdb.sqltypes.DuckDBPyType(list[Union[bool]])
                        col_def = f"{prop_name} {dtype}"
                    elif item_type == "object":
                        col_def = f"{prop_name} JSON"
                    else:
                        raise ValueError(
                            f"Unsupported array item type '{item_type}' for '{prop_name}'"
                        )
            elif prop_type == "object":
                col_def = f"{prop_name} JSON"
            elif prop_type == "string":
                fmt = prop_spec.get("format")
                if fmt == "date":
                    col_def = f"{prop_name} DATE"
                elif fmt == "date-time":
                    col_def = f"{prop_name} TIMESTAMP"
                elif fmt == "time":
                    col_def = f"{prop_name} TIME"
                else:
                    col_def = f"{prop_name} VARCHAR"
            elif prop_type == "number":
                dtype = duckdb.sqltypes.DuckDBPyType(float)
                col_def = f"{prop_name} {dtype}"
            elif prop_type == "integer":
                dtype = duckdb.sqltypes.DuckDBPyType(int)
                col_def = f"{prop_name} {dtype}"
            elif prop_type == "boolean":
                dtype = duckdb.sqltypes.DuckDBPyType(bool)
                col_def = f"{prop_name} {dtype}"
            else:
                raise ValueError(f"Unsupported JSON schema type: '{prop_type}'")

            if first_col and col_def:
                col_def += " PRIMARY KEY"
                first_col = False

            if col_def:
                out.append(col_def)

        if self.embedding_model:
            out.append(f"{self.vss_key} FLOAT[{self.vector_dim}]")
        return ", ".join(out)

    def get_symbolic_data_models(
        self,
        remove_embedding=True,
    ) -> List[SymbolicDataModel]:
        with self._connect(read_only=True) as con:
            tables = con.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='main';
            """).fetchall()

            symbolic_data_models = []
            for (table_name,) in tables:
                schema = self._duckdb_table_to_json_schema(table_name)
                model = SymbolicDataModel(schema=schema, name=table_name)
                symbolic_data_models.append(model)
            return symbolic_data_models

    async def _ensure_vector_dim(self, sample_vector=None):
        """Resolve the embedding dimension lazily, on the current event loop.

        Prefers the length of an embedding vector already in hand (records
        arrive pre-embedded from ``EmbedKnowledge``), falling back to a probe
        awaited on *this* loop only when the vector column must be created
        before any embedded record is available. Never uses ``run_maybe_nested``
        (a transient-loop network call poisons litellm's global client).
        """
        if self.vector_dim is not None or not self.embedding_model:
            return
        if sample_vector:
            self.vector_dim = len(sample_vector)
            return
        probe = await self.embedding_model(EmbeddingRequest(texts=["text"]))
        embeddings = probe.get("embeddings") if probe is not None else None
        if not embeddings:
            raise ValueError(
                f"Embedding model {self.embedding_model} returned no embeddings "
                "while resolving the vector dimension. This usually means the "
                "model name is wrong or unavailable for your provider/API key. "
                "Fix the embedding model, or pass an explicit `vector_dim=...`."
            )
        self.vector_dim = len(embeddings[0])

    def _maybe_create_table(
        self,
        data_model: Union[JsonDataModel, SymbolicDataModel],
    ):
        with self._connect(read_only=False) as con:
            json_schema = data_model.get_schema()
            table_name = table_identifier(json_schema.get("title"))

            exists = con.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema='{MAIN_TABLE}' AND table_name='{table_name}';
            """).fetchone()[0]

            if exists:
                return

            columns = self._json_schema_to_duckdb_columns(json_schema)
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"

            try:
                con.execute(create_sql)
            except Exception as e:
                raise RuntimeError(f"Failed to create table '{table_name}': {e}")

    @staticmethod
    def _render_sql_literal(value: Any) -> str:
        """Render a Python value as a SQL literal for DDL/PRAGMA options.

        Used to interpolate adapter init kwargs (stemmer, stopwords,
        HNSW ``WITH`` options, etc.) into statements where DuckDB
        rejects ``?``-parameters at the parser layer. Bools become
        ``0`` / ``1`` (DuckDB's accepted BOOLEAN literal forms in
        PRAGMA / WITH clauses); ints and floats are emitted bare;
        strings are single-quoted with embedded quotes doubled.
        """
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    def _fts_index_options(self, overwrite: bool) -> str:
        """Render the optional kwargs for ``PRAGMA create_fts_index``.

        Only adapter init values that were explicitly supplied (i.e.
        not ``None``) get serialized — DuckDB picks its own defaults
        otherwise, so this avoids hard-coding upstream defaults
        (``stopwords='english'``, ``strip_accents=1`` etc.) into the
        adapter. ``overwrite`` is always emitted because callers pass
        an explicit ``True`` / ``False`` per rebuild.
        """
        opts: List[str] = [f"stemmer='{self.stemmer}'"]
        for key, value in (
            ("stopwords", self.stopwords),
            ("ignore", self.ignore),
            ("strip_accents", self.strip_accents),
            ("lower", self.lower),
        ):
            if value is None:
                continue
            opts.append(f"{key}={self._render_sql_literal(value)}")
        opts.append(f"overwrite={1 if overwrite else 0}")
        return ", ".join(opts)

    def _maybe_create_fulltext_index(
        self,
        data_model: Union[JsonDataModel, SymbolicDataModel],
        overwrite: bool = True,
    ):
        with self._connect(read_only=False) as con:
            json_schema = data_model.get_schema()
            table_name = table_identifier(json_schema.get("title"))
            id_key = self._get_id_key(json_schema)

            if not self._has_indexable_text_columns(json_schema):
                return
            # '*' tells DuckDB to index every VARCHAR column (the id
            # column is excluded by the pragma itself), so callers
            # don't need a whitelist of "searchable" field names.
            con.execute(f"""
                PRAGMA create_fts_index(
                    'main.{table_name}',
                    '{id_key}',
                    '*',
                    {self._fts_index_options(overwrite)}
                );
            """)

    def _hnsw_with_clause(self) -> str:
        """Render the ``WITH (...)`` clause for ``CREATE INDEX ... USING HNSW``.

        ``metric`` is always emitted (the adapter validates it at
        init); the other knobs (``ef_construction``, ``ef_search``,
        ``M``, ``M0``) follow the None-defers-to-engine convention so
        users only pin the values they actually want to tune.
        """
        opts: List[str] = [f"metric = '{self.metric}'"]
        for key, value in (
            ("ef_construction", self.ef_construction),
            ("ef_search", self.ef_search),
            ("M", self.M),
            ("M0", self.M0),
        ):
            if value is None:
                continue
            opts.append(f"{key} = {self._render_sql_literal(value)}")
        return ", ".join(opts)

    def _maybe_create_vector_index(
        self,
        data_model: Union[JsonDataModel, SymbolicDataModel],
        overwrite: bool = True,
    ):
        """Build (or rebuild) the HNSW vector index. No-op without embeddings."""
        if not self.embedding_model:
            return

        with self._connect(read_only=False) as con:
            json_schema = data_model.get_schema()
            table_name = table_identifier(json_schema.get("title"))

            has_embeddings = con.execute(
                f"SELECT EXISTS ("
                f"SELECT 1 FROM {table_name} "
                f"WHERE {self.vss_key} IS NOT NULL"
                f")"
            ).fetchone()[0]
            if not has_embeddings:
                return

            index_name = f"vector_main_{table_name}"
            with_clause = self._hnsw_with_clause()
            if overwrite:
                # HNSW indexes don't support CREATE OR REPLACE.
                con.execute(f"DROP INDEX IF EXISTS {index_name};")
                con.execute(
                    f"CREATE INDEX {index_name} ON {table_name}"
                    f" USING HNSW ({self.vss_key})"
                    f" WITH ({with_clause});"
                )
            else:
                con.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}"
                    f" USING HNSW ({self.vss_key})"
                    f" WITH ({with_clause});"
                )

    async def update(
        self,
        data_model_or_data_models: Union[List[JsonDataModel], JsonDataModel],
    ) -> Union[Any, List[Any]]:
        """Update or insert records. Returns the primary key value(s)."""
        if not isinstance(data_model_or_data_models, list):
            data_models = [data_model_or_data_models]
            return_single = True
        else:
            data_models = data_model_or_data_models
            return_single = False

        # Bucket by (table, column-shape) so each bucket uses one executemany.
        ids: List[Any] = []
        buckets: Dict[tuple, Dict[str, Any]] = {}
        tables_seen: List[Any] = []
        tables_seen_set: set = set()

        for data_model in data_models:
            if not isinstance(data_model, JsonDataModel):
                data_model = data_model.to_json_data_model()

            schema = data_model.get_schema()
            table = table_identifier(schema["title"])
            json_data = sanitize_properties(data_model.get_json())
            id_key = self._get_id_key(schema)

            id_val = json_data.get(id_key)
            if id_val is None:
                raise ValueError(f"Primary key '{id_key}' is required but not provided")

            if table not in tables_seen_set:
                # Learn the dimension from this record's embedding (or an
                # on-loop probe) before the FLOAT[dim] column is created.
                await self._ensure_vector_dim(json_data.get(self.vss_key))
                self._maybe_create_table(data_model)
                tables_seen.append(data_model)
                tables_seen_set.add(table)

            cols = tuple(json_data.keys())
            bucket = buckets.setdefault(
                (table, cols),
                {"id_key": id_key, "cols": cols, "params": []},
            )
            bucket["params"].append([json_data[c] for c in cols])
            ids.append(id_val)

        with self._connect(read_only=False) as con:
            con.execute("BEGIN TRANSACTION;")
            try:
                for (table, _cols), bucket in buckets.items():
                    cols = bucket["cols"]
                    id_key = bucket["id_key"]
                    col_sql = ", ".join(cols)
                    placeholders = ", ".join(["?"] * len(cols))
                    update_cols = [c for c in cols if c != id_key]

                    if update_cols:
                        update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
                        conflict_clause = (
                            f"ON CONFLICT ({id_key}) DO UPDATE SET {update_sql}"
                        )
                    else:
                        conflict_clause = f"ON CONFLICT ({id_key}) DO NOTHING"

                    sql = (
                        f"INSERT INTO {table} ({col_sql}) "
                        f"VALUES ({placeholders}) "
                        f"{conflict_clause};"
                    )

                    con.executemany(sql, bucket["params"])
            except Exception:
                con.execute("ROLLBACK;")
                raise
            con.execute("COMMIT;")

        # FTS/HNSW rebuilds are best-effort — data is already committed.
        for data_model in tables_seen:
            try:
                self._maybe_create_fulltext_index(data_model)
            except Exception as e:
                table = table_identifier(data_model.get_schema()["title"])
                warnings.warn(
                    f"FTS index rebuild failed for '{table}'; "
                    f"fulltext_search results may be stale. ({e})"
                )

        for data_model in tables_seen:
            try:
                self._maybe_create_vector_index(data_model)
            except Exception as e:
                table = table_identifier(data_model.get_schema()["title"])
                warnings.warn(
                    f"Vector index rebuild failed for '{table}'; "
                    f"similarity_search will fall back to scan. ({e})"
                )

        return ids[0] if return_single else ids

    async def from_csv(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        header: bool = True,
    ) -> SymbolicDataModel:
        """Bulk-load a CSV file directly into a new (or existing) table.

        Uses DuckDB's native ``read_csv`` so column types are
        auto-detected from the file (with the conservative bias that
        zero-padded ids like ``"00123"`` stay ``VARCHAR``). The first
        column is promoted to ``PRIMARY KEY``. Column names are
        normalized to ``snake_case``; the table name to ``PascalCase``.

        Use this over ``update(CSVDataset(...))`` for non-trivial files
        — the bulk path is orders of magnitude faster because it
        bypasses the per-row Pydantic / Python pipeline. Prefer
        ``update`` when source rows need transformation before storage.

        Args:
            path: Path to the CSV file.
            table_name: Target table name. Defaults to the file's stem
                (``/data/my-docs.csv`` → ``MyDocs``). Always normalized
                to PascalCase.
            table_description: Optional natural-language description
                attached to the resulting schema's top-level
                ``description`` field.
            delimiter: Field delimiter. Defaults to ``","``.
            encoding: File encoding. Defaults to ``"utf-8"``.
            header: Whether the first row is a header. Defaults to
                ``True``.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table. Pass
            it to ``fulltext_search`` / ``similarity_search`` / ``get``
            to query the data.
        """
        reader_kwargs = {
            "delim": f"'{delimiter.replace(chr(39), chr(39) * 2)}'",
            "header": "true" if header else "false",
            "encoding": f"'{encoding.replace(chr(39), chr(39) * 2)}'",
        }
        return await self._bulk_load(
            path,
            table_name=table_name,
            table_description=table_description,
            reader_fn="read_csv",
            reader_kwargs=reader_kwargs,
        )

    async def from_parquet(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> SymbolicDataModel:
        """Bulk-load a Parquet file directly into a new (or existing) table.

        Same fast-path trade-offs as :meth:`from_csv` — bypasses the
        Python row pipeline for native DuckDB ingestion. Parquet's
        schema is explicit in the file's footer, so types are
        preserved end-to-end without auto-detection guesswork.

        Args:
            path: Path to the Parquet file.
            table_name: Target table name. Defaults to the file's stem,
                PascalCase-normalized.
            table_description: Optional schema description.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table.
        """
        return await self._bulk_load(
            path,
            table_name=table_name,
            table_description=table_description,
            reader_fn="read_parquet",
            reader_kwargs={},
        )

    async def from_json(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> SymbolicDataModel:
        """Bulk-load a JSON file (top-level array of objects).

        Same fast-path trade-offs as :meth:`from_csv` /
        :meth:`from_parquet`. Use :meth:`from_jsonl` for one-object-
        per-line NDJSON sources.

        Args:
            path: Path to the JSON file. Must contain a top-level array
                of objects, e.g. ``[{"id": "a", "text": "..."}, ...]``.
            table_name: Target table name. Defaults to the file's stem,
                PascalCase-normalized.
            table_description: Optional schema description.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table.
        """
        return await self._bulk_load(
            path,
            table_name=table_name,
            table_description=table_description,
            reader_fn="read_json",
            reader_kwargs={"format": "'array'"},
        )

    async def from_jsonl(
        self,
        path: str,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> SymbolicDataModel:
        """Bulk-load a JSON Lines (NDJSON) file.

        Same fast-path trade-offs as :meth:`from_csv` /
        :meth:`from_parquet`. Use this for very large JSON sources that
        aren't a single array.

        Args:
            path: Path to the JSONL file.
            table_name: Target table name. Defaults to the file's stem,
                PascalCase-normalized.
            table_description: Optional schema description.

        Returns:
            The :class:`SymbolicDataModel` for the loaded table.
        """
        return await self._bulk_load(
            path,
            table_name=table_name,
            table_description=table_description,
            reader_fn="read_json",
            reader_kwargs={"format": "'newline_delimited'"},
        )

    async def _bulk_load(
        self,
        path: str,
        *,
        table_name: Optional[str],
        table_description: Optional[str],
        reader_fn: str,
        reader_kwargs: Dict[str, str],
    ) -> SymbolicDataModel:
        """Native DuckDB bulk load shared by the ``from_csv`` / ``from_parquet``
        / ``from_json`` / ``from_jsonl`` helpers.

        Cycles the sandboxed connection out for a loose one because
        ``read_*`` table functions need ``enable_external_access=true``.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        raw_name = (
            table_name
            if table_name is not None
            else os.path.splitext(os.path.basename(path))[0]
        )
        if not to_pascal_case(raw_name):
            raise ValueError(
                f"Cannot derive a table name from {raw_name!r}: "
                "after PascalCase normalization, no alphanumeric "
                "content remains. Pass an explicit `table_name=`."
            )
        table_name = table_identifier(raw_name)

        if reader_kwargs:
            reader_extras = ", " + ", ".join(f"{k}={v}" for k, v in reader_kwargs.items())
        else:
            reader_extras = ""
        reader_sql = f"{reader_fn}(?{reader_extras})"

        self._con.close()
        self._con = None
        try:
            loose = self._open_loose_connection()
            try:
                desc_rows = loose.execute(
                    f"DESCRIBE SELECT * FROM {reader_sql}",
                    [path],
                ).fetchall()
                if not desc_rows:
                    raise ValueError(f"{path}: no columns detected in source file.")

                # orig_col_names quote-survive header values; db_col_names
                # are snake_cased. INSERT maps positionally between them.
                col_defs = []
                orig_col_names = []
                db_col_names = []
                for i, row in enumerate(desc_rows):
                    orig_name = row[0]
                    db_name = column_identifier(orig_name)
                    orig_col_names.append(orig_name)
                    db_col_names.append(db_name)
                    col_type = row[1]
                    if i == 0:
                        col_defs.append(f"{db_name} {col_type} PRIMARY KEY")
                    else:
                        col_defs.append(f"{db_name} {col_type}")
                pk_col = db_col_names[0]

                loose.execute(
                    f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
                )

                update_cols = [c for c in db_col_names if c != pk_col]
                if update_cols:
                    update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
                    conflict_clause = f"ON CONFLICT ({pk_col}) DO UPDATE SET {update_sql}"
                else:
                    conflict_clause = f"ON CONFLICT ({pk_col}) DO NOTHING"

                def _quote_src(c):
                    return '"' + c.replace('"', '""') + '"'

                select_list = ", ".join(_quote_src(c) for c in orig_col_names)
                insert_col_list = ", ".join(db_col_names)
                loose.execute(
                    f"INSERT INTO {table_name} ({insert_col_list}) "
                    f"SELECT {select_list} FROM {reader_sql} "
                    f"{conflict_clause}",
                    [path],
                )
            finally:
                loose.close()
        finally:
            self._con = self._open_main_connection()
            self._sandbox(self._con)

        schema = self._duckdb_table_to_json_schema(table_name)
        schema["title"] = table_name
        if table_description is not None:
            schema["description"] = table_description
        symbolic_model = SymbolicDataModel(schema=schema, name=table_name)

        # setdefault: preserve any richer caller-supplied DataModel.
        self.data_models.setdefault(table_name, symbolic_model)

        try:
            self._maybe_create_fulltext_index(symbolic_model)
        except Exception as e:
            warnings.warn(
                f"FTS index rebuild failed for '{table_name}'; "
                f"fulltext_search results may be stale. ({e})"
            )

        try:
            self._maybe_create_vector_index(symbolic_model)
        except Exception as e:
            warnings.warn(
                f"Vector index rebuild failed for '{table_name}'; "
                f"similarity_search will fall back to scan. ({e})"
            )

        return symbolic_model

    async def rename(
        self,
        source: Union[Any, str],
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ) -> SymbolicDataModel:
        """Rename a table and/or update its schema description.

        At least one of ``table_name`` / ``table_description`` must be
        given. When ``table_name`` changes, the FTS and HNSW indexes
        are dropped and rebuilt under the new name, and the adapter's
        known-tables registry is updated so subsequent searches find
        the table at its new name.

        Args:
            source: ``SymbolicDataModel`` for the table to rename, or
                its name as a string. String form is PascalCase-
                normalized so callers can pass the original filename-
                style input they used in :meth:`from_csv`.
            table_name: New table name. Optional. Always normalized to
                PascalCase.
            table_description: New schema description. Optional. Lives
                in the ``SymbolicDataModel`` layer (DuckDB doesn't
                carry per-table descriptions natively).

        Returns:
            A fresh :class:`SymbolicDataModel` for the (possibly
            renamed) table, reflecting the post-rename column shape
            and the supplied description.
        """
        if table_name is None and table_description is None:
            raise ValueError(
                "rename(): pass at least one of `table_name=` or `table_description=`."
            )

        if isinstance(source, str):
            raw_old = source
        else:
            raw_old = source.get_schema().get("title")
            if not raw_old:
                raise ValueError(
                    "rename(): source SymbolicDataModel has no schema "
                    "title; cannot determine the table to rename."
                )
        old_name = table_identifier(raw_old)

        with self._connect(read_only=True) as con:
            exists = con.execute(
                f"SELECT COUNT(*) FROM information_schema.tables "
                f"WHERE table_schema='{MAIN_TABLE}' AND table_name=?",
                [old_name],
            ).fetchone()[0]
        if not exists:
            raise ValueError(
                f"rename(): no table named {old_name!r} found in the knowledge base."
            )

        new_name = old_name
        if table_name is not None:
            new_name = table_identifier(table_name)

            if new_name != old_name:
                # FTS/HNSW indexes are name-bound; drop then rebuild.
                with self._connect(read_only=False) as con:
                    try:
                        con.execute(f"PRAGMA drop_fts_index('main.{old_name}');")
                    except duckdb.Error:
                        pass

                    old_vector_index = f"vector_main_{old_name}"
                    con.execute(f"DROP INDEX IF EXISTS {old_vector_index};")

                    con.execute(f"ALTER TABLE {old_name} RENAME TO {new_name};")

        schema = self._duckdb_table_to_json_schema(new_name)
        schema["title"] = new_name
        if table_description is not None:
            schema["description"] = table_description
        else:
            if not isinstance(source, str):
                old_schema = source.get_schema()
                if "description" in old_schema:
                    schema["description"] = old_schema["description"]
        renamed_model = SymbolicDataModel(schema=schema, name=new_name)

        self.data_models.pop(old_name, None)
        self.data_models[new_name] = renamed_model

        try:
            self._maybe_create_fulltext_index(renamed_model)
        except Exception as e:
            warnings.warn(
                f"FTS index rebuild failed for '{new_name}'; "
                f"fulltext_search results may be stale. ({e})"
            )
        try:
            self._maybe_create_vector_index(renamed_model)
        except Exception as e:
            warnings.warn(
                f"Vector index rebuild failed for '{new_name}'; "
                f"similarity_search will fall back to scan. ({e})"
            )

        return renamed_model

    async def get(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
        remove_embedding: bool = True,
    ) -> Union[Optional[JsonDataModel], List[Optional[JsonDataModel]]]:
        """Look up one or more records by primary key in a single table.

        Args:
            id_or_ids: A single primary key value, or a list of values.
            table_name: Target table.
            remove_embedding: Strip the embedding column from the
                returned records. Defaults to ``True`` to keep results
                LM-friendly.

        Returns:
            For a scalar id: the matching ``JsonDataModel``, or ``None``
            if not found. For a list of ids: a list in the same order,
            with ``None`` in the slots that didn't match.
        """
        return_single = not isinstance(id_or_ids, list)
        ids = [id_or_ids] if return_single else list(id_or_ids)

        if not ids:
            return None if return_single else []

        table = table_identifier(table_name)
        json_schema = self._duckdb_table_to_json_schema(table)
        id_key = self._get_id_key(json_schema)

        results: List[Optional[JsonDataModel]] = [None] * len(ids)

        with self._connect(read_only=True) as con:
            placeholders = ", ".join(["?"] * len(ids))
            try:
                sql = f"SELECT * FROM {table} WHERE {id_key} IN ({placeholders})"
                cursor = con.execute(sql, ids)
            except Exception as e:
                warnings.warn(f"get(): SELECT from '{table}' failed. ({e})")
                return None if return_single else results

            rows = cursor.arrow().read_all().to_pylist()
            if not rows:
                return None if return_single else results

            json_columns = _get_json_columns_from_schema(json_schema)
            rows_by_id = {row[id_key]: row for row in rows}

            for idx, id_val in enumerate(ids):
                row = rows_by_id.get(id_val)
                if row is None:
                    continue
                json_data = _parse_json_columns(row, json_columns)
                if remove_embedding and self.vss_key in json_data:
                    json_data.pop(self.vss_key)
                results[idx] = JsonDataModel(
                    json=json_data,
                    schema=json_schema,
                    name=str(json_data.get(id_key)),
                )

        return results[0] if return_single else results

    async def getall(
        self,
        *,
        table_name: str,
        limit: int = 50,
        offset: int = 0,
        remove_embedding: bool = True,
    ) -> List[JsonDataModel]:
        """List rows from a single table, paginated.

        Returns an empty list (with a warning) if the table doesn't
        exist, so callers can safely enumerate without pre-checking.

        Args:
            table_name: Target table.
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            remove_embedding: Strip the embedding column from results.

        Returns:
            A list of :class:`JsonDataModel` records.
        """
        table = table_identifier(table_name)
        try:
            json_schema = self._duckdb_table_to_json_schema(table)
        except duckdb.Error as e:
            warnings.warn(f"Failed to read table '{table}': {e}")
            return []
        id_key = self._get_id_key(json_schema)

        with self._connect(read_only=True) as con:
            sql = f"SELECT * FROM {table} LIMIT ? OFFSET ?"
            try:
                cursor = con.execute(sql, [limit, offset])
                rows = cursor.arrow().read_all().to_pylist()
            except duckdb.Error as e:
                warnings.warn(f"Failed to read table '{table}': {e}")
                return []

            if not rows:
                return []

            json_columns = _get_json_columns_from_schema(json_schema)

            results = []
            for row in rows:
                json_data = _parse_json_columns(row, json_columns)
                if remove_embedding and self.vss_key in json_data:
                    json_data.pop(self.vss_key)
                results.append(
                    JsonDataModel(
                        json=json_data,
                        schema=json_schema,
                        name=str(json_data.get(id_key)),
                    )
                )
            return results

    async def delete(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
    ) -> int:
        """Delete records by primary key from a single table.

        Rebuilds the FTS and HNSW indexes after the delete so
        subsequent search calls don't return ghost rows.

        Args:
            id_or_ids: A single primary key value, or a list of values.
            table_name: Target table.

        Returns:
            The number of rows actually deleted (0 if none matched or
            if the table doesn't exist).
        """
        ids = [id_or_ids] if not isinstance(id_or_ids, list) else list(id_or_ids)
        if not ids:
            return 0

        table = table_identifier(table_name)
        try:
            json_schema = self._duckdb_table_to_json_schema(table)
        except duckdb.Error:
            warnings.warn(f"delete(): no table named '{table}'; nothing to delete.")
            return 0
        id_key = self._get_id_key(json_schema)

        placeholders = ", ".join(["?"] * len(ids))
        sql = (
            f"DELETE FROM {table} WHERE {id_key} IN ({placeholders}) RETURNING {id_key};"
        )

        with self._connect(read_only=False) as con:
            try:
                rows = con.execute(sql, ids).fetchall()
            except Exception as e:
                raise RuntimeError(f"delete from '{table}' failed: {e}") from e

        deleted = len(rows)

        symbolic_model = self.data_models.get(table)
        if symbolic_model is None:
            symbolic_model = SymbolicDataModel(schema=json_schema, name=table)

        try:
            self._maybe_create_fulltext_index(symbolic_model)
        except Exception as e:
            warnings.warn(
                f"FTS index rebuild failed for '{table}' after delete; "
                f"fulltext_search results may be stale. ({e})"
            )
        try:
            self._maybe_create_vector_index(symbolic_model)
        except Exception as e:
            warnings.warn(
                f"Vector index rebuild failed for '{table}' after "
                f"delete; similarity_search will fall back to scan. ({e})"
            )

        return deleted

    async def drop_table(self, table_name: str) -> bool:
        """Drop a table and its associated FTS / HNSW indexes.

        Also removes the table from the adapter's known-tables
        registry so subsequent operations stop seeing it.

        Args:
            table_name: Target table.

        Returns:
            ``True`` if a table was dropped, ``False`` if no such
            table existed.
        """
        table = table_identifier(table_name)

        with self._connect(read_only=True) as con:
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                f"WHERE table_schema='{MAIN_TABLE}' AND table_name=?",
                [table],
            ).fetchone()[0]
        if not exists:
            return False

        with self._connect(read_only=False) as con:
            # FTS schema doesn't cascade-drop with the table.
            try:
                con.execute(f"PRAGMA drop_fts_index('main.{table}');")
            except duckdb.Error:
                pass

            vector_index = f"vector_main_{table}"
            con.execute(f"DROP INDEX IF EXISTS {vector_index};")

            con.execute(f"DROP TABLE IF EXISTS {table};")

        self.data_models.pop(table, None)
        return True

    async def sql(
        self,
        sql: str,
        *,
        params: Optional[List[Any]] = None,
        read_only: bool = True,
        output_format: str = "json",
        **kwargs,
    ):
        """Execute a raw SQL query against the database.

        Args:
            sql: The SQL string.
            params: Optional positional parameters for the query.
            read_only: When ``True`` (default), enforces SELECT-only at
                the parser layer to reject multi-statement injection,
                ``COPY ... TO '/path'`` filesystem writes, and other
                non-SELECT statements. Set ``False`` for trusted
                mutations.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
        """
        # read_only enforces SELECT-only at the parser. A connection-level
        # RO flag alone would let `COPY (...) TO '/path'` through.
        if read_only:
            try:
                statements = duckdb.extract_statements(sql)
            except duckdb.Error as e:
                raise duckdb.InvalidInputException(f"Invalid SQL: {e}") from e
            if not statements:
                raise duckdb.InvalidInputException("Empty SQL query.")
            for stmt in statements:
                if stmt.type != duckdb.StatementType.SELECT:
                    raise duckdb.InvalidInputException(
                        f"read_only=True only permits SELECT statements; "
                        f"got {stmt.type.name}."
                    )
        with self._connect(read_only=read_only) as con:
            arrow_table = con.execute(sql, params or []).arrow().read_all()
        return format_search_results(arrow_table, output_format)

    @contextmanager
    def _hnsw_ef_search_override(self, con, ef_search: Optional[int]):
        """Apply ``SET hnsw_ef_search = <n>`` for the duration of a block.

        DuckDB exposes ``hnsw_ef_search`` as a session-level override
        of the HNSW ``ef_search`` parameter baked into the index at
        ``CREATE INDEX`` time. ``None`` is a no-op so the default (or
        the index-time value) wins; otherwise we ``SET`` on entry and
        ``RESET`` on exit so we don't leak the override onto whatever
        the next caller wants.
        """
        if ef_search is None:
            yield
            return
        con.execute(f"SET hnsw_ef_search = {int(ef_search)};")
        try:
            yield
        finally:
            con.execute("RESET hnsw_ef_search;")

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
            text_or_texts: Query text, or list of query texts. Multiple
                queries are merged into a single ranked result set
                (best score per id kept).
            table_name: Target table.
            k: Maximum number of rows returned.
            threshold: Optional maximum vector distance — rows beyond
                this distance are dropped.
            ef_search: Optional override for HNSW's search-time
                candidate-list depth. ``None`` keeps the index-time
                value (or DuckDB's default of 64); higher = better
                recall at slower query time. Mirrors Ladybug's
                ``ef_search``.
            output_format: ``"json"`` (list of dicts, default — Python
                data) or ``"csv"`` (CSV string, more compact for LM
                input).
        """
        if not text_or_texts:
            return format_search_results([], output_format)
        if not self.embedding_model:
            raise ValueError(
                "similarity_search requires an embedding model on the adapter."
            )

        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts

        label = table_identifier(table_name)
        schema = self._duckdb_table_to_json_schema(label)
        id_key = self._get_id_key(schema)

        embeddings = await self.embedding_model(EmbeddingRequest(texts=texts))
        vectors = embeddings.get("embeddings")
        if vectors:
            await self._ensure_vector_dim(vectors[0])
        dist_fn = _VSS_DISTANCE_FN[self.metric]
        # Canonical score, shared with LanceDBAdapter. ``array_distance`` is the
        # (non-squared) Euclidean distance, so square it for "l2sq" to match
        # LanceDB's squared-L2; "cosine"/"ip" use the raw function. ``threshold``
        # is compared in the same units as the score (one ``?`` per occurrence).
        _d = f"{dist_fn}({self.vss_key}, ?::FLOAT[{self.vector_dim}])"
        score_expr = f"power({_d}, 2)" if self.metric == "l2sq" else _d

        if len(vectors) == 1:
            vector = vectors[0]
            where_clause = f"WHERE {score_expr} < ?" if threshold else ""
            sql = f"""
                SELECT *,
                    {score_expr} AS score
                FROM {label}
                {where_clause}
                ORDER BY score ASC
                LIMIT ?;
            """
            params = [vector]
            if threshold is not None:
                params.extend([vector, threshold])
            params.append(k)

            with (
                self._connect(read_only=True) as con,
                self._hnsw_ef_search_override(con, ef_search),
            ):
                try:
                    arrow_table = con.execute(sql, params).arrow().read_all()
                except Exception as e:
                    raise RuntimeError(f"Vector search failed for table '{label}': {e}")
            return format_search_results(arrow_table, output_format)

        # Multi-query: dedupe by id, keep best score, take top-k.
        results: Dict[Any, Dict[str, Any]] = {}
        with (
            self._connect(read_only=True) as con,
            self._hnsw_ef_search_override(con, ef_search),
        ):
            for vector in vectors:
                where_clause = f"WHERE {score_expr} < ?" if threshold else ""
                sql = f"""
                    SELECT *,
                        {score_expr} AS score
                    FROM {label}
                    {where_clause}
                    ORDER BY score ASC
                    LIMIT ?;
                """
                params = [vector]
                if threshold is not None:
                    params.extend([vector, threshold])
                params.append(k)

                try:
                    rows = con.execute(sql, params).arrow().read_all().to_pylist()
                except Exception as e:
                    raise RuntimeError(f"Vector search failed for table '{label}': {e}")

                for row in rows:
                    uid = row[id_key]
                    prev = results.get(uid)
                    if prev is None or row["score"] < prev["score"]:
                        results[uid] = row

        ranked = sorted(results.values(), key=lambda r: r["score"])[:k]
        return format_search_results(ranked, output_format)

    def _bm25_call(
        self,
        id_key: str,
        *,
        bm25_b: Optional[float] = None,
        bm25_k: Optional[float] = None,
        conjunctive: Optional[bool] = None,
    ) -> str:
        """Render a ``match_bm25(id, ?[, named-args])`` call fragment.

        The ``?`` placeholder is reserved for the query string (bound
        as a parameter by the caller). BM25 tuning knobs (``b``,
        ``k``, ``conjunctive``) are inlined via DuckDB's ``key := value``
        named-argument syntax so we don't have to specify the
        positional args we want to leave at the engine default. Only
        non-``None`` overrides are emitted.
        """
        extras: List[str] = []
        for key, value in (
            ("k", bm25_k),
            ("b", bm25_b),
            ("conjunctive", conjunctive),
        ):
            if value is None:
                continue
            extras.append(f"{key} := {self._render_sql_literal(value)}")
        suffix = (", " + ", ".join(extras)) if extras else ""
        return f"match_bm25({id_key}, ?{suffix})"

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
            text_or_texts: Query text, or list of query texts. Multiple
                queries are merged (best BM25 per id kept).
            table_name: Target table.
            k: Maximum number of rows returned.
            threshold: Optional minimum relevance on the normalized
                ``[0, 1]`` scale. The returned result set is min-max
                scaled (best hit ``1.0``, worst ``0.0``) so ``score`` is
                comparable with the LanceDB adapter despite different raw
                BM25 ranges.
            conjunctive: AND-mode query (every term must match).
                Default ``False`` keeps OR semantics (DuckDB's
                ``conjunctive=0``). Mirrors Ladybug's ``conjunctive``.
            bm25_b: Optional override for BM25's ``b`` parameter
                (document-length normalization). ``None`` defers to
                DuckDB's default (0.75).
            bm25_k: Optional override for BM25's ``k1`` parameter
                (term-frequency saturation). ``None`` defers to
                DuckDB's default (1.2). Named ``bm25_k`` to avoid
                collision with the result-limit ``k`` above.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
        """
        if not text_or_texts:
            return format_search_results([], output_format)

        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts

        label = table_identifier(table_name)
        schema = self._duckdb_table_to_json_schema(label)
        id_key = self._get_id_key(schema)
        if not self._has_indexable_text_columns(schema):
            warnings.warn(f"Skipping FTS search for {label}: no text columns to index.")
            return format_search_results([], output_format)
        fts_table = sanitize_identifier(f"fts_main_{label}")
        # Build the BM25 call once; the named-arg fragments (b/k/
        # conjunctive) are inlined as literals, leaving the ``?`` for
        # the query string bound per iteration. ``conjunctive=False``
        # is DuckDB's default, so only ``True`` triggers an override.
        bm25_call = self._bm25_call(
            id_key,
            bm25_b=bm25_b,
            bm25_k=bm25_k,
            conjunctive=conjunctive or None,
        )

        sql = f"""
            SELECT t.*, fts.score
            FROM {label} t
            JOIN (
                SELECT
                    {id_key},
                    {fts_table}.{bm25_call} AS score
                FROM {label}
            ) fts ON t.{id_key} = fts.{id_key}
            WHERE fts.score IS NOT NULL
            ORDER BY fts.score DESC
            LIMIT ?;
        """

        # Fetch the top-k by raw BM25 (dedupe across queries, keep best), then
        # rescale to [0, 1] so scores are comparable with the LanceDB adapter.
        results: Dict[Any, Dict[str, Any]] = {}
        with self._connect(read_only=True) as con:
            for text in texts:
                try:
                    rows = con.execute(sql, [text, k]).arrow().read_all().to_pylist()
                except Exception as e:
                    raise RuntimeError(f"FTS query failed for table '{label}': {e}")
                for row in rows:
                    uid = row[id_key]
                    prev = results.get(uid)
                    if prev is None or row["score"] > prev["score"]:
                        results[uid] = row

        ranked = sorted(results.values(), key=lambda r: r["score"], reverse=True)[:k]
        # ``threshold`` filters on the same normalized [0, 1] scale as the score.
        minmax_normalize_scores(ranked, key="score")
        if threshold is not None:
            ranked = [r for r in ranked if r["score"] >= threshold]
        return format_search_results(ranked, output_format)

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

        Uses DuckDB's ``regexp_matches`` under the hood. DuckDB ships
        RE2 (Google's regex library) so evaluation is linear-time — no
        catastrophic-backtracking exposure even if ``pattern`` comes
        from an untrusted source.

        Args:
            pattern: The regex pattern (RE2 syntax).
            table_name: Target table.
            fields: Field names to match against. Defaults to every
                string-typed field on the schema. Names not present
                as string columns are silently dropped.
            case_sensitive: When ``False``, matches case-insensitively.
            k: Maximum number of rows returned.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
        """
        if not pattern:
            return format_search_results([], output_format)

        label = table_identifier(table_name)
        schema = self._duckdb_table_to_json_schema(label)
        properties = schema.get("properties", {})

        string_cols = [
            column_identifier(name)
            for name, info in properties.items()
            if info.get("type") == "string"
        ]
        if fields is not None:
            requested = {column_identifier(f) for f in fields}
            cols = [c for c in string_cols if c in requested]
        else:
            cols = string_cols
        if not cols:
            warnings.warn(
                f"Skipping regex search for {label}: no matching string fields."
            )
            return format_search_results([], output_format)

        flag = "i" if not case_sensitive else ""
        where = " OR ".join(f"regexp_matches({c}, ?, ?)" for c in cols)
        sql = f"SELECT * FROM {label} WHERE {where} LIMIT ?;"
        params = []
        for _ in cols:
            params.extend([pattern, flag])
        params.append(k)

        with self._connect(read_only=True) as con:
            try:
                arrow_table = con.execute(sql, params).arrow().read_all()
            except Exception as e:
                raise RuntimeError(f"Regex query failed for table '{label}': {e}")
        return format_search_results(arrow_table, output_format)

    async def hybrid_search(self, *args, **kwargs):
        """Deprecated alias of :meth:`hybrid_fts_search`.

        Kept so call sites pre-dating the rename keep working. Prefer
        the new name in new code — it's symmetric with
        :meth:`hybrid_regex_search`.
        """
        return await self.hybrid_fts_search(*args, **kwargs)

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

        Internally runs :meth:`similarity_search` and
        :meth:`fulltext_search` against the same ``table_name``, then
        fuses their rankings with the RRF formula
        ``sum(1 / (k_rank + rank))``. Falls back to pure FTS if no
        embedding model is configured.

        ``text_or_texts`` feeds the vector branch; ``keywords`` (when
        provided) feeds the BM25 branch instead — the two signals
        look for different things (semantic vs lexical) and the
        natural-language query that drives the vectors is usually
        not the keyword set you'd hand to BM25. When ``keywords`` is
        omitted, the text is reused for both branches so existing
        call sites keep working.

        Args:
            text_or_texts: Query text or list of query texts for the
                vector branch.
            table_name: Target table.
            keywords: Query text or list of query texts for the BM25
                branch. Aligns by position with ``text_or_texts``;
                when omitted, the text is reused.
            k: Maximum number of rows returned.
            k_rank: RRF smoothing constant (default 60). Lower values
                weight top-ranked rows more strongly.
            similarity_threshold: Optional maximum vector distance.
            fulltext_threshold: Optional minimum fulltext relevance on
                the normalized ``[0, 1]`` scale.
            ef_search: Forwarded to the vector branch
                (:meth:`similarity_search`); overrides HNSW's
                search-time candidate-list depth.
            conjunctive: Forwarded to the BM25 branch
                (:meth:`fulltext_search`); switches to AND-mode query.
            bm25_b: Forwarded to the BM25 branch; document-length
                normalization override.
            bm25_k: Forwarded to the BM25 branch; term-frequency
                saturation (``k1``) override.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
        """
        if not text_or_texts:
            return format_search_results([], output_format)

        if not self.embedding_model:
            # Fulltext-only fallback. Prefer explicit keywords when
            # the caller passed them — that's what the BM25 branch
            # would have used in the full hybrid path anyway. Tag
            # each row with ``rrf_score`` (set to the BM25 score) so
            # the result shape matches the full-hybrid path; callers
            # can always read ``rrf_score`` without branching.
            fts_rows = await self.fulltext_search(
                keywords if keywords is not None else text_or_texts,
                table_name=table_name,
                k=k,
                threshold=fulltext_threshold,
                conjunctive=conjunctive,
                bm25_b=bm25_b,
                bm25_k=bm25_k,
                output_format="json",
            )
            for row in fts_rows:
                row.setdefault("rrf_score", row.get("score", 0.0))
                row.setdefault("fulltext_score", row.get("score", 0.0))
            return format_search_results(fts_rows, output_format)

        queries = (
            [text_or_texts] if isinstance(text_or_texts, str) else list(text_or_texts)
        )
        if keywords is None:
            keyword_queries = list(queries)
        else:
            keyword_queries = [keywords] if isinstance(keywords, str) else list(keywords)
            if len(keyword_queries) != len(queries):
                raise ValueError(
                    f"`keywords` must align with `text_or_texts`: got "
                    f"{len(keyword_queries)} keyword(s) vs "
                    f"{len(queries)} text(s)."
                )

        label = table_identifier(table_name)
        schema = self._duckdb_table_to_json_schema(label)
        id_key = self._get_id_key(schema)

        final_results: Dict[Any, Dict[str, Any]] = {}

        for query_text, keyword_text in zip(queries, keyword_queries):
            try:
                try:
                    fts_results = await self.fulltext_search(
                        keyword_text,
                        table_name=label,
                        k=k * 5,
                        threshold=fulltext_threshold,
                        conjunctive=conjunctive,
                        bm25_b=bm25_b,
                        bm25_k=bm25_k,
                        output_format="json",
                    )
                except Exception:
                    fts_results = []
                try:
                    vss_results = await self.similarity_search(
                        query_text,
                        table_name=label,
                        k=k * 5,
                        threshold=similarity_threshold,
                        ef_search=ef_search,
                        output_format="json",
                    )
                except Exception:
                    vss_results = []

                if not fts_results and not vss_results:
                    warnings.warn(f"No results for query='{query_text}'.")
                    continue

                fts_rank = {r[id_key]: i + 1 for i, r in enumerate(fts_results)}
                vss_rank = {r[id_key]: i + 1 for i, r in enumerate(vss_results)}

                combined_rows: Dict[Any, Dict[str, Any]] = {}
                for row in fts_results + vss_results:
                    uid = row[id_key]
                    if uid not in combined_rows:
                        combined_rows[uid] = dict(row)
                    else:
                        combined_rows[uid].update(row)

                # RRF formula: sum(1 / (k_rank + rank))
                for uid in set(fts_rank) | set(vss_rank):
                    score = 0.0
                    if uid in fts_rank:
                        score += 1.0 / (k_rank + fts_rank[uid])
                    if uid in vss_rank:
                        score += 1.0 / (k_rank + vss_rank[uid])
                    combined_rows[uid]["score"] = score

                top_rows = heapq.nlargest(
                    k, combined_rows.values(), key=lambda r: r["score"]
                )
                for r in top_rows:
                    uid = r[id_key]
                    if (
                        uid not in final_results
                        or r["score"] > final_results[uid]["score"]
                    ):
                        final_results[uid] = r
            except Exception as e:
                warnings.warn(f"Hybrid search iteration failed: {e}")
                continue

        results_sorted = sorted(
            heapq.nlargest(k, final_results.values(), key=lambda r: r["score"]),
            key=lambda r: (-r["score"], r.get(id_key)),
        )
        return format_search_results(results_sorted, output_format)

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
        """Reciprocal-Rank-Fusion of vector similarity + regex match.

        Sibling of :meth:`hybrid_fts_search`. The vector half embeds
        ``text_or_texts``; the regex half matches
        ``pattern_or_patterns`` against the table's string columns.
        Degenerates to plain :meth:`similarity_search` when no
        patterns are supplied, or to plain :meth:`regex_search` when
        no embedding model is configured.

        Args:
            text_or_texts: Query text (or list) for the vector half.
            pattern_or_patterns: RE2 pattern (or list) for the regex
                half. ``None`` skips the regex half.
            table_name: Target table.
            k: Maximum number of rows returned.
            k_rank: RRF smoothing constant.
            similarity_threshold: Optional maximum vector distance.
            ef_search: Forwarded to the vector branch
                (:meth:`similarity_search`); overrides HNSW's
                search-time candidate-list depth.
            fields: Forwarded to :meth:`regex_search`.
            case_sensitive: Forwarded to :meth:`regex_search`.
            output_format: ``"json"`` (list of dicts, default) or
                ``"csv"`` (CSV string).
        """
        if not text_or_texts and not pattern_or_patterns:
            return format_search_results([], output_format)

        if not self.embedding_model:
            if not pattern_or_patterns:
                return format_search_results([], output_format)
            patterns_list = (
                [pattern_or_patterns]
                if isinstance(pattern_or_patterns, str)
                else list(pattern_or_patterns)
            )
            merged: Dict[bytes, Dict[str, Any]] = {}
            for p in patterns_list:
                rows = await self.regex_search(
                    p,
                    table_name=table_name,
                    fields=fields,
                    case_sensitive=case_sensitive,
                    k=k,
                    output_format="json",
                )
                for r in rows:
                    sig = orjson.dumps(r, default=str)
                    merged[sig] = r
            return format_search_results(list(merged.values())[:k], output_format)

        texts = [text_or_texts] if isinstance(text_or_texts, str) else list(text_or_texts)
        if pattern_or_patterns is None:
            patterns: List[str] = []
        elif isinstance(pattern_or_patterns, str):
            patterns = [pattern_or_patterns]
        else:
            patterns = list(pattern_or_patterns)

        label = table_identifier(table_name)
        schema = self._duckdb_table_to_json_schema(label)
        id_key = self._get_id_key(schema)

        final_results: Dict[Any, Dict[str, Any]] = {}

        for query_text in texts:
            try:
                try:
                    vss_results = await self.similarity_search(
                        query_text,
                        table_name=label,
                        k=k * 5,
                        threshold=similarity_threshold,
                        ef_search=ef_search,
                        output_format="json",
                    )
                except Exception:
                    vss_results = []

                rx_results: List[Dict[str, Any]] = []
                if patterns:
                    seen_rx = set()
                    for pattern in patterns:
                        try:
                            rows = await self.regex_search(
                                pattern,
                                table_name=label,
                                fields=fields,
                                case_sensitive=case_sensitive,
                                k=k * 5,
                                output_format="json",
                            )
                        except Exception:
                            rows = []
                        for row in rows:
                            sig = orjson.dumps(row, default=str)
                            if sig not in seen_rx:
                                seen_rx.add(sig)
                                rx_results.append(row)

                if not vss_results and not rx_results:
                    warnings.warn(f"No results for query='{query_text}'.")
                    continue

                vss_rank = {r[id_key]: i + 1 for i, r in enumerate(vss_results)}
                rx_rank = {r[id_key]: i + 1 for i, r in enumerate(rx_results)}

                combined_rows: Dict[Any, Dict[str, Any]] = {}
                for row in vss_results + rx_results:
                    uid = row[id_key]
                    if uid not in combined_rows:
                        combined_rows[uid] = dict(row)
                    else:
                        combined_rows[uid].update(row)

                for uid in set(vss_rank) | set(rx_rank):
                    score = 0.0
                    if uid in vss_rank:
                        score += 1.0 / (k_rank + vss_rank[uid])
                    if uid in rx_rank:
                        score += 1.0 / (k_rank + rx_rank[uid])
                    combined_rows[uid]["score"] = score

                top_rows = heapq.nlargest(
                    k, combined_rows.values(), key=lambda r: r["score"]
                )
                for r in top_rows:
                    uid = r[id_key]
                    if (
                        uid not in final_results
                        or r["score"] > final_results[uid]["score"]
                    ):
                        final_results[uid] = r
            except Exception as e:
                warnings.warn(f"Hybrid-regex iteration failed: {e}")
                continue

        results_sorted = sorted(
            heapq.nlargest(k, final_results.values(), key=lambda r: r["score"]),
            key=lambda r: (-r["score"], r.get(id_key)),
        )
        return format_search_results(results_sorted, output_format)
