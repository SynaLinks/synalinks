import heapq
import os
import re
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import duckdb
import duckdb.sqltypes
import orjson

from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend.config import synalinks_home
from synalinks.src.knowledge_bases.database_adapters.database_adapter import (
    DatabaseAdapter,
)
from synalinks.src.utils.async_utils import run_maybe_nested

FTS_KEYS = ["description", "text", "content", "message", "name", "query", "question"]

VSS_KEY = "embedding"

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
    "l2seq",
    "cosine",
    "ip",
]

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


def sanitize_identifier(name: str) -> str:
    """Allow only alphanumeric, underscore, and enforce starting with a letter."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


def sanitize_properties(properties: dict):
    """Prevent SQL injections by sanitazing the dict keys for updates.

    Args:
        properties (dict): The properties to sanitize

    Returns:
        (dict): The sanitized properties
    """
    return {sanitize_identifier(k): v for k, v in properties.items()}


class DuckDBAdapter(DatabaseAdapter):
    def __init__(
        self,
        uri=None,
        embedding_model=None,
        data_models=None,
        stemmer="porter",
        metric="cosine",
        fts_keys=FTS_KEYS,
        vss_key=VSS_KEY,
        main_table=MAIN_TABLE,
        vector_dim=None,
        wipe_on_start=False,
        name=None,
        **kwargs,
    ):
        uri = uri.replace("duckdb://", "") if uri else None
        self.uri = uri

        self.embedding_model = embedding_model

        if self.embedding_model:
            if not vector_dim:
                embedded_text = run_maybe_nested(self.embedding_model(["text"]))
                self.vector_dim = len(embedded_text["embeddings"][0])
            else:
                self.vector_dim = vector_dim

        if stemmer not in STEMMERS:
            raise ValueError(f"`stemmer` parameter should be one of {STEMMERS}")
        self.stemmer = stemmer

        if metric not in METRICS:
            raise ValueError(f"`metric` parameter should be one of {METRICS}")
        self.metric = metric

        self.fts_keys = fts_keys
        self.vss_key = vss_key

        self.uri = uri or os.path.join(
            synalinks_home(), name + ".db" if name else "database.db"
        )

        self._install_extensions()

        if wipe_on_start:
            self.wipe_database()

        if data_models:
            self.data_models = data_models
            for dm in data_models:
                self._maybe_create_table(dm)
        else:
            self.data_models = self.get_symbolic_data_models()

    def _connect(self, read_only=False):
        return duckdb.connect(self.uri, read_only=read_only)

    def _install_extensions(self):
        """Install required extensions per-connection."""
        with self._connect(read_only=False) as con:
            con.execute("INSTALL fts;")
            con.execute("LOAD fts;")
            if self.embedding_model:
                con.execute("INSTALL vss;")
                con.execute("LOAD vss;")

    def wipe_database(self):
        with self._connect(read_only=False) as con:
            tables = con.execute(
                f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='{MAIN_TABLE}';
            """
            ).fetchall()

            for (table_name,) in tables:
                try:
                    con.execute(f"DROP TABLE IF EXISTS {table_name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to drop table {table_name}: {e}")

    def _get_id_key(self, schema_or_json: dict) -> str:
        """Get the first property key from schema or json as the primary key."""
        if "properties" in schema_or_json:
            props = schema_or_json.get("properties", {})
            if props:
                return next(iter(props.keys()))
        elif schema_or_json:
            return next(iter(schema_or_json.keys()))
        raise ValueError("Cannot determine primary key: schema has no properties")

    def _get_fts_field(self, schema_or_json: dict) -> str | None:
        if "properties" in schema_or_json:
            props = schema_or_json.get("properties")
            for field in self.fts_keys:
                if field in props:
                    return field
            return None
        for field in self.fts_keys:
            if field in schema_or_json:
                return field
        return None

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
                "title": table_name.title(),
                "type": "object",
                "additionalProperties": False,
                "required": list(props.keys()),
                "properties": props,
            }

    def _json_schema_to_duckdb_columns(self, json_schema: dict):
        """Convert JSON schema to DuckDB column definitions.

        Uses the first property as the primary key.
        """
        properties = json_schema.get("properties", {})
        out = []
        first_col = True

        for prop_name, prop_spec in properties.items():
            prop_name = sanitize_identifier(prop_name)
            prop_type = prop_spec.get("type")

            if prop_name == self.vss_key:
                continue

            if not prop_type:
                raise ValueError(f"Malformed JSON schema: missing type for '{prop_name}'")

            col_def = None

            if prop_type == "array":
                item_spec = prop_spec.get("items")
                if not item_spec:
                    # Array without items spec - use JSON
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
                        # Array of objects - use JSON
                        col_def = f"{prop_name} JSON"
                    else:
                        raise ValueError(
                            f"Unsupported array item type '{item_type}' for '{prop_name}'"
                        )
            elif prop_type == "object":
                # Dict/object types stored as JSON
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

            # First column becomes primary key
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
            tables = con.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='main';
            """
            ).fetchall()

            symbolic_data_models = []
            for (table_name,) in tables:
                schema = self._duckdb_table_to_json_schema(table_name)
                model = SymbolicDataModel(schema=schema, name=table_name)
                symbolic_data_models.append(model)
            return symbolic_data_models

    def _maybe_create_table(
        self,
        data_model: Union[JsonDataModel, SymbolicDataModel],
    ):
        with self._connect(read_only=False) as con:
            json_schema = data_model.get_schema()
            table_name = sanitize_identifier(json_schema.get("title"))

            exists = con.execute(
                f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema='{MAIN_TABLE}' AND table_name='{table_name}';
            """
            ).fetchone()[0]

            if exists:
                return

            columns = self._json_schema_to_duckdb_columns(json_schema)
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"

            try:
                con.execute(create_sql)
            except Exception as e:
                raise RuntimeError(f"Failed to create table '{table_name}': {e}")

    def _maybe_create_fulltext_index(
        self,
        data_model: Union[JsonDataModel, SymbolicDataModel],
        overwrite: bool = True,
    ):
        with self._connect(read_only=False) as con:
            json_schema = data_model.get_schema()
            table_name = sanitize_identifier(json_schema.get("title"))
            col = self._get_fts_field(schema_or_json=json_schema)
            id_key = self._get_id_key(json_schema)

            if not col:
                return
            con.execute(
                f"""
                PRAGMA create_fts_index(
                    'main.{table_name}',
                    '{id_key}',
                    '{col}',
                    stemmer='{self.stemmer}',
                    overwrite={1 if overwrite else 0}
                );
            """
            )

    def _maybe_create_vector_index(
        self,
        data_model: Union[JsonDataModel, SymbolicDataModel],
        overwrite: bool = True,
    ):
        with self._connect(read_only=False) as con:
            json_schema = data_model.get_schema()
            table_name = sanitize_identifier(json_schema.get("title"))
            con.execute(
                f"CREATE INDEX vector_main_{table_name} ON {table_name}"
                f" USING HNSW ({self.vss_key})"
                f" WITH (metric = '{self.metric}');"
            )

    async def update(
        self,
        data_model_or_data_models: Union[List[JsonDataModel], JsonDataModel],
    ) -> Union[Any, List[Any]]:
        """Update or insert records. Returns the primary key value(s)."""
        with self._connect(read_only=False) as con:
            if not isinstance(data_model_or_data_models, list):
                data_models = [data_model_or_data_models]
                return_single = True
            else:
                data_models = data_model_or_data_models
                return_single = False

            ids = []

            for data_model in data_models:
                if not isinstance(data_model, JsonDataModel):
                    data_model = data_model.to_json_data_model()

                self._maybe_create_table(data_model)

                schema = data_model.get_schema()
                table = sanitize_identifier(schema["title"])
                json_data = sanitize_properties(data_model.get_json())
                id_key = self._get_id_key(schema)

                id_val = json_data.get(id_key)
                if id_val is None:
                    raise ValueError(
                        f"Primary key '{id_key}' is required but not provided"
                    )

                cols = list(json_data.keys())
                col_sql = ", ".join(cols)
                placeholders = ", ".join(["?"] * len(cols))
                params = [json_data[c] for c in cols]

                update_cols = [c for c in cols if c != id_key]
                update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

                sql = f"""
                    INSERT INTO {table} ({col_sql})
                    VALUES ({placeholders})
                    ON CONFLICT ({id_key}) DO UPDATE SET
                        {update_sql};
                """

                con.execute(sql, params)

                ids.append(id_val)

        self._maybe_create_fulltext_index(data_model)
        return ids[0] if return_single else ids

    async def get(
        self,
        id_or_ids: Any,
        data_models: List[SymbolicDataModel],
        remove_embedding: bool = True,
    ) -> Optional[JsonDataModel]:
        """Get a record by its primary key value."""
        with self._connect(read_only=True) as con:
            if not data_models:
                data_models = self.get_symbolic_data_models()

            for data_model in data_models:
                json_schema = data_model.get_schema()
                table = sanitize_identifier(json_schema.get("title"))
                id_key = self._get_id_key(json_schema)

                try:
                    sql = f"SELECT * FROM {table} WHERE {id_key} = ?"
                    cursor = con.execute(sql, [id_or_ids])
                except Exception as e:
                    warnings.warn(f" {e}")
                    continue

                rows = cursor.arrow().read_all().to_pylist()
                if not rows:
                    continue

                json_columns = _get_json_columns_from_schema(json_schema)
                json_data = _parse_json_columns(rows[0], json_columns)

                if remove_embedding:
                    if self.vss_key in json_data:
                        json_data.pop(self.vss_key)

                id_val = json_data.get(id_key)

                return JsonDataModel(
                    json=json_data,
                    schema=json_schema,
                    name=str(id_val),
                )
            return None

    async def getall(
        self,
        data_model: SymbolicDataModel,
        limit: int = 50,
        offset: int = 0,
        remove_embedding: bool = True,
    ):
        json_schema = data_model.get_schema()
        table = sanitize_identifier(json_schema.get("title"))
        id_key = self._get_id_key(json_schema)

        try:
            with self._connect(read_only=True) as con:
                sql = f"SELECT * FROM {table} LIMIT ? OFFSET ?"
                cursor = con.execute(sql, [limit, offset])
                rows = cursor.arrow().read_all().to_pylist()

                if not rows:
                    return []

                schema = data_model.get_schema()
                json_columns = _get_json_columns_from_schema(schema)

                results = []
                for row in rows:
                    json_data = _parse_json_columns(row, json_columns)

                    if remove_embedding and self.vss_key in json_data:
                        json_data.pop(self.vss_key)

                    results.append(
                        JsonDataModel(
                            json=json_data,
                            schema=schema,
                            name=str(json_data.get(id_key)),
                        )
                    )
                return results
        except Exception as e:
            warnings.warn(f"Failed to read table '{table}': {e}")
            return []

    async def query(self, query: str, params=None, read_only=True, **kwargs):
        with self._connect(read_only=read_only) as con:
            cursor = con.execute(query, params or [])
            return cursor.arrow().read_all().to_pylist()

    async def similarity_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[SymbolicDataModel]] = None,
        k: int = 10,
        threshold: float = None,
    ):
        with self._connect(read_only=True) as con:
            if not text_or_texts:
                return []

            texts = (
                [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts
            )

            if not data_models:
                data_models = self.get_symbolic_data_models()

            results = {}

            for model in data_models:
                schema = model.get_schema()
                label = sanitize_identifier(schema.get("title"))
                id_key = self._get_id_key(schema)

                vectors = await self.embedding_model(texts)
                vectors = vectors.get("embeddings")

                for vector in vectors:
                    where_clause = (
                        (
                            f"WHERE array_distance({self.vss_key}, "
                            f"?::FLOAT[{self.vector_dim}]) < ?"
                        )
                        if threshold
                        else ""
                    )

                    sql = f"""
                        SELECT *,
                            array_distance(
                                {self.vss_key}, ?::FLOAT[{self.vector_dim}]
                            ) AS score
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
                        cursor = con.execute(sql, params)
                        rows = cursor.arrow().read_all().to_pylist()
                    except Exception as e:
                        raise RuntimeError(
                            f"Vector search failed for table '{label}': {e}"
                        )

                    for row in rows:
                        results[row[id_key]] = row

            ranked = sorted(results.values(), key=lambda r: r["score"])
            return ranked[:k]

    async def fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[SymbolicDataModel]] = None,
        k: int = 10,
        threshold: float = None,
    ):
        if not text_or_texts:
            return []

        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts

        if not data_models:
            data_models = self.get_symbolic_data_models()

        results = {}
        with self._connect(read_only=True) as con:
            for model in data_models:
                schema = model.get_schema()
                label = sanitize_identifier(schema.get("title"))
                id_key = self._get_id_key(schema)
                col = self._get_fts_field(schema)
                if not col:
                    warnings.warn(
                        f"Skipping FTS search for {label}: no FTS field found."
                    )
                    continue

                fts_table = sanitize_identifier(f"fts_main_{label}")

                for text in texts:
                    sql = f"""
                        SELECT t.*, fts.score
                        FROM {label} t
                        JOIN (
                            SELECT
                                {id_key},
                                {fts_table}.match_bm25({id_key}, ?) AS score
                            FROM {label}
                        ) fts ON t.{id_key} = fts.{id_key}
                        WHERE fts.score IS NOT NULL
                        ORDER BY fts.score DESC
                        LIMIT ?;
                    """
                    params = [text]
                    if threshold:
                        params.append(threshold)
                    params.append(k)
                    try:
                        cursor = con.execute(sql, params)
                        rows = cursor.arrow().read_all().to_pylist()
                    except Exception as e:
                        raise RuntimeError(f"FTS query failed for table '{label}': {e}")

                    for row in rows:
                        results[row[id_key]] = row

            ranked = sorted(results.values(), key=lambda r: r["score"])
            return ranked[:k]

    async def hybrid_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[SymbolicDataModel]] = None,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: float = None,
        fulltext_threshold: float = None,
    ):
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).

        Combines ranks from:
          - Full-text search (BM25-based)
          - Vector similarity search (embedding distance)

        Args:
            text_or_texts: Single query or list of queries.
            data_models: Symbolic data models to search (optional).
            k: Maximum number of results to return.
            similarity_threshold: Optional maximum distance for vector search
                (lower = better).
            fulltext_threshold: Optional minimum BM25 score for text search
                (higher = better).
            k_rank: RRF smoothing constant. Lower emphasizes top ranks (default=60).

        Returns:
            List of dicts containing merged, reranked results with "score".
        """

        if not text_or_texts:
            return []

        if not self.embedding_model:
            return await self.fulltext_search(text_or_texts, data_models, k)

        queries = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
        if not data_models:
            data_models = self.get_symbolic_data_models()

        final_results = {}

        for query_text in queries:
            try:
                try:
                    fts_results = await self.fulltext_search(
                        query_text, data_models, k=k * 5, threshold=fulltext_threshold
                    )
                except Exception:
                    fts_results = []

                try:
                    vss_results = await self.similarity_search(
                        query_text, data_models, k=k * 5, threshold=similarity_threshold
                    )
                except Exception:
                    vss_results = []

                if not fts_results and not vss_results:
                    warnings.warn(f"No results for query='{query_text}'.")
                    continue

                # Get id_key from first data model
                id_key = self._get_id_key(data_models[0].get_schema())

                fts_rank = {r[id_key]: i + 1 for i, r in enumerate(fts_results)}
                vss_rank = {r[id_key]: i + 1 for i, r in enumerate(vss_results)}

                combined_rows: Dict[str, Dict[str, Any]] = {}

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

        results_sorted = heapq.nlargest(
            k, final_results.values(), key=lambda r: r["score"]
        )

        # Get id_key for final sorting
        if data_models:
            id_key = self._get_id_key(data_models[0].get_schema())
        else:
            id_key = next(iter(results_sorted[0].keys())) if results_sorted else "id"

        return sorted(results_sorted, key=lambda r: (-r["score"], r.get(id_key)))
