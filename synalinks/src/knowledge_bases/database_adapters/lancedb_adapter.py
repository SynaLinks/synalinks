# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""LanceDB database adapter.

LanceDB is a vector-native, embedded columnar store. This adapter mirrors the
:class:`DuckDBAdapter` contract (same public methods, same result shapes) on top
of LanceDB:

- vector similarity search is native (``table.search(vector)``),
- full-text search uses LanceDB's Tantivy-backed FTS index,
- ``hybrid_fts_search`` / ``hybrid_regex_search`` reuse the engine-agnostic RRF
  fusion (they just call the single-signal search methods),
- ``regex_search`` scans the column(s) and filters with Python ``re`` (RE2-free
  but correct),
- ``sql()`` is delegated to **DuckDB**, which scans the Lance datasets in place
  (LanceDB has no SQL engine of its own).

LanceDB has no primary-key constraint, so upserts are done with
``merge_insert(on=<pk>)`` keyed off the first declared field — the same
"primary key = first field" convention the DuckDB adapter uses. The original
JSON schema is stashed in the Arrow schema metadata so reflection round-trips
losslessly (object/array-typed columns are stored as JSON strings).
"""

import os
import re
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import orjson
import pyarrow as pa

from synalinks.src.backend import EmbeddingRequest
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.knowledge_bases.adapters_utils import align_keywords
from synalinks.src.knowledge_bases.adapters_utils import format_search_results
from synalinks.src.knowledge_bases.adapters_utils import minmax_normalize_scores
from synalinks.src.knowledge_bases.adapters_utils import resolve_db_path
from synalinks.src.knowledge_bases.adapters_utils import to_pascal_identifier
from synalinks.src.knowledge_bases.adapters_utils import to_snake_identifier
from synalinks.src.knowledge_bases.database_adapters.database_adapter import (
    DatabaseAdapter,
)
from synalinks.src.modules.embedding_models import get as _get_em
from synalinks.src.utils.async_utils import run_maybe_nested

VSS_KEY = "embedding"
# Canonical metric vocabulary, shared with ``DuckDBAdapter`` so the two adapters
# stay drop-in compatible. ``_NATIVE_METRIC`` maps each canonical name to the
# string LanceDB's ``Query.metric()`` actually expects.
METRICS = ["l2sq", "cosine", "ip"]
_NATIVE_METRIC = {"l2sq": "l2", "cosine": "cosine", "ip": "dot"}
# Additive shift turning LanceDB's raw ``_distance`` into the canonical score
# shared with DuckDB. LanceDB's "dot" distance is ``1 - <a,b>`` while the
# canonical "ip" score is ``-<a,b>``, so shift by -1. "l2sq" (squared L2) and
# "cosine" already agree with DuckDB and need no shift.
_SCORE_OFFSET = {"l2sq": 0.0, "cosine": 0.0, "ip": -1.0}
_SCHEMA_META_KEY = b"synalinks_schema"
_DATE_LIKE_FORMATS = frozenset({"date", "date-time", "time"})

table_identifier = to_pascal_identifier


def _column_identifier(name: str) -> str:
    return to_snake_identifier(name)


def _sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal for a LanceDB/Lance filter."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


class LanceDBAdapter(DatabaseAdapter):
    def __init__(
        self,
        uri=None,
        embedding_model=None,
        data_models=None,
        metric="cosine",
        vss_key=VSS_KEY,
        vector_dim=None,
        wipe_on_start=False,
        name=None,
        **kwargs,
    ):
        import lancedb

        # ``lancedb://<path>`` -> ``<path>``; ``None`` -> ~/.synalinks/<name>.lance
        self.uri = resolve_db_path(uri, scheme="lancedb", extension="lance", name=name)

        self.embedding_model = _get_em(embedding_model)
        self.vector_dim = None
        if self.embedding_model:
            if not vector_dim:
                probe = run_maybe_nested(
                    self.embedding_model(EmbeddingRequest(texts=["text"]))
                )
                embeddings = probe.get("embeddings") if probe is not None else None
                if not embeddings:
                    raise ValueError(
                        f"Embedding model {self.embedding_model} returned no "
                        "embeddings while probing the vector dimension. This "
                        "usually means the model name is wrong or unavailable for "
                        "your provider/API key (check the warnings above for the "
                        "underlying error). Fix the embedding model, or pass an "
                        "explicit `vector_dim=...` to skip the probe."
                    )
                self.vector_dim = len(embeddings[0])
            else:
                self.vector_dim = vector_dim

        if metric not in METRICS:
            raise ValueError(f"`metric` parameter should be one of {METRICS}")
        self.metric = metric
        self.vss_key = vss_key
        self.name = name

        self._db = lancedb.connect(self.uri)

        if wipe_on_start:
            self.wipe_database()

        self.data_models: Dict[str, Any] = {}
        if data_models:
            for dm in data_models:
                title = dm.get_schema().get("title")
                if title is None:
                    raise ValueError(
                        "Each registered data model must carry a schema `title`; "
                        "got a schema with no title."
                    )
                self.data_models[table_identifier(title)] = dm
                self._maybe_create_table(dm)
        else:
            for dm in self.get_symbolic_data_models():
                self.data_models[dm.get_schema().get("title")] = dm

    # -- schema mapping --------------------------------------------------------

    @staticmethod
    def _resolve_ref(spec: dict, defs: dict) -> dict:
        if isinstance(spec, dict) and "$ref" in spec:
            key = spec["$ref"].rsplit("/", 1)[-1]
            return dict(defs.get(key, {}))
        return spec

    def _get_id_key(self, schema: dict) -> str:
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not props:
            raise ValueError("Cannot determine primary key: schema has no `properties`.")
        return _column_identifier(next(iter(props.keys())))

    def _json_schema_to_arrow(self, json_schema: dict):
        """Build a PyArrow schema + the set of JSON-encoded column names."""
        props = json_schema.get("properties", {})
        defs = json_schema.get("$defs", {})
        fields = []
        json_cols = set()
        for raw_name, raw_spec in props.items():
            name = _column_identifier(raw_name)
            spec = self._resolve_ref(raw_spec, defs)
            t = spec.get("type")
            if name == self.vss_key:
                if self.vector_dim:
                    fields.append(pa.field(name, pa.list_(pa.float32(), self.vector_dim)))
                else:
                    fields.append(pa.field(name, pa.list_(pa.float32())))
            elif t == "string":
                fields.append(pa.field(name, pa.string()))
            elif t == "number":
                fields.append(pa.field(name, pa.float64()))
            elif t == "integer":
                fields.append(pa.field(name, pa.int64()))
            elif t == "boolean":
                fields.append(pa.field(name, pa.bool_()))
            elif t == "array":
                items = self._resolve_ref(spec.get("items", {}), defs)
                it = items.get("type")
                if it == "number":
                    fields.append(pa.field(name, pa.list_(pa.float64())))
                elif it == "integer":
                    fields.append(pa.field(name, pa.list_(pa.int64())))
                elif it == "boolean":
                    fields.append(pa.field(name, pa.list_(pa.bool_())))
                elif it == "object":
                    fields.append(pa.field(name, pa.string()))
                    json_cols.add(name)
                else:
                    fields.append(pa.field(name, pa.list_(pa.string())))
            elif t == "object":
                fields.append(pa.field(name, pa.string()))
                json_cols.add(name)
            else:
                # enum / unknown -> string
                fields.append(pa.field(name, pa.string()))
        metadata = {_SCHEMA_META_KEY: orjson.dumps(json_schema)}
        return pa.schema(fields, metadata=metadata), json_cols

    def _table_json_schema(self, table_name: str, remove_embedding: bool = True) -> dict:
        """Reflect a table back to a JSON schema (from stashed metadata)."""
        table = table_identifier(table_name)
        tbl = self._db.open_table(table)
        arrow_schema = tbl.schema
        meta = arrow_schema.metadata or {}
        stored = meta.get(_SCHEMA_META_KEY)
        if stored:
            schema = orjson.loads(stored)
            if remove_embedding:
                schema = dict(schema)
                schema["properties"] = {
                    k: v
                    for k, v in schema.get("properties", {}).items()
                    if _column_identifier(k) != self.vss_key
                }
                schema["required"] = list(schema["properties"].keys())
            return schema
        return self._arrow_to_json_schema(table, arrow_schema, remove_embedding)

    def _arrow_to_json_schema(self, table, arrow_schema, remove_embedding) -> dict:
        props = {}
        for field in arrow_schema:
            name = field.name
            if name == self.vss_key and remove_embedding:
                continue
            pt = field.type
            if pa.types.is_string(pt) or pa.types.is_large_string(pt):
                props[name] = {"title": name.title(), "type": "string"}
            elif pa.types.is_floating(pt):
                props[name] = {"title": name.title(), "type": "number"}
            elif pa.types.is_integer(pt):
                props[name] = {"title": name.title(), "type": "integer"}
            elif pa.types.is_boolean(pt):
                props[name] = {"title": name.title(), "type": "boolean"}
            elif pa.types.is_list(pt) or pa.types.is_fixed_size_list(pt):
                vt = pt.value_type
                if pa.types.is_floating(vt):
                    item = {"type": "number"}
                elif pa.types.is_integer(vt):
                    item = {"type": "integer"}
                elif pa.types.is_boolean(vt):
                    item = {"type": "boolean"}
                else:
                    item = {"type": "string"}
                props[name] = {"title": name.title(), "type": "array", "items": item}
            else:
                props[name] = {"title": name.title(), "type": "string"}
        return {
            "title": table,
            "type": "object",
            "additionalProperties": False,
            "required": list(props.keys()),
            "properties": props,
        }

    def _json_columns(self, json_schema: dict) -> set:
        props = json_schema.get("properties", {})
        defs = json_schema.get("$defs", {})
        cols = set()
        for raw_name, raw_spec in props.items():
            spec = self._resolve_ref(raw_spec, defs)
            t = spec.get("type")
            if t == "object":
                cols.add(_column_identifier(raw_name))
            elif t == "array":
                items = self._resolve_ref(spec.get("items", {}), defs)
                if items.get("type") == "object":
                    cols.add(_column_identifier(raw_name))
        return cols

    def _string_columns(self, json_schema: dict, *, exclude_pk=False) -> List[str]:
        id_key = self._get_id_key(json_schema)
        out = []
        defs = json_schema.get("$defs", {})
        for raw_name, raw_spec in json_schema.get("properties", {}).items():
            name = _column_identifier(raw_name)
            if exclude_pk and name == id_key:
                continue
            spec = self._resolve_ref(raw_spec, defs)
            if (
                spec.get("type") == "string"
                and spec.get("format") not in _DATE_LIKE_FORMATS
            ):
                out.append(name)
        return out

    # -- table lifecycle -------------------------------------------------------

    def wipe_database(self):
        for name in self._db.table_names():
            self._db.drop_table(name, ignore_missing=True)

    def get_symbolic_data_models(self) -> List[SymbolicDataModel]:
        models = []
        for table_name in self._db.table_names():
            schema = self._table_json_schema(table_name, remove_embedding=False)
            models.append(SymbolicDataModel(schema=schema))
        return models

    def _maybe_create_table(self, data_model):
        json_schema = data_model.get_schema()
        table = table_identifier(json_schema["title"])
        if table in self._db.table_names():
            return
        arrow_schema, _ = self._json_schema_to_arrow(json_schema)
        self._db.create_table(table, schema=arrow_schema)

    def _row_for_storage(self, json_data: dict, json_cols: set, arrow_schema) -> dict:
        row = {}
        for field in arrow_schema:
            name = field.name
            value = json_data.get(name)
            if name in json_cols and value is not None and not isinstance(value, str):
                value = orjson.dumps(value).decode()
            row[name] = value
        return row

    def _decode_row(self, row: dict, json_cols: set, remove_embedding: bool) -> dict:
        out = {}
        for k, v in row.items():
            if remove_embedding and k == self.vss_key:
                continue
            if k in json_cols and isinstance(v, str):
                try:
                    v = orjson.loads(v)
                except orjson.JSONDecodeError:
                    pass
            out[k] = v
        return out

    # -- writes ----------------------------------------------------------------

    async def update(
        self,
        data_model_or_data_models: Union[List[JsonDataModel], JsonDataModel],
    ) -> Union[Any, List[Any]]:
        return_single = not isinstance(data_model_or_data_models, list)
        data_models = (
            [data_model_or_data_models] if return_single else data_model_or_data_models
        )

        ids: List[Any] = []
        buckets: Dict[str, Dict[str, Any]] = {}

        for data_model in data_models:
            if not isinstance(data_model, JsonDataModel):
                data_model = data_model.to_json_data_model()
            schema = data_model.get_schema()
            table = table_identifier(schema["title"])
            json_data = {
                _column_identifier(k): v for k, v in data_model.get_json().items()
            }
            id_key = self._get_id_key(schema)
            id_val = json_data.get(id_key)
            if id_val is None:
                raise ValueError(f"Primary key '{id_key}' is required but not provided")

            if table not in buckets:
                self._maybe_create_table(data_model)
                arrow_schema, json_cols = self._json_schema_to_arrow(schema)
                buckets[table] = {
                    "id_key": id_key,
                    "rows": [],
                    "json_cols": json_cols,
                    "arrow_schema": arrow_schema,
                }
            bucket = buckets[table]
            bucket["rows"].append(
                self._row_for_storage(
                    json_data, bucket["json_cols"], bucket["arrow_schema"]
                )
            )
            ids.append(id_val)

        for table, bucket in buckets.items():
            tbl = self._db.open_table(table)
            (
                tbl.merge_insert(bucket["id_key"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(bucket["rows"])
            )
            schema = self._table_json_schema(table, remove_embedding=False)
            # LanceDB FTS indexes are one-per-field, so build a separate
            # inverted index per text column; ``fulltext_search`` then queries
            # across all of them via ``query_type="fts"``.
            for col in self._string_columns(schema, exclude_pk=True):
                try:
                    tbl.create_fts_index(col, replace=True)
                except Exception as e:
                    warnings.warn(
                        f"FTS index rebuild failed for '{table}.{col}'; "
                        f"fulltext_search results may be stale. ({e})"
                    )

        return ids[0] if return_single else ids

    async def get(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
        remove_embedding: bool = True,
    ):
        return_single = not isinstance(id_or_ids, list)
        ids = [id_or_ids] if return_single else list(id_or_ids)
        if not ids:
            return None if return_single else []

        table = table_identifier(table_name)
        schema = self._table_json_schema(table, remove_embedding=remove_embedding)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        id_key = self._get_id_key(full_schema)
        json_cols = self._json_columns(full_schema)

        tbl = self._db.open_table(table)
        predicate = f"{id_key} IN ({', '.join(_sql_literal(i) for i in ids)})"
        try:
            rows = tbl.search().where(predicate).to_arrow().to_pylist()
        except Exception as e:
            warnings.warn(f"get(): read from '{table}' failed. ({e})")
            return None if return_single else [None] * len(ids)

        results: List[Optional[JsonDataModel]] = [None] * len(ids)
        rows_by_id = {row[id_key]: row for row in rows}
        for idx, id_val in enumerate(ids):
            row = rows_by_id.get(id_val)
            if row is None:
                continue
            json_data = self._decode_row(row, json_cols, remove_embedding)
            results[idx] = JsonDataModel(
                json=json_data, schema=schema, name=str(json_data.get(id_key))
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
        table = table_identifier(table_name)
        if table not in self._db.table_names():
            warnings.warn(f"Failed to read table '{table}': not found")
            return []
        schema = self._table_json_schema(table, remove_embedding=remove_embedding)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        id_key = self._get_id_key(full_schema)
        json_cols = self._json_columns(full_schema)

        tbl = self._db.open_table(table)
        try:
            rows = tbl.search().limit(limit).offset(offset).to_arrow().to_pylist()
        except Exception as e:
            warnings.warn(f"Failed to read table '{table}': {e}")
            return []

        results = []
        for row in rows:
            json_data = self._decode_row(row, json_cols, remove_embedding)
            results.append(
                JsonDataModel(
                    json=json_data, schema=schema, name=str(json_data.get(id_key))
                )
            )
        return results

    async def delete(
        self,
        id_or_ids: Union[Any, List[Any]],
        *,
        table_name: str,
    ) -> int:
        ids = [id_or_ids] if not isinstance(id_or_ids, list) else list(id_or_ids)
        if not ids:
            return 0
        table = table_identifier(table_name)
        if table not in self._db.table_names():
            return 0
        tbl = self._db.open_table(table)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        id_key = self._get_id_key(full_schema)
        before = tbl.count_rows()
        predicate = f"{id_key} IN ({', '.join(_sql_literal(i) for i in ids)})"
        tbl.delete(predicate)
        return before - tbl.count_rows()

    async def drop_table(self, table_name: str) -> bool:
        table = table_identifier(table_name)
        if table not in self._db.table_names():
            return False
        self._db.drop_table(table)
        self.data_models.pop(table, None)
        return True

    async def rename(
        self,
        source: Any,
        *,
        table_name: Optional[str] = None,
        table_description: Optional[str] = None,
    ):
        old = table_identifier(
            source.get_schema()["title"] if hasattr(source, "get_schema") else source
        )
        new = table_identifier(table_name) if table_name else old
        if new != old:
            self._db.rename_table(old, new)
        schema = self._table_json_schema(new, remove_embedding=False)
        schema["title"] = new
        if table_description is not None:
            schema["description"] = table_description
        return SymbolicDataModel(schema=schema)

    # -- bulk file loaders -----------------------------------------------------

    async def _from_arrow(self, arrow_table, table_name, table_description):
        table = table_identifier(table_name)
        json_schema = self._arrow_to_json_schema(table, arrow_table.schema, False)
        if table_description is not None:
            json_schema["description"] = table_description
        metadata = {_SCHEMA_META_KEY: orjson.dumps(json_schema)}
        arrow_table = arrow_table.replace_schema_metadata(metadata)
        self._db.drop_table(table, ignore_missing=True)
        self._db.create_table(table, data=arrow_table)
        return SymbolicDataModel(schema=json_schema)

    async def from_csv(self, path, *, table_name=None, table_description=None, **kwargs):
        import pyarrow.csv as pa_csv

        return await self._from_arrow(
            pa_csv.read_csv(path), table_name or _stem(path), table_description
        )

    async def from_parquet(
        self, path, *, table_name=None, table_description=None, **kwargs
    ):
        import pyarrow.parquet as pa_parquet

        return await self._from_arrow(
            pa_parquet.read_table(path), table_name or _stem(path), table_description
        )

    async def from_json(self, path, *, table_name=None, table_description=None, **kwargs):
        with open(path, "rb") as f:
            records = orjson.loads(f.read())
        return await self._from_arrow(
            pa.Table.from_pylist(records), table_name or _stem(path), table_description
        )

    async def from_jsonl(
        self, path, *, table_name=None, table_description=None, **kwargs
    ):
        records = []
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(orjson.loads(line))
        return await self._from_arrow(
            pa.Table.from_pylist(records), table_name or _stem(path), table_description
        )

    # -- sql (delegated to DuckDB scanning the Lance datasets) -----------------

    async def sql(
        self,
        sql: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        **kwargs,
    ):
        """Run SQL over the LanceDB tables via DuckDB.

        LanceDB has no SQL engine; DuckDB scans each table (registered as an
        Arrow view under its table name) so arbitrary read-only ``SELECT`` /
        joins / aggregates work.
        """
        import duckdb

        con = duckdb.connect(":memory:")
        try:
            for table in self._db.table_names():
                con.register(table, self._db.open_table(table).to_arrow())
            cursor = con.execute(sql, params) if params else con.execute(sql)
            arrow_table = cursor.arrow().read_all()
        finally:
            con.close()
        return format_search_results(arrow_table, output_format)

    # -- search ----------------------------------------------------------------

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
        if not text_or_texts:
            return format_search_results([], output_format)
        if not self.embedding_model:
            raise ValueError(
                "similarity_search requires an embedding model on the adapter."
            )
        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts
        table = table_identifier(table_name)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        id_key = self._get_id_key(full_schema)
        json_cols = self._json_columns(full_schema)

        embeddings = await self.embedding_model(EmbeddingRequest(texts=texts))
        vectors = embeddings.get("embeddings")

        offset = _SCORE_OFFSET[self.metric]
        tbl = self._db.open_table(table)
        merged: Dict[Any, Dict[str, Any]] = {}
        for vector in vectors:
            q = tbl.search(vector, vector_column_name=self.vss_key).metric(
                _NATIVE_METRIC[self.metric]
            )
            rows = q.limit(k).to_arrow().to_pylist()
            for row in rows:
                dist = row.pop("_distance", None)
                score = dist + offset if dist is not None else None
                if threshold is not None and score is not None and not score < threshold:
                    continue
                row["score"] = score
                row = self._decode_row(row, json_cols, remove_embedding=False)
                uid = row[id_key]
                prev = merged.get(uid)
                if prev is None or row["score"] < prev["score"]:
                    merged[uid] = row
        ranked = sorted(merged.values(), key=lambda r: r["score"])[:k]
        return format_search_results(ranked, output_format)

    async def fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        table_name: str,
        k: int = 10,
        threshold: Optional[float] = None,
        output_format: str = "json",
        **kwargs,
    ):
        if not text_or_texts:
            return format_search_results([], output_format)
        texts = [text_or_texts] if not isinstance(text_or_texts, list) else text_or_texts
        table = table_identifier(table_name)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        id_key = self._get_id_key(full_schema)
        json_cols = self._json_columns(full_schema)
        if not self._string_columns(full_schema, exclude_pk=True):
            warnings.warn(f"Skipping FTS search for {table}: no text columns to index.")
            return format_search_results([], output_format)

        tbl = self._db.open_table(table)
        merged: Dict[Any, Dict[str, Any]] = {}
        for text in texts:
            try:
                rows = tbl.search(text, query_type="fts").limit(k).to_arrow().to_pylist()
            except Exception as e:
                raise RuntimeError(f"FTS query failed for table '{table}': {e}")
            for row in rows:
                row["score"] = row.pop("_score", None)
                row = self._decode_row(row, json_cols, remove_embedding=False)
                uid = row[id_key]
                prev = merged.get(uid)
                if prev is None or (row["score"] or 0) > (prev["score"] or 0):
                    merged[uid] = row
        ranked = sorted(merged.values(), key=lambda r: r["score"] or 0, reverse=True)[:k]
        # Rescale raw Tantivy scores to [0, 1] so they're comparable with the
        # DuckDB adapter; ``threshold`` filters on the same normalized scale.
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
        if not pattern:
            return format_search_results([], output_format)
        table = table_identifier(table_name)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        json_cols = self._json_columns(full_schema)
        string_cols = self._string_columns(full_schema, exclude_pk=False)
        if fields is not None:
            requested = {_column_identifier(f) for f in fields}
            cols = [c for c in string_cols if c in requested]
        else:
            cols = string_cols
        if not cols:
            warnings.warn(
                f"Skipping regex search for {table}: no matching string fields."
            )
            return format_search_results([], output_format)

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise RuntimeError(f"Invalid regex pattern for table '{table}': {e}")

        tbl = self._db.open_table(table)
        out = []
        for row in tbl.to_arrow().to_pylist():
            if any(isinstance(row.get(c), str) and compiled.search(row[c]) for c in cols):
                out.append(self._decode_row(row, json_cols, remove_embedding=False))
                if len(out) >= k:
                    break
        return format_search_results(out, output_format)

    async def hybrid_search(self, *args, **kwargs):
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
        output_format: str = "json",
        **kwargs,
    ):
        if not text_or_texts:
            return format_search_results([], output_format)
        table = table_identifier(table_name)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        id_key = self._get_id_key(full_schema)

        if not self.embedding_model:
            fts_rows = await self.fulltext_search(
                keywords if keywords is not None else text_or_texts,
                table_name=table,
                k=k,
                threshold=fulltext_threshold,
                output_format="json",
            )
            for row in fts_rows:
                row.setdefault("rrf_score", row.get("score", 0.0))
            return format_search_results(fts_rows, output_format)

        queries, keyword_queries = align_keywords(text_or_texts, keywords)

        final: Dict[Any, Dict[str, Any]] = {}
        for query_text, keyword_text in zip(queries, keyword_queries):
            try:
                fts_results = await self.fulltext_search(
                    keyword_text,
                    table_name=table,
                    k=k * 5,
                    threshold=fulltext_threshold,
                    output_format="json",
                )
            except Exception:
                fts_results = []
            try:
                vss_results = await self.similarity_search(
                    query_text,
                    table_name=table,
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
            combined: Dict[Any, Dict[str, Any]] = {}
            for row in fts_results + vss_results:
                uid = row[id_key]
                combined.setdefault(uid, {}).update(row)
            for uid in set(fts_rank) | set(vss_rank):
                score = 0.0
                if uid in fts_rank:
                    score += 1.0 / (k_rank + fts_rank[uid])
                if uid in vss_rank:
                    score += 1.0 / (k_rank + vss_rank[uid])
                combined[uid]["score"] = score
            for uid, row in combined.items():
                if uid not in final or row["score"] > final[uid]["score"]:
                    final[uid] = row

        ranked = sorted(final.values(), key=lambda r: (-r["score"], str(r.get(id_key))))[
            :k
        ]
        return format_search_results(ranked, output_format)

    async def hybrid_regex_search(
        self,
        text_or_texts: Union[str, List[str]],
        *,
        pattern_or_patterns: Union[str, List[str], None] = None,
        table_name: str,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        ef_search: Optional[int] = None,
        output_format: str = "json",
        **kwargs,
    ):
        if not text_or_texts:
            return format_search_results([], output_format)
        table = table_identifier(table_name)
        full_schema = self._table_json_schema(table, remove_embedding=False)
        id_key = self._get_id_key(full_schema)

        queries = (
            [text_or_texts] if isinstance(text_or_texts, str) else list(text_or_texts)
        )
        if pattern_or_patterns is None:
            patterns: List[Optional[str]] = [None] * len(queries)
        elif isinstance(pattern_or_patterns, str):
            patterns = [pattern_or_patterns] * len(queries)
        else:
            patterns = list(pattern_or_patterns)
            if len(patterns) != len(queries):
                raise ValueError("`pattern_or_patterns` must align with `text_or_texts`.")

        final: Dict[Any, Dict[str, Any]] = {}
        for query_text, pattern in zip(queries, patterns):
            try:
                vss_results = await self.similarity_search(
                    query_text,
                    table_name=table,
                    k=k * 5,
                    threshold=similarity_threshold,
                    ef_search=ef_search,
                    output_format="json",
                )
            except Exception:
                vss_results = []
            regex_results = []
            if pattern is not None:
                try:
                    regex_results = await self.regex_search(
                        pattern,
                        table_name=table,
                        fields=fields,
                        case_sensitive=case_sensitive,
                        k=k * 5,
                        output_format="json",
                    )
                except Exception:
                    regex_results = []
            if not vss_results and not regex_results:
                continue
            vss_rank = {r[id_key]: i + 1 for i, r in enumerate(vss_results)}
            rgx_rank = {r[id_key]: i + 1 for i, r in enumerate(regex_results)}
            combined: Dict[Any, Dict[str, Any]] = {}
            for row in vss_results + regex_results:
                uid = row[id_key]
                combined.setdefault(uid, {}).update(row)
            for uid in set(vss_rank) | set(rgx_rank):
                score = 0.0
                if uid in vss_rank:
                    score += 1.0 / (k_rank + vss_rank[uid])
                if uid in rgx_rank:
                    score += 1.0 / (k_rank + rgx_rank[uid])
                combined[uid]["score"] = score
            for uid, row in combined.items():
                if uid not in final or row["score"] > final[uid]["score"]:
                    final[uid] = row

        ranked = sorted(final.values(), key=lambda r: (-r["score"], str(r.get(id_key))))[
            :k
        ]
        return format_search_results(ranked, output_format)

    def __repr__(self):
        return f"<LanceDBAdapter uri={self.uri}>"
