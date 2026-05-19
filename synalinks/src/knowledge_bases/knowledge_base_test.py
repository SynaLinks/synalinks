# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules.embedding_models import EmbeddingModel


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The content of the document")


class Chunk(DataModel):
    id: str = Field(description="The chunk id")
    text: str = Field(description="The content of the chunk")
    document_id: str = Field(description="The parent document id")


class KnowledgeBaseTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_knowledge_base(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document, Chunk],
            metric="cosine",
            wipe_on_start=False,
        )

        result = await knowledge_base.query("SELECT 1 as value")
        self.assertEqual(result, [{"value": 1}])

    async def test_knowledge_base_crud(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
            wipe_on_start=False,
        )

        # Insert a document
        doc = Document(id="doc1", text="Hello World")
        result = await knowledge_base.update(JsonDataModel(data_model=doc))
        self.assertEqual(result, "doc1")

        # Retrieve the document
        retrieved = await knowledge_base.get("doc1", table_name="Document")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get_json()["text"], "Hello World")

    async def test_knowledge_base_fulltext_search(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
            wipe_on_start=False,
        )

        # Insert documents
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await knowledge_base.update(docs)

        # Full-text search
        results = await knowledge_base.fulltext_search(
            "quick", table_name="Document", k=10
        )
        self.assertGreater(len(results), 0)

    async def test_search_output_format_csv(self):
        # CSV output is the agent-friendly format: one header row, one
        # data row per match, separator-delimited. Built via
        # pyarrow.csv.write_csv straight off the Arrow result table.
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        await kb.update(
            [
                JsonDataModel(data_model=Document(id="doc1", text="quick brown fox")),
                JsonDataModel(data_model=Document(id="doc2", text="quick rabbit runs")),
            ]
        )

        csv_out = await kb.fulltext_search(
            "quick", table_name="Document", k=10, output_format="csv"
        )

        self.assertIsInstance(csv_out, str)
        # Header followed by data lines.
        lines = csv_out.strip().splitlines()
        self.assertGreaterEqual(len(lines), 2)
        # The header includes the table columns plus the score column
        # the search added.
        header = lines[0]
        self.assertIn("id", header)
        self.assertIn("text", header)
        self.assertIn("score", header)
        # The data lines mention the seeded ids.
        body = "\n".join(lines[1:])
        self.assertIn("doc1", body)
        self.assertIn("doc2", body)

    async def test_search_output_format_json_is_python_list(self):
        # ``"json"`` returns JSON-shaped Python data — a list of
        # dicts, not a serialized string. (Callers serialize via
        # orjson/json themselves if they need bytes on the wire.)
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        await kb.update(
            [JsonDataModel(data_model=Document(id="doc1", text="quick brown fox"))]
        )

        default = await kb.fulltext_search("quick", table_name="Document", k=10)
        explicit = await kb.fulltext_search(
            "quick",
            table_name="Document",
            k=10,
            output_format="json",
        )

        self.assertIsInstance(default, list)
        self.assertIsInstance(explicit, list)
        self.assertEqual(default, explicit)

    async def test_query_output_format_csv(self):
        # `query()` (the raw SQL escape hatch) honours the same
        # ``output_format`` contract as the search methods.
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        await kb.update(
            [
                JsonDataModel(data_model=Document(id="d1", text="alpha")),
                JsonDataModel(data_model=Document(id="d2", text="beta")),
            ]
        )

        records = await kb.query("SELECT id, text FROM Document ORDER BY id")
        self.assertEqual(
            records,
            [
                {"id": "d1", "text": "alpha"},
                {"id": "d2", "text": "beta"},
            ],
        )

        csv_out = await kb.query(
            "SELECT id, text FROM Document ORDER BY id",
            output_format="csv",
        )
        self.assertIsInstance(csv_out, str)
        lines = csv_out.strip().splitlines()
        # pyarrow quotes header cells; values that don't need quoting
        # come back unquoted.
        self.assertIn("id", lines[0])
        self.assertIn("text", lines[0])
        body = "\n".join(lines[1:])
        self.assertIn("d1", body)
        self.assertIn("alpha", body)
        self.assertIn("d2", body)
        self.assertIn("beta", body)

    async def test_search_output_format_empty_query_each_format(self):
        # The empty-query short-circuit respects output_format:
        # ``"json"`` → ``[]``, ``"csv"`` → ``""``.
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])

        default = await kb.fulltext_search("", table_name="Document")
        self.assertEqual(default, [])

        csv_out = await kb.fulltext_search("", table_name="Document", output_format="csv")
        self.assertEqual(csv_out, "")

        json_out = await kb.fulltext_search(
            "", table_name="Document", output_format="json"
        )
        self.assertEqual(json_out, [])

    def test_knowledge_base_encryption_key_not_serialized(self):
        # The encryption key is a secret — it must never appear in
        # `get_config()` output, in `vars(kb)`, or in `repr(kb)`. The
        # only place it should live is inside the adapter's private
        # `_encryption_key` attribute.
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document, Chunk],
            wipe_on_start=False,
            encryption_key="s3cret-passphrase",
        )
        try:
            config = knowledge_base.get_config()
            self.assertNotIn("encryption_key", config)
            # And — defensively — the secret string itself must not
            # appear anywhere in the serialised representation.
            self.assertNotIn("s3cret-passphrase", str(config))
            self.assertNotIn("s3cret-passphrase", repr(knowledge_base))
        finally:
            knowledge_base.adapter.close()

    async def test_update_from_csv_dataset_streams_batch_by_batch(self):
        # End-to-end: ingest a CSV file via CSVDataset into the KB.
        # Verify every row landed AND that the adapter saw one call per
        # batch (i.e. we didn't materialize the whole dataset into a
        # single update — the streaming guarantee is what makes this
        # feature useful for files bigger than RAM).
        from synalinks.src.datasets.csv_dataset import CSVDataset

        csv_path = os.path.join(self.temp_dir, "docs.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text\n")
            for i in range(7):
                f.write(f"doc{i},content {i}\n")

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])

        ds = CSVDataset(
            path=csv_path,
            input_data_model=Document,
            input_template='{"id": {{ id | tojson }}, "text": {{ text | tojson }}}',
            batch_size=3,  # 7 rows / 3 → 3 batches: [3, 3, 1]
        )

        original_update = kb.adapter.update
        call_count = {"n": 0, "batch_sizes": []}

        async def counted_update(arg):
            call_count["n"] += 1
            if isinstance(arg, list):
                call_count["batch_sizes"].append(len(arg))
            return await original_update(arg)

        kb.adapter.update = counted_update
        try:
            ids = await kb.update(ds)
        finally:
            kb.adapter.update = original_update

        # All seven rows ingested, ids returned flat in source order.
        self.assertEqual(ids, [f"doc{i}" for i in range(7)])
        # One adapter call per dataset batch — proves the streaming.
        self.assertEqual(call_count["n"], 3)
        self.assertEqual(call_count["batch_sizes"], [3, 3, 1])

        # And the rows are actually queryable.
        first = await kb.get("doc0", table_name="Document")
        self.assertEqual(first.get_json()["text"], "content 0")
        last = await kb.get("doc6", table_name="Document")
        self.assertEqual(last.get_json()["text"], "content 6")

    async def test_update_from_empty_dataset_returns_empty_list(self):
        # Empty source (header-only CSV) is a real "the producer
        # didn't write anything" failure mode. KB.update should
        # return [] without calling the adapter at all — there's
        # nothing to insert and no transaction to open.
        from synalinks.src.datasets.csv_dataset import CSVDataset

        csv_path = os.path.join(self.temp_dir, "empty.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text\n")

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])

        ds = CSVDataset(
            path=csv_path,
            input_data_model=Document,
            input_template='{"id": {{ id | tojson }}, "text": {{ text | tojson }}}',
            batch_size=4,
        )

        adapter_calls = {"n": 0}
        original_update = kb.adapter.update

        async def counted_update(arg):
            adapter_calls["n"] += 1
            return await original_update(arg)

        kb.adapter.update = counted_update
        try:
            result = await kb.update(ds)
        finally:
            kb.adapter.update = original_update

        self.assertEqual(result, [])
        self.assertEqual(adapter_calls["n"], 0)

    async def test_update_rejects_dataset_with_output_template(self):
        # The KB stores records, not (input, target) pairs. A dataset
        # with an output_template was configured for training, not for
        # ingestion — reject it eagerly so the user gets a clear
        # message instead of silently dropping the targets.
        from synalinks.src.datasets.csv_dataset import CSVDataset

        csv_path = os.path.join(self.temp_dir, "labeled.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text,label\n")
            f.write("d1,hello,greeting\n")

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])

        class Label(DataModel):
            label: str

        labeled = CSVDataset(
            path=csv_path,
            input_data_model=Document,
            input_template='{"id": {{ id | tojson }}, "text": {{ text | tojson }}}',
            output_data_model=Label,
            output_template='{"label": {{ label | tojson }}}',
            batch_size=1,
        )

        with self.assertRaises(ValueError) as cm:
            await kb.update(labeled)
        self.assertIn("inputs-only", str(cm.exception))

    async def test_update_preserves_existing_paths_for_non_dataset_inputs(self):
        # The Dataset branch is additive — single-instance and list
        # inputs still go straight to adapter.update unchanged. Guards
        # against regressions in the pre-existing API shape.
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])

        single_id = await kb.update(
            JsonDataModel(data_model=Document(id="x1", text="hi"))
        )
        self.assertEqual(single_id, "x1")

        many_ids = await kb.update(
            [
                JsonDataModel(data_model=Document(id="x2", text="a")),
                JsonDataModel(data_model=Document(id="x3", text="b")),
            ]
        )
        self.assertEqual(many_ids, ["x2", "x3"])

    async def test_update_from_dataset_with_raw_datamodel_instances(self):
        # CSVDataset yields raw DataModel instances (not JsonDataModels)
        # when configured with `input_data_model`. The adapter's update
        # accepts both forms, so this round-trips without the KB needing
        # to coerce anything in between.
        from synalinks.src.datasets.csv_dataset import CSVDataset

        csv_path = os.path.join(self.temp_dir, "raw.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text\n")
            f.write("d1,alpha\n")
            f.write("d2,beta\n")

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])

        ds = CSVDataset(
            path=csv_path,
            input_data_model=Document,
            input_template='{"id": {{ id | tojson }}, "text": {{ text | tojson }}}',
            batch_size=2,
        )

        ids = await kb.update(ds)
        self.assertEqual(ids, ["d1", "d2"])

        retrieved = await kb.get(["d1", "d2"], table_name="Document")
        self.assertEqual([r.get_json()["text"] for r in retrieved], ["alpha", "beta"])

    async def test_update_from_dataset_verbose_modes(self):
        # `verbose="auto"` (the default) resolves to 1 when a Dataset
        # is passed — same as the trainer's `fit()` convention — so an
        # un-annotated `kb.update(ds)` call still shows a progress bar.
        # `verbose=0` must stay silent. The unit_name "batch" is the
        # safest substring to assert on because it appears regardless
        # of ANSI styling; CI runs without a TTY so the bar falls
        # through to its non-dynamic branch but still prints.
        import io
        import sys

        from synalinks.src.datasets.csv_dataset import CSVDataset

        csv_path = os.path.join(self.temp_dir, "verbose.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text\n")
            for i in range(4):
                f.write(f"v{i},row{i}\n")

        def make_dataset():
            return CSVDataset(
                path=csv_path,
                input_data_model=Document,
                input_template=('{"id": {{ id | tojson }}, "text": {{ text | tojson }}}'),
                batch_size=2,
            )

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])

        # Default ("auto") — should emit progbar for a Dataset.
        buf_auto = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf_auto
        try:
            ids_auto = await kb.update(make_dataset())
        finally:
            sys.stdout = old_stdout
        self.assertEqual(ids_auto, ["v0", "v1", "v2", "v3"])
        out_auto = buf_auto.getvalue()
        self.assertIn("batch", out_auto)
        self.assertIn("rows", out_auto)

        # Explicit `verbose=0` — must stay silent even for a Dataset.
        buf_silent = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf_silent
        try:
            ids_silent = await kb.update(make_dataset(), verbose=0)
        finally:
            sys.stdout = old_stdout
        self.assertEqual(ids_silent, ["v0", "v1", "v2", "v3"])
        self.assertEqual(buf_silent.getvalue(), "")

    async def test_from_csv_round_trips_rows_into_kb(self):
        # End-to-end: a CSV file goes in, all rows are queryable.
        # The fast path skips Pydantic / Jinja, so this is also the
        # one place we verify the native reader's output ends up
        # shaped the same as it would via update().
        import csv

        csv_path = os.path.join(self.temp_dir, "docs.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            for i in range(50):
                w.writerow([f"doc{i:03d}", f"row {i} content"])

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            model = await kb.from_csv(csv_path, table_name="Document")
            # The fast path returns the SymbolicDataModel for the
            # loaded table — its title matches the requested name.
            self.assertEqual(model.get_schema()["title"], "Document")

            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 50)

            # All rows queryable through the returned model.
            first = await kb.get("doc000", table_name=model.get_schema()["title"])
            self.assertEqual(first.get_json()["text"], "row 0 content")
            last = await kb.get("doc049", table_name=model.get_schema()["title"])
            self.assertEqual(last.get_json()["text"], "row 49 content")
        finally:
            kb.adapter.close()

    async def test_from_csv_does_upsert_not_duplicate(self):
        # The conflict clause matters: re-running the same load
        # against the same KB must overwrite the existing rows, not
        # raise on PRIMARY KEY collision (which a plain INSERT would).
        import csv

        csv_path = os.path.join(self.temp_dir, "docs.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["d1", "version 1"])
            w.writerow(["d2", "v1"])

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            model = await kb.from_csv(csv_path, table_name="Document")

            # Overwrite the same ids with new content.
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "text"])
                w.writerow(["d1", "version 2"])
                w.writerow(["d3", "v1"])  # new row, mixed with the overwrite

            model = await kb.from_csv(csv_path, table_name="Document")
            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 3)  # d1 (updated) + d2 (kept) + d3 (new)

            d1 = await kb.get("d1", table_name=model.get_schema()["title"])
            self.assertEqual(d1.get_json()["text"], "version 2")
            d2 = await kb.get("d2", table_name=model.get_schema()["title"])
            self.assertEqual(d2.get_json()["text"], "v1")
            d3 = await kb.get("d3", table_name=model.get_schema()["title"])
            self.assertEqual(d3.get_json()["text"], "v1")
        finally:
            kb.adapter.close()

    async def test_from_csv_fts_index_rebuilt_so_search_works(self):
        # After from_csv returns, fulltext_search must find rows in
        # the loaded data. Pre-fix sloppiness would have left the FTS
        # index empty (or stale) and the search would return nothing.
        import csv

        csv_path = os.path.join(self.temp_dir, "docs.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["d1", "the quick brown fox"])
            w.writerow(["d2", "the lazy dog sleeps"])
            w.writerow(["d3", "another quick rabbit"])

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            model = await kb.from_csv(csv_path, table_name="Document")
            results = await kb.fulltext_search(
                "quick", table_name=model.get_schema()["title"], k=10
            )
            ids = {r["id"] for r in results}
            self.assertIn("d1", ids)
            self.assertIn("d3", ids)
            self.assertNotIn("d2", ids)
        finally:
            kb.adapter.close()

    async def test_from_csv_restores_sandbox_after_load(self):
        # The fast path tears the sandboxed connection down to let a
        # loose connection run read_csv (which needs external access),
        # then reopens the sandboxed one. Verify the sandbox is back
        # afterwards by trying to do exactly what the sandbox blocks —
        # a SELECT from read_csv via query() should fail because the
        # post-load persistent connection has enable_external_access
        # disabled again.
        import csv

        csv_path = os.path.join(self.temp_dir, "docs.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["d1", "hello"])

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.from_csv(csv_path, table_name="Document")

            # The post-load persistent connection must still refuse
            # read_csv via query(read_only=False) — the sandbox is
            # connection-level, so re-attaching means the new
            # connection was re-sandboxed in __init__'s path.
            import duckdb

            with self.assertRaises(duckdb.Error):
                await kb.query(
                    f"SELECT * FROM read_csv('{csv_path}', AUTO_DETECT=TRUE)",
                    read_only=False,
                )
        finally:
            kb.adapter.close()

    async def test_from_csv_raises_on_missing_file(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            import pytest

            with pytest.raises(FileNotFoundError):
                await kb.from_csv("/nonexistent/path/x.csv", table_name="Document")
        finally:
            kb.adapter.close()

    async def test_from_csv_infers_types_like_other_formats(self):
        # CSV bulk-load now relies on DuckDB's native type
        # auto-detection (no ``all_varchar`` override) so a column of
        # plain integers ends up as ``BIGINT``, a column of decimals
        # as ``DOUBLE``, and a column of text as ``VARCHAR`` — same
        # contract Parquet / JSON / JSONL already had.
        import csv

        csv_path = os.path.join(self.temp_dir, "typed.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "score", "label"])
            for i in range(5):
                w.writerow([i + 1, 0.5 * i, f"row {i}"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_csv(csv_path, table_name="Scores")

            schema = model.get_schema()
            self.assertEqual(schema["properties"]["row_id"]["type"], "integer")
            self.assertEqual(schema["properties"]["score"]["type"], "number")
            self.assertEqual(schema["properties"]["label"]["type"], "string")

            # Lookup uses the actual inferred type — pass an int.
            r = await kb.get(3, table_name=model.get_schema()["title"])
            self.assertIsNotNone(r)
            self.assertEqual(r.get_json()["label"], "row 2")
            self.assertEqual(r.get_json()["score"], 1.0)
        finally:
            kb.adapter.close()

    async def test_from_csv_preserves_leading_zeros(self):
        # DuckDB's CSV auto-detect is conservative about strings that
        # look numeric — values with leading zeros like ``"00123"``
        # are kept as ``VARCHAR`` rather than promoted to ``INTEGER``.
        # This means id columns formatted with leading zeros survive
        # the round-trip without losing them, even though we no
        # longer force ``all_varchar`` globally.
        import csv

        csv_path = os.path.join(self.temp_dir, "padded.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            for i in range(5):
                w.writerow([f"{i:04d}", f"row {i}"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_csv(csv_path, table_name="Padded")
            self.assertEqual(model.get_schema()["properties"]["id"]["type"], "string")
            r = await kb.get("0002", table_name=model.get_schema()["title"])
            self.assertIsNotNone(r)
            self.assertEqual(r.get_json()["label"], "row 2")
        finally:
            kb.adapter.close()

    async def test_from_parquet_round_trips_rows_into_kb(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = os.path.join(self.temp_dir, "docs.parquet")
        table = pa.table(
            {
                "id": [f"p{i:03d}" for i in range(50)],
                "text": [f"row {i} content" for i in range(50)],
            }
        )
        pq.write_table(table, path)

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            model = await kb.from_parquet(path, table_name="Document")
            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 50)

            first = await kb.get("p000", table_name=model.get_schema()["title"])
            self.assertEqual(first.get_json()["text"], "row 0 content")
            last = await kb.get("p049", table_name=model.get_schema()["title"])
            self.assertEqual(last.get_json()["text"], "row 49 content")
        finally:
            kb.adapter.close()

    async def test_from_parquet_upserts_existing_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path1 = os.path.join(self.temp_dir, "v1.parquet")
        pq.write_table(pa.table({"id": ["d1", "d2"], "text": ["one", "two"]}), path1)
        path2 = os.path.join(self.temp_dir, "v2.parquet")
        pq.write_table(
            pa.table({"id": ["d1", "d3"], "text": ["one-updated", "three"]}),
            path2,
        )

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.from_parquet(path1, table_name="Document")
            model = await kb.from_parquet(path2, table_name="Document")
            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 3)

            d1 = await kb.get("d1", table_name=model.get_schema()["title"])
            self.assertEqual(d1.get_json()["text"], "one-updated")
            d2 = await kb.get("d2", table_name=model.get_schema()["title"])
            self.assertEqual(d2.get_json()["text"], "two")
            d3 = await kb.get("d3", table_name=model.get_schema()["title"])
            self.assertEqual(d3.get_json()["text"], "three")
        finally:
            kb.adapter.close()

    async def test_from_parquet_raises_on_missing_file(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            import pytest

            with pytest.raises(FileNotFoundError):
                await kb.from_parquet(
                    "/nonexistent/path/x.parquet", table_name="Document"
                )
        finally:
            kb.adapter.close()

    async def test_from_json_round_trips_rows(self):
        # JSON array form: the fast path drives DuckDB's
        # read_json(format='array').
        import json as _json

        path = os.path.join(self.temp_dir, "docs.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(
                [{"id": f"doc{i:03d}", "text": f"row {i} content"} for i in range(50)],
                f,
            )

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            model = await kb.from_json(path, table_name="Document")
            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 50)
            first = await kb.get("doc000", table_name=model.get_schema()["title"])
            self.assertEqual(first.get_json()["text"], "row 0 content")
            last = await kb.get("doc049", table_name=model.get_schema()["title"])
            self.assertEqual(last.get_json()["text"], "row 49 content")
        finally:
            kb.adapter.close()

    async def test_from_json_does_upsert_not_duplicate(self):
        import json as _json

        path = os.path.join(self.temp_dir, "docs.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump([{"id": "d1", "text": "v1"}, {"id": "d2", "text": "v1"}], f)

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.from_json(path, table_name="Document")

            with open(path, "w", encoding="utf-8") as f:
                _json.dump([{"id": "d1", "text": "v2"}, {"id": "d3", "text": "v1"}], f)

            model = await kb.from_json(path, table_name="Document")
            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 3)

            d1 = await kb.get("d1", table_name=model.get_schema()["title"])
            self.assertEqual(d1.get_json()["text"], "v2")
        finally:
            kb.adapter.close()

    async def test_from_json_fts_index_built_so_search_works(self):
        # Same FTS-rebuild-after-load contract as CSV/Parquet — search
        # works against the rows just loaded.
        import json as _json

        path = os.path.join(self.temp_dir, "docs.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(
                [
                    {"id": "d1", "text": "quick brown fox"},
                    {"id": "d2", "text": "lazy dog sleeps"},
                    {"id": "d3", "text": "another quick rabbit"},
                ],
                f,
            )

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            model = await kb.from_json(path, table_name="Document")
            results = await kb.fulltext_search(
                "quick", table_name=model.get_schema()["title"], k=10
            )
            ids = {r["id"] for r in results}
            self.assertEqual(ids, {"d1", "d3"})
        finally:
            kb.adapter.close()

    async def test_from_json_restores_sandbox_after_load(self):
        # Mirrors the CSV sandbox test for the from_json code path —
        # the tear-down + reopen logic lives in _bulk_load and is
        # shared, but verifying each fast path independently catches
        # any future regression that special-cases JSON.
        import json as _json

        path = os.path.join(self.temp_dir, "docs.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump([{"id": "d1", "text": "hello"}], f)

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.from_json(path, table_name="Document")
            import duckdb

            with self.assertRaises(duckdb.Error):
                await kb.query(
                    f"SELECT * FROM read_json('{path}', format='array')",
                    read_only=False,
                )
        finally:
            kb.adapter.close()

    async def test_from_jsonl_restores_sandbox_after_load(self):
        # Same shape as the JSON sandbox test, exercising the
        # newline_delimited reader argument.
        import json as _json

        path = os.path.join(self.temp_dir, "docs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"id": "d1", "text": "hello"}) + "\n")

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.from_jsonl(path, table_name="Document")
            import duckdb

            with self.assertRaises(duckdb.Error):
                await kb.query(
                    f"SELECT * FROM read_json('{path}', format='newline_delimited')",
                    read_only=False,
                )
        finally:
            kb.adapter.close()

    async def test_from_json_raises_on_missing_file(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            import pytest

            with pytest.raises(FileNotFoundError):
                await kb.from_json("/nonexistent/x.json", table_name="Document")
        finally:
            kb.adapter.close()

    async def test_from_jsonl_round_trips_rows(self):
        import json as _json

        path = os.path.join(self.temp_dir, "docs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(50):
                f.write(
                    _json.dumps({"id": f"doc{i:03d}", "text": f"row {i} content"}) + "\n"
                )

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            model = await kb.from_jsonl(path, table_name="Document")
            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 50)
            first = await kb.get("doc000", table_name=model.get_schema()["title"])
            self.assertEqual(first.get_json()["text"], "row 0 content")
            last = await kb.get("doc049", table_name=model.get_schema()["title"])
            self.assertEqual(last.get_json()["text"], "row 49 content")
        finally:
            kb.adapter.close()

    async def test_from_jsonl_upserts_existing_rows(self):
        import json as _json

        path1 = os.path.join(self.temp_dir, "v1.jsonl")
        with open(path1, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"id": "d1", "text": "one"}) + "\n")
            f.write(_json.dumps({"id": "d2", "text": "two"}) + "\n")

        path2 = os.path.join(self.temp_dir, "v2.jsonl")
        with open(path2, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"id": "d1", "text": "one-updated"}) + "\n")
            f.write(_json.dumps({"id": "d3", "text": "three"}) + "\n")

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.from_jsonl(path1, table_name="Document")
            model = await kb.from_jsonl(path2, table_name="Document")
            rows = await kb.getall(
                table_name=model.get_schema()["title"], limit=100, offset=0
            )
            self.assertEqual(len(rows), 3)

            d1 = await kb.get("d1", table_name=model.get_schema()["title"])
            self.assertEqual(d1.get_json()["text"], "one-updated")
            d2 = await kb.get("d2", table_name=model.get_schema()["title"])
            self.assertEqual(d2.get_json()["text"], "two")
            d3 = await kb.get("d3", table_name=model.get_schema()["title"])
            self.assertEqual(d3.get_json()["text"], "three")
        finally:
            kb.adapter.close()

    async def test_from_jsonl_raises_on_missing_file(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            import pytest

            with pytest.raises(FileNotFoundError):
                await kb.from_jsonl("/nonexistent/x.jsonl", table_name="Document")
        finally:
            kb.adapter.close()

    async def test_from_csv_without_preregistered_data_model(self):
        # The headline use case for the new signature: no Pydantic
        # DataModel pre-declared on the KB, no data_model= kwarg —
        # the file's columns are enough. The returned SymbolicDataModel
        # is the handle the caller uses for subsequent queries, and it
        # gets registered on the adapter so default-table searches
        # find it.
        import csv

        csv_path = os.path.join(self.temp_dir, "auto.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])
            w.writerow(["b", "beta"])

        kb = KnowledgeBase(uri=self.db_path)  # no data_models=
        try:
            model = await kb.from_csv(
                csv_path,
                table_name="Article",
                table_description="Bulk-loaded articles for the demo.",
            )

            schema = model.get_schema()
            self.assertEqual(schema["title"], "Article")
            self.assertEqual(schema["description"], "Bulk-loaded articles for the demo.")
            self.assertEqual(list(schema["properties"].keys()), ["id", "text"])

            # The returned model resolves rows correctly.
            row = await kb.get("a", table_name=model.get_schema()["title"])
            self.assertEqual(row.get_json()["text"], "alpha")

            # Newly-created tables are visible to get_symbolic_data_models.
            registered_titles = {
                dm.get_schema().get("title") for dm in kb.get_symbolic_data_models()
            }
            self.assertIn("Article", registered_titles)
        finally:
            kb.adapter.close()

    async def test_from_csv_rejects_non_identifier_name(self):
        # ``name`` is normalized to PascalCase before being used as a
        # SQL identifier. PascalCase normalization strips the
        # separators in "bad name; DROP TABLE x;" but leaves the
        # alphanumeric core ("BadNameDropTableX") — that's a valid
        # identifier, so the call SHOULDN'T raise here. Verify the
        # SQL-injection attempt is *neutralized*, not just rejected.
        import csv

        csv_path = os.path.join(self.temp_dir, "x.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_csv(csv_path, table_name="bad name; DROP TABLE x;")
            # The malicious tokens are gone — only the alphanumeric
            # PascalCase residue survives as the table name.
            self.assertEqual(model.get_schema()["title"], "BadNameDropTableX")
        finally:
            kb.adapter.close()

    async def test_from_csv_name_defaults_to_filename_stem(self):
        # When `name` is omitted, the file's stem becomes the table
        # name — kebab-case and snake-case filenames both normalize to
        # PascalCase.
        import csv

        # Stem "my-articles" -> "MyArticles".
        csv_path = os.path.join(self.temp_dir, "my-articles.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])
            w.writerow(["b", "beta"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_csv(csv_path)
            self.assertEqual(model.get_schema()["title"], "MyArticles")
            row = await kb.get("a", table_name=model.get_schema()["title"])
            self.assertEqual(row.get_json()["text"], "alpha")
        finally:
            kb.adapter.close()

    async def test_from_csv_name_is_pascal_cased(self):
        # Explicit names also get coerced — callers can use whatever
        # casing convention they want, the stored table is consistent.
        import csv

        csv_path = os.path.join(self.temp_dir, "x.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            m1 = await kb.from_csv(csv_path, table_name="my_documents")
            self.assertEqual(m1.get_schema()["title"], "MyDocuments")

            # Re-loading with a synonymous name lands on the same
            # table (idempotent upsert), proving the normalization is
            # consistent across callers.
            csv_path_2 = os.path.join(self.temp_dir, "x2.csv")
            with open(csv_path_2, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "text"])
                w.writerow(["a", "alpha-updated"])
            m2 = await kb.from_csv(csv_path_2, table_name="my-documents")
            self.assertEqual(m2.get_schema()["title"], "MyDocuments")
            row = await kb.get("a", table_name=m2.get_schema()["title"])
            self.assertEqual(row.get_json()["text"], "alpha-updated")
        finally:
            kb.adapter.close()

    async def test_from_csv_normalizes_columns_to_snake_case(self):
        # File headers can be anything — mixedCase, spaced, kebab.
        # The adapter snake-cases every header before it lands in the
        # table, so the resulting schema's properties are uniformly
        # snake_case regardless of the source convention.
        import csv

        csv_path = os.path.join(self.temp_dir, "wild.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            # Three different naming conventions in the same header.
            w.writerow(["UserID", "First Name", "email-address"])
            w.writerow(["u001", "Ada", "ada@example.com"])
            w.writerow(["u002", "Grace", "grace@example.com"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_csv(csv_path, table_name="Person")

            # Schema's `title` is PascalCase, properties are snake_case.
            schema = model.get_schema()
            self.assertEqual(schema["title"], "Person")
            self.assertEqual(
                list(schema["properties"].keys()),
                ["user_id", "first_name", "email_address"],
            )

            # Rows are queryable through the canonical primary key
            # name (the snake_case form, not the file's "UserID").
            row = await kb.get("u001", table_name=model.get_schema()["title"])
            self.assertEqual(row.get_json()["first_name"], "Ada")
            self.assertEqual(row.get_json()["email_address"], "ada@example.com")
        finally:
            kb.adapter.close()

    async def test_from_parquet_normalizes_columns_to_snake_case(self):
        # Parquet headers come from the file's footer schema, not a
        # text header row, but they still need the same snake_case
        # treatment so callers can rely on the column-naming
        # convention regardless of source format.
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = os.path.join(self.temp_dir, "wild.parquet")
        table = pa.table(
            {
                "UserID": ["u001", "u002"],
                "First Name": ["Ada", "Grace"],
                "email-address": ["ada@example.com", "grace@example.com"],
            }
        )
        pq.write_table(table, path)

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_parquet(path, table_name="Person")

            schema = model.get_schema()
            self.assertEqual(schema["title"], "Person")
            self.assertEqual(
                list(schema["properties"].keys()),
                ["user_id", "first_name", "email_address"],
            )

            row = await kb.get("u001", table_name=model.get_schema()["title"])
            self.assertEqual(row.get_json()["first_name"], "Ada")
        finally:
            kb.adapter.close()

    async def test_from_json_normalizes_columns_to_snake_case(self):
        # JSON object keys can be anything — same canonicalization
        # applies at the column boundary.
        import json as _json

        path = os.path.join(self.temp_dir, "wild.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(
                [
                    {
                        "UserID": "u001",
                        "First Name": "Ada",
                        "email-address": "ada@example.com",
                    },
                    {
                        "UserID": "u002",
                        "First Name": "Grace",
                        "email-address": "grace@example.com",
                    },
                ],
                f,
            )

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_json(path, table_name="Person")

            schema = model.get_schema()
            self.assertEqual(schema["title"], "Person")
            self.assertEqual(
                list(schema["properties"].keys()),
                ["user_id", "first_name", "email_address"],
            )

            row = await kb.get("u001", table_name=model.get_schema()["title"])
            self.assertEqual(row.get_json()["first_name"], "Ada")
        finally:
            kb.adapter.close()

    async def test_from_jsonl_normalizes_columns_to_snake_case(self):
        # JSONL is the streaming sibling of JSON — keys can still be
        # anything per line, and the same column-name canonicalization
        # applies.
        import json as _json

        path = os.path.join(self.temp_dir, "wild.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                _json.dumps(
                    {
                        "UserID": "u001",
                        "First Name": "Ada",
                        "email-address": "ada@example.com",
                    }
                )
                + "\n"
            )
            f.write(
                _json.dumps(
                    {
                        "UserID": "u002",
                        "First Name": "Grace",
                        "email-address": "grace@example.com",
                    }
                )
                + "\n"
            )

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_jsonl(path, table_name="Person")

            schema = model.get_schema()
            self.assertEqual(schema["title"], "Person")
            self.assertEqual(
                list(schema["properties"].keys()),
                ["user_id", "first_name", "email_address"],
            )

            row = await kb.get("u001", table_name=model.get_schema()["title"])
            self.assertEqual(row.get_json()["first_name"], "Ada")
        finally:
            kb.adapter.close()

    async def test_from_parquet_name_defaults_to_filename_stem(self):
        # Filename-stem fallback applies to every fast-path format.
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = os.path.join(self.temp_dir, "blog-posts.parquet")
        pq.write_table(pa.table({"id": ["a", "b"], "text": ["one", "two"]}), path)

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_parquet(path)
            self.assertEqual(model.get_schema()["title"], "BlogPosts")
        finally:
            kb.adapter.close()

    async def test_from_json_name_defaults_to_filename_stem(self):
        import json as _json

        path = os.path.join(self.temp_dir, "blog-posts.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump([{"id": "a", "text": "one"}], f)

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_json(path)
            self.assertEqual(model.get_schema()["title"], "BlogPosts")
        finally:
            kb.adapter.close()

    async def test_from_jsonl_name_defaults_to_filename_stem(self):
        import json as _json

        path = os.path.join(self.temp_dir, "blog-posts.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"id": "a", "text": "one"}) + "\n")

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_jsonl(path)
            self.assertEqual(model.get_schema()["title"], "BlogPosts")
        finally:
            kb.adapter.close()

    async def test_delete_single_id_returns_count(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.update(
                [
                    JsonDataModel(data_model=Document(id="d1", text="one")),
                    JsonDataModel(data_model=Document(id="d2", text="two")),
                    JsonDataModel(data_model=Document(id="d3", text="three")),
                ]
            )

            n = await kb.delete("d2", table_name="Document")
            self.assertEqual(n, 1)

            # d2 is gone, d1 and d3 remain.
            self.assertIsNone(await kb.get("d2", table_name="Document"))
            self.assertIsNotNone(await kb.get("d1", table_name="Document"))
            self.assertIsNotNone(await kb.get("d3", table_name="Document"))
        finally:
            kb.adapter.close()

    async def test_delete_list_of_ids_returns_count_of_matches(self):
        # Mixed list (some present, one missing): only the matching
        # rows count toward the return value.
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.update(
                [
                    JsonDataModel(data_model=Document(id="d1", text="one")),
                    JsonDataModel(data_model=Document(id="d2", text="two")),
                ]
            )
            n = await kb.delete(["d1", "ghost", "d2"], table_name="Document")
            self.assertEqual(n, 2)
            rows = await kb.getall(table_name="Document", limit=10)
            self.assertEqual(len(rows), 0)
        finally:
            kb.adapter.close()

    async def test_delete_empty_list_is_noop(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.update(JsonDataModel(data_model=Document(id="d1", text="one")))
            n = await kb.delete([], table_name="Document")
            self.assertEqual(n, 0)
            # Row still there.
            self.assertIsNotNone(await kb.get("d1", table_name="Document"))
        finally:
            kb.adapter.close()

    async def test_delete_rebuilds_fts_so_search_doesnt_return_ghost(self):
        # Pre-fix would leave the FTS index pointing at the deleted
        # row's text; verify the rebuild step actually drops it from
        # the search results.
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.update(
                [
                    JsonDataModel(data_model=Document(id="d1", text="quick fox")),
                    JsonDataModel(data_model=Document(id="d2", text="quick rabbit")),
                ]
            )
            await kb.delete("d1", table_name="Document")
            hits = await kb.fulltext_search("quick", table_name="Document", k=10)
            ids = {r["id"] for r in hits}
            self.assertNotIn("d1", ids)
            self.assertIn("d2", ids)
        finally:
            kb.adapter.close()

    async def test_delete_from_missing_table_warns_and_returns_zero(self):
        import warnings as _w

        kb = KnowledgeBase(uri=self.db_path)
        try:
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                n = await kb.delete("x", table_name="NoSuchTable")
            self.assertEqual(n, 0)
            self.assertTrue(
                any("NoSuchTable" in str(w.message) for w in caught),
                "expected a warning naming the missing table",
            )
        finally:
            kb.adapter.close()

    async def test_drop_table_removes_table_and_returns_true(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.update(JsonDataModel(data_model=Document(id="d1", text="x")))

            dropped = await kb.drop_table("Document")
            self.assertTrue(dropped)

            # Table is gone from the registered list.
            registered = {
                m.get_schema().get("title") for m in kb.get_symbolic_data_models()
            }
            self.assertNotIn("Document", registered)

            # And getall against the dropped name no longer returns
            # rows (it warns + returns [] under the soft-mismatch
            # contract).
            rows = await kb.getall(table_name="Document", limit=10)
            self.assertEqual(rows, [])
        finally:
            kb.adapter.close()

    async def test_drop_table_returns_false_when_table_missing(self):
        kb = KnowledgeBase(uri=self.db_path)
        try:
            dropped = await kb.drop_table("NoSuchTable")
            self.assertFalse(dropped)
        finally:
            kb.adapter.close()

    async def test_drop_table_normalizes_name_to_pascal_case(self):
        # Same canonicalization rule as everywhere else: the caller can
        # pass kebab-case / snake_case and it still finds the table.
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        try:
            await kb.update(JsonDataModel(data_model=Document(id="d1", text="x")))
            dropped = await kb.drop_table("document")
            self.assertTrue(dropped)
        finally:
            kb.adapter.close()

    async def test_rename_changes_table_name_and_preserves_rows(self):
        # ALTER TABLE round-trip: the same rows are queryable under
        # the new name afterwards, and the returned SymbolicDataModel
        # reflects the new title.
        import csv

        csv_path = os.path.join(self.temp_dir, "src.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])
            w.writerow(["b", "beta"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            original = await kb.from_csv(csv_path, table_name="OldName")

            renamed = await kb.rename(original, table_name="NewName")

            self.assertEqual(renamed.get_schema()["title"], "NewName")
            row = await kb.get("a", table_name=renamed.get_schema()["title"])
            self.assertEqual(row.get_json()["text"], "alpha")

            # The old name no longer resolves.
            registered_titles = {
                dm.get_schema().get("title") for dm in kb.get_symbolic_data_models()
            }
            self.assertIn("NewName", registered_titles)
            self.assertNotIn("OldName", registered_titles)
        finally:
            kb.adapter.close()

    async def test_rename_normalizes_new_name_to_pascal_case(self):
        # Same input-normalization rule as from_*: whatever casing
        # the caller passes ends up as PascalCase.
        import csv

        csv_path = os.path.join(self.temp_dir, "src.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            original = await kb.from_csv(csv_path, table_name="Source")
            renamed = await kb.rename(original, table_name="my-new-table")
            self.assertEqual(renamed.get_schema()["title"], "MyNewTable")
        finally:
            kb.adapter.close()

    async def test_rename_updates_description_without_renaming(self):
        # Description-only update: no ALTER TABLE, just a new
        # SymbolicDataModel with the description applied at the
        # schema's top level.
        import csv

        csv_path = os.path.join(self.temp_dir, "src.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            original = await kb.from_csv(csv_path, table_name="Doc")
            renamed = await kb.rename(
                original, table_description="Now with a description."
            )

            self.assertEqual(renamed.get_schema()["title"], "Doc")
            self.assertEqual(
                renamed.get_schema()["description"],
                "Now with a description.",
            )
        finally:
            kb.adapter.close()

    async def test_rename_preserves_existing_description_when_only_renaming(self):
        # If the caller renames without touching description, the
        # description from the source model is carried over to the
        # renamed model — saves the caller from re-passing it.
        import csv

        csv_path = os.path.join(self.temp_dir, "src.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            original = await kb.from_csv(
                csv_path,
                table_name="OldName",
                table_description="Source description.",
            )
            renamed = await kb.rename(original, table_name="NewName")
            self.assertEqual(renamed.get_schema()["description"], "Source description.")
        finally:
            kb.adapter.close()

    async def test_rename_accepts_string_source(self):
        # Convenience: caller doesn't need to keep the original
        # SymbolicDataModel around — passing the table name string
        # works too, including pre-normalized variants.
        import csv

        csv_path = os.path.join(self.temp_dir, "src.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            await kb.from_csv(csv_path, table_name="Doc")

            # Plain string source.
            renamed = await kb.rename("Doc", table_name="Article")
            self.assertEqual(renamed.get_schema()["title"], "Article")

            # Kebab-case input is also accepted (gets normalized).
            renamed2 = await kb.rename("article", table_name="blog-post")
            self.assertEqual(renamed2.get_schema()["title"], "BlogPost")
        finally:
            kb.adapter.close()

    async def test_rename_requires_one_of_name_or_description(self):
        import csv

        csv_path = os.path.join(self.temp_dir, "src.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            model = await kb.from_csv(csv_path, table_name="Doc")
            with self.assertRaises(ValueError):
                await kb.rename(model)
        finally:
            kb.adapter.close()

    async def test_rename_raises_on_unknown_table(self):
        kb = KnowledgeBase(uri=self.db_path)
        try:
            with self.assertRaises(ValueError):
                await kb.rename("NoSuchTable", table_name="NewName")
        finally:
            kb.adapter.close()

    async def test_rename_keeps_fts_search_working(self):
        # After rename, fulltext_search through the new SymbolicDataModel
        # must keep returning hits — the FTS index is rebuilt under
        # the new table name as part of the rename.
        import csv

        csv_path = os.path.join(self.temp_dir, "src.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "the quick brown fox"])
            w.writerow(["b", "lazy dog sleeps"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            original = await kb.from_csv(csv_path, table_name="OldName")
            renamed = await kb.rename(original, table_name="NewName")
            hits = await kb.fulltext_search(
                "quick", table_name=renamed.get_schema()["title"], k=10
            )
            self.assertEqual({r["id"] for r in hits}, {"a"})
        finally:
            kb.adapter.close()

    async def test_from_csv_rejects_unnameable_stem(self):
        # A filename whose stem has no alphanumeric content after
        # PascalCase normalization can't yield a valid table name.
        # The error message points the caller at the explicit `table_name=`
        # escape hatch.
        import csv

        csv_path = os.path.join(self.temp_dir, "---.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text"])
            w.writerow(["a", "alpha"])

        kb = KnowledgeBase(uri=self.db_path)
        try:
            with self.assertRaises(ValueError):
                await kb.from_csv(csv_path)
        finally:
            kb.adapter.close()

    def test_knowledge_base_serialization(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document, Chunk],
            metric="cosine",
            wipe_on_start=False,
        )

        config = knowledge_base.get_config()
        cloned_knowledge_base = KnowledgeBase.from_config(config)
        self.assertEqual(
            cloned_knowledge_base.get_config(),
            knowledge_base.get_config(),
        )

    @patch("litellm.aembedding")
    def test_knowledge_base_serialization_with_embedding(self, mock_embedding):
        expected_value = np.random.rand(384).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document, Chunk],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=False,
        )

        config = knowledge_base.get_config()
        cloned_knowledge_base = KnowledgeBase.from_config(config)
        self.assertEqual(
            cloned_knowledge_base.get_config(),
            knowledge_base.get_config(),
        )
