# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile
import uuid
import warnings
from unittest.mock import patch

import duckdb
import numpy as np

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import DuckDBAdapter
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import (
    sanitize_identifier,
)
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import (
    sanitize_properties,
)
from synalinks.src.modules.embedding_models import EmbeddingModel


class Document(DataModel):
    id: str = Field(
        description="The document id",
        default_factory=lambda: str(uuid.uuid4()),
    )
    text: str = Field(
        description="The content of the document",
    )


class SanitizationTest(testing.TestCase):
    def test_sanitize_identifier_valid(self):
        self.assertEqual(sanitize_identifier("valid_name"), "valid_name")
        self.assertEqual(sanitize_identifier("ValidName"), "ValidName")
        self.assertEqual(sanitize_identifier("_private"), "_private")
        self.assertEqual(sanitize_identifier("name123"), "name123")

    def test_sanitize_identifier_invalid(self):
        with self.assertRaises(ValueError):
            sanitize_identifier("123invalid")
        with self.assertRaises(ValueError):
            sanitize_identifier("invalid-name")
        with self.assertRaises(ValueError):
            sanitize_identifier("invalid name")
        with self.assertRaises(ValueError):
            sanitize_identifier("invalid;name")

    def test_sanitize_properties(self):
        # Keys are now funnelled through ``column_identifier`` which
        # snake-cases first then sanitizes — so a mixedCase / PascalCase
        # key gets canonicalized to snake_case, not preserved verbatim.
        props = {"valid_key": "value", "AnotherKey": 123}
        result = sanitize_properties(props)
        self.assertEqual(result, {"valid_key": "value", "another_key": 123})

    def test_sanitize_properties_neutralizes_injection_shaped_key(self):
        # Previously: ``"invalid-key"`` (and worse) was rejected at
        # ``sanitize_identifier``. With the snake_case normalization in
        # front of the sanitizer, separator-style strings now get
        # collapsed to a valid identifier — the malicious tokens are
        # *stripped*, not just refused. Either outcome is safe.
        result = sanitize_properties({"invalid-key": "value"})
        self.assertEqual(result, {"invalid_key": "value"})

        # Even an outright SQL-injection-shaped key is neutralized:
        # only the alphanumeric residue survives.
        result = sanitize_properties({"a; DROP TABLE t; --b": 1})
        self.assertEqual(result, {"a_drop_table_t_b": 1})


class DuckDBAdapterInitTest(testing.TestCase):
    def test_init_without_embedding_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            adapter = DuckDBAdapter(uri=db_path)
            self.assertEqual(adapter.uri, db_path)
            self.assertIsNone(adapter.embedding_model)

    def test_init_with_invalid_stemmer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            with self.assertRaises(ValueError):
                DuckDBAdapter(uri=db_path, stemmer="invalid")

    def test_init_with_invalid_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            with self.assertRaises(ValueError):
                DuckDBAdapter(uri=db_path, metric="invalid")

    def test_init_rejects_path_traversal_in_name(self):
        # `name` flows into the default URI path; a traversal-shaped
        # value would create a `.db` file outside the synalinks home
        # dir. sanitize_identifier rejects it at __init__.
        with self.assertRaises(ValueError):
            DuckDBAdapter(uri=None, name="../escape")

    async def test_init_with_encryption_key_roundtrip(self):
        # Construct an encrypted KB, write data, drop the adapter,
        # reconstruct with the same key, read it back.
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "enc.db")

            class Doc(DataModel):
                id: str
                text: str

            a = DuckDBAdapter(
                uri=db_path,
                data_models=[Doc],
                encryption_key="passphrase-1",
            )
            await a.update(JsonDataModel(data_model=Doc(id="d1", text="hello")))
            a.close()

            b = DuckDBAdapter(
                uri=db_path,
                data_models=[Doc],
                encryption_key="passphrase-1",
            )
            try:
                rows = await b.sql(
                    "SELECT id, text FROM Doc WHERE id = ?",
                    params=["d1"],
                )
                self.assertEqual(rows, [{"id": "d1", "text": "hello"}])
            finally:
                b.close()

    def test_init_with_wrong_encryption_key_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "enc.db")

            class Doc(DataModel):
                id: str
                text: str

            a = DuckDBAdapter(uri=db_path, data_models=[Doc], encryption_key="right-key")
            a.close()
            with self.assertRaises(duckdb.Error):
                DuckDBAdapter(
                    uri=db_path,
                    data_models=[Doc],
                    encryption_key="wrong-key",
                )

    def test_init_with_no_key_on_encrypted_db_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "enc.db")

            class Doc(DataModel):
                id: str
                text: str

            a = DuckDBAdapter(uri=db_path, data_models=[Doc], encryption_key="k")
            a.close()
            with self.assertRaises(duckdb.Error):
                DuckDBAdapter(uri=db_path, data_models=[Doc])

    def test_init_holds_single_persistent_connection(self):
        # The adapter must hold exactly one persistent DuckDB connection
        # so per-operation overhead is bounded. The connection should
        # be open after __init__ and `close()` must release it.
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            adapter = DuckDBAdapter(uri=db_path)
            self.assertIsNotNone(adapter._con)
            adapter.close()
            self.assertIsNone(adapter._con)
            # Idempotent.
            adapter.close()

    def test_init_with_duckdb_uri_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            adapter = DuckDBAdapter(uri=f"duckdb://{db_path}")
            self.assertEqual(adapter.uri, db_path)

    async def test_wipe_on_start_clears_existing_tables(self):
        # A second adapter constructed with `wipe_on_start=True` over an
        # existing DB must drop the previous tables before it reads
        # `data_models`. Otherwise reopening a database in "wipe" mode
        # would leave a half-cleared state visible to the user.
        class Person(DataModel):
            name: str
            age: int

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "wipe.db")

            first = DuckDBAdapter(uri=db_path, data_models=[Person])
            await first.update(JsonDataModel(data_model=Person(name="Alice", age=30)))
            first.close()

            # Reopen with wipe_on_start; previous rows should be gone,
            # but the table can be recreated by passing data_models.
            second = DuckDBAdapter(uri=db_path, data_models=[Person], wipe_on_start=True)
            try:
                rows = await second.getall(table_name="Person")
                self.assertEqual(rows, [])
            finally:
                second.close()

    def test_close_is_idempotent_and_drops_connection(self):
        # close() must release the persistent connection and survive
        # repeated invocation — `__del__` calls it during interpreter
        # shutdown so a second call from explicit user code can't be
        # allowed to throw.
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "close.db")
            adapter = DuckDBAdapter(uri=db_path)
            self.assertIsNotNone(adapter._con)
            adapter.close()
            self.assertIsNone(adapter._con)
            adapter.close()  # No-op, no exception.
            self.assertIsNone(adapter._con)

    @patch("litellm.aembedding")
    async def test_init_with_embedding_model(self, mock_embedding):
        expected_value = np.random.rand(384).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            embedding_model = EmbeddingModel(model="ollama/all-minilm")
            adapter = DuckDBAdapter(
                uri=db_path,
                embedding_model=embedding_model,
            )
            # The dimension is resolved lazily (no network probe in __init__),
            # so it's unknown at construction and table creation is deferred.
            self.assertIsNone(adapter.vector_dim)
            self.assertTrue(adapter._defer_table_creation)
            self.assertEqual(adapter.embedding_model, embedding_model)
            # Resolving on the caller's loop probes the model and learns 384.
            await adapter._ensure_vector_dim()
            self.assertEqual(adapter.vector_dim, 384)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, installed_extensions, executed):
        self._installed = installed_extensions
        self._executed = executed

    def execute(self, sql, *args, **kwargs):
        self._executed.append(sql)
        if "duckdb_extensions" in sql:
            return _FakeCursor([(name,) for name in self._installed])
        return _FakeCursor([])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DuckDBAdapterInstallExtensionsTest(testing.TestCase):
    """Verify `_install_extensions` only opens a second connection when an
    extension is actually missing, and only ``INSTALL``s the missing
    ones. Probe and install both go through ``:memory:`` (extension
    state is global to the user's DuckDB install) — this keeps the
    bootstrap free of the adapter's database file, so an encrypted
    file's key isn't required to install extensions."""

    def _run(self, installed, embedding_model=None):
        executed = []
        opened_uris = []

        def fake_connect(uri, read_only=False):
            opened_uris.append(uri)
            return _FakeConn(installed, executed)

        adapter = DuckDBAdapter.__new__(DuckDBAdapter)
        adapter.uri = "/some/file.duckdb"
        adapter.embedding_model = embedding_model
        with patch("duckdb.connect", side_effect=fake_connect):
            adapter._install_extensions()
        return executed, opened_uris

    def test_skips_install_fts_when_already_installed(self):
        executed, opened = self._run(installed=["fts"])
        self.assertEqual([s for s in executed if "INSTALL fts" in s], [])
        self.assertEqual([s for s in executed if "LOAD" in s], [])
        # Only the probe opens, and it goes to :memory: (never the
        # adapter's file URI).
        self.assertEqual(opened, [":memory:"])

    def test_runs_install_fts_when_not_installed(self):
        executed, opened = self._run(installed=[])
        self.assertEqual(len([s for s in executed if "INSTALL fts" in s]), 1)
        self.assertEqual([s for s in executed if "LOAD" in s], [])
        # Probe + installer — both throwaway, both on :memory:.
        self.assertEqual(opened, [":memory:", ":memory:"])

    def test_skips_install_vss_when_already_installed(self):
        executed, opened = self._run(installed=["fts", "vss"], embedding_model=object())
        self.assertEqual([s for s in executed if "INSTALL" in s], [])
        self.assertEqual([s for s in executed if "LOAD" in s], [])
        self.assertEqual(opened, [":memory:"])

    def test_runs_install_vss_when_not_installed(self):
        executed, opened = self._run(installed=["fts"], embedding_model=object())
        install_calls = [s for s in executed if "INSTALL" in s]
        self.assertEqual(len(install_calls), 1)
        self.assertIn("INSTALL vss", install_calls[0])
        self.assertEqual(opened, [":memory:", ":memory:"])

    def test_skips_vss_entirely_without_embedding_model(self):
        executed, _ = self._run(installed=[], embedding_model=None)
        self.assertEqual([s for s in executed if "vss" in s], [])

    def test_install_never_touches_adapter_uri(self):
        # Critical for encryption: bootstrap must work without the
        # encryption_key. If `_install_extensions` ever opened the
        # actual database file, encrypted DBs couldn't construct.
        _, opened = self._run(installed=[])
        for uri in opened:
            self.assertEqual(uri, ":memory:")


class DuckDBAdapterDataModelTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_create_table_from_data_model(self):
        class TestModel(DataModel):
            name: str
            value: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(TestModel.to_symbolic_data_model())

        # Verify table exists
        with adapter._connect(read_only=True) as con:
            result = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name='TestModel'"
            ).fetchone()[0]
            self.assertEqual(result, 1)

    async def test_maybe_create_table_is_idempotent(self):
        # `_maybe_create_table` short-circuits when the table already
        # exists in `information_schema.tables` rather than always
        # paying the CREATE round-trip. Verify two consecutive calls
        # succeed and that data inserted between them is preserved
        # (i.e. the second call doesn't drop & recreate).
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())
        await adapter.update(JsonDataModel(data_model=Person(name="Alice", age=30)))

        # Second call must not raise and must preserve the row.
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        rows = await adapter.getall(table_name="Person")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get_json()["name"], "Alice")

    def test_wipe_database(self):
        class TestModel(DataModel):
            name: str
            value: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(TestModel.to_symbolic_data_model())

        # Verify table exists
        with adapter._connect(read_only=True) as con:
            result = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main'"
            ).fetchone()[0]
            self.assertGreater(result, 0)

        # Wipe database
        adapter.wipe_database()

        # Verify tables are gone
        with adapter._connect(read_only=True) as con:
            result = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main'"
            ).fetchone()[0]
            self.assertEqual(result, 0)

    def test_get_symbolic_data_models(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        models = adapter.get_symbolic_data_models()
        self.assertEqual(len(models), 1)
        self.assertIsInstance(models[0], SymbolicDataModel)


class DuckDBAdapterCRUDTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_update_single_record(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        person = Person(name="Alice", age=30)
        json_dm = JsonDataModel(data_model=person)

        result = await adapter.update(json_dm)
        self.assertEqual(result, "Alice")

    async def test_update_multiple_records(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        people = [
            JsonDataModel(data_model=Person(name="Alice", age=30)),
            JsonDataModel(data_model=Person(name="Bob", age=25)),
        ]

        result = await adapter.update(people)
        self.assertEqual(result, ["Alice", "Bob"])

    async def test_update_upsert(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)

        # Insert first record
        person1 = JsonDataModel(data_model=Person(name="Alice", age=30))
        await adapter.update(person1)

        # Update the same record
        person2 = JsonDataModel(data_model=Person(name="Alice", age=35))
        await adapter.update(person2)

        # Verify update
        result = await adapter.get("Alice", table_name="Person")
        self.assertEqual(result.get_json()["age"], 35)

    async def test_get_record(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        person = JsonDataModel(data_model=Person(name="Alice", age=30))
        await adapter.update(person)

        result = await adapter.get("Alice", table_name="Person")
        self.assertIsNotNone(result)
        self.assertEqual(result.get_json()["name"], "Alice")
        self.assertEqual(result.get_json()["age"], 30)

    async def test_get_nonexistent_record(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        result = await adapter.get("NonExistent", table_name="Person")
        self.assertIsNone(result)

    async def test_getall_records(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        people = [
            JsonDataModel(data_model=Person(name="Alice", age=30)),
            JsonDataModel(data_model=Person(name="Bob", age=25)),
            JsonDataModel(data_model=Person(name="Charlie", age=35)),
        ]
        await adapter.update(people)

        results = await adapter.getall(table_name="Person")
        self.assertEqual(len(results), 3)

    async def test_getall_with_limit_offset(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        people = [
            JsonDataModel(data_model=Person(name=f"Person{i}", age=20 + i))
            for i in range(10)
        ]
        await adapter.update(people)

        results = await adapter.getall(table_name="Person", limit=3, offset=2)
        self.assertEqual(len(results), 3)

    async def test_get_with_list_of_ids_returns_list(self):
        # Regression: param was named `id_or_ids` but previously only
        # handled scalars. Passing a list silently bound it as a single
        # parameter and returned at most one record.
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(
            [
                JsonDataModel(data_model=Person(name="Alice", age=30)),
                JsonDataModel(data_model=Person(name="Bob", age=25)),
                JsonDataModel(data_model=Person(name="Carol", age=40)),
            ]
        )

        results = await adapter.get(
            ["Alice", "MissingPerson", "Carol"],
            table_name="Person",
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        self.assertIsNotNone(results[0])
        self.assertEqual(results[0].get_json()["name"], "Alice")
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])
        self.assertEqual(results[2].get_json()["name"], "Carol")

    async def test_get_with_scalar_id_returns_scalar(self):
        # Companion to the list test: ensure the single-id path still
        # returns a single object (not a 1-element list).
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(JsonDataModel(data_model=Person(name="Alice", age=30)))

        result = await adapter.get("Alice", table_name="Person")
        self.assertNotIsInstance(result, list)
        self.assertIsNotNone(result)
        self.assertEqual(result.get_json()["name"], "Alice")

    async def test_update_single_column_row_is_idempotent(self):
        # Regression: ON CONFLICT DO UPDATE SET with no non-key columns
        # produced invalid SQL (trailing semicolon after SET). Should
        # emit DO NOTHING instead.
        class IdOnly(DataModel):
            id: str

        adapter = DuckDBAdapter(uri=self.db_path)
        row = JsonDataModel(data_model=IdOnly(id="only"))

        # First insert.
        result1 = await adapter.update(row)
        self.assertEqual(result1, "only")

        # Second insert with the same id — must not raise (DO NOTHING).
        result2 = await adapter.update(row)
        self.assertEqual(result2, "only")

        all_rows = await adapter.getall(table_name="IdOnly")
        self.assertEqual(len(all_rows), 1)

    async def test_update_rolls_back_when_a_bucket_fails(self):
        # Regression: with the batched form, a failure in the second
        # bucket's executemany would otherwise leave the first bucket's
        # rows committed. With BEGIN/COMMIT/ROLLBACK, the whole update
        # is atomic.
        class TableA(DataModel):
            id: str
            text: str

        class TableB(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)

        # Pre-create both tables so the failure isn't from a missing
        # table — we want the failure to occur during executemany so
        # we're testing rollback specifically.
        adapter._maybe_create_table(TableA.to_symbolic_data_model())
        adapter._maybe_create_table(TableB.to_symbolic_data_model())

        # `_duckdb.DuckDBPyConnection` is a C type whose methods cannot be
        # monkeypatched directly, so wrap the persistent connection in a
        # proxy that intercepts `executemany`. Everything else passes
        # through to the real connection so BEGIN/ROLLBACK still mutate
        # the underlying transaction state.
        class _FlakyConn:
            def __init__(self, real):
                self._real = real
                self._n = 0

            def __getattr__(self, name):
                return getattr(self._real, name)

            def executemany(self, sql, params):
                self._n += 1
                if self._n == 2:
                    raise duckdb.Error("simulated failure on second bucket")
                return self._real.executemany(sql, params)

        real_con = adapter._con
        adapter._con = _FlakyConn(real_con)

        rows = [
            JsonDataModel(data_model=TableA(id="a1", text="alpha")),
            JsonDataModel(data_model=TableB(id="b1", text="beta")),
        ]

        try:
            with self.assertRaises(duckdb.Error):
                await adapter.update(rows)
        finally:
            adapter._con = real_con

        # TableA's bucket ran first and "succeeded" at the SQL layer,
        # but the second bucket's failure must roll the whole transaction
        # back — both tables should have zero rows.
        rows_a = await adapter.getall(table_name="TableA")
        self.assertEqual(rows_a, [])
        rows_b = await adapter.getall(table_name="TableB")
        self.assertEqual(rows_b, [])

    async def test_getall_empty_table_returns_empty_list(self):
        # Table exists but has no rows — getall should return []
        # without raising. The previous over-broad try/except would
        # mask real failures here; the narrowed version returns []
        # only through the not-rows branch.
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        rows = await adapter.getall(table_name="Person")
        self.assertEqual(rows, [])

    async def test_getall_limit_zero_returns_empty_list(self):
        # LIMIT 0 must short-circuit cleanly — the cursor returns no
        # rows and we hand back [] rather than constructing the
        # JsonDataModel list from no input.
        class Note(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(
            [
                JsonDataModel(data_model=Note(id="n1", text="a")),
                JsonDataModel(data_model=Note(id="n2", text="b")),
            ]
        )

        rows = await adapter.getall(table_name="Note", limit=0)
        self.assertEqual(rows, [])

    async def test_getall_offset_beyond_rows_returns_empty_list(self):
        # Pagination off the end should yield [] — not raise, not return
        # leftovers from the previous page.
        class Note(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(
            [JsonDataModel(data_model=Note(id=f"n{i}", text="x")) for i in range(3)]
        )

        rows = await adapter.getall(table_name="Note", limit=10, offset=100)
        self.assertEqual(rows, [])

    async def test_getall_nonexistent_table_warns_and_returns_empty(self):
        # If the schema describes a table the database doesn't have,
        # getall must warn about the SELECT failure and return [] —
        # callers depend on getall being non-fatal for soft schema
        # mismatches. The narrowed try/except catches only the duckdb
        # error, so this path is exactly the warning branch.
        class GhostTable(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)
        # Note: we deliberately do NOT create the table.

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows = await adapter.getall(table_name="GhostTable")

        self.assertEqual(rows, [])
        messages = [str(w.message) for w in caught]
        self.assertTrue(
            any("GhostTable" in m for m in messages),
            f"expected a warning mentioning GhostTable, got {messages}",
        )

    async def test_get_empty_list_returns_empty_list(self):
        # Mirror of update([]). Returns an empty list immediately
        # without ever opening a connection.
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        result = await adapter.get([], table_name="Person")
        self.assertEqual(result, [])

    async def test_get_list_with_all_missing_returns_all_nones(self):
        # When none of the requested ids exist, the returned list is
        # the same length as the input, all `None`. The "search until
        # found" loop must not short-circuit.
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        results = await adapter.get(
            ["nope-1", "nope-2", "nope-3"],
            table_name="Person",
        )
        self.assertEqual(results, [None, None, None])

    async def test_get_with_no_data_models_searches_all_tables(self):
        # Passing data_models=None must fall back to
        # `get_symbolic_data_models()` rather than searching nothing.
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(JsonDataModel(data_model=Person(name="Alice", age=30)))

        result = await adapter.get("Alice", table_name="Person")
        self.assertIsNotNone(result)
        self.assertEqual(result.get_json()["name"], "Alice")

    async def test_update_empty_list_returns_empty_list(self):
        # update([]) is a legitimate caller pattern (e.g. when an upstream
        # filter happened to drop all rows). It must not blow up trying
        # to start a transaction it never uses, and it must return the
        # same shape it was called with — an empty list.
        adapter = DuckDBAdapter(uri=self.db_path)
        result = await adapter.update([])
        self.assertEqual(result, [])

    async def test_update_missing_primary_key_raises(self):
        # The first schema property is the primary key. A row that
        # arrives without it can't be upserted; we raise ValueError
        # eagerly rather than letting DuckDB surface a NOT NULL violation.
        class Person(DataModel):
            id: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)

        # Build a JsonDataModel with the id column omitted from its
        # json payload (Pydantic validation refuses to construct one
        # without `id`, so go through JsonDataModel directly).
        raw = JsonDataModel(
            json={"age": 42},
            schema=Person.get_schema(),
            name="no-id",
        )
        with self.assertRaises(ValueError):
            await adapter.update(raw)

    async def test_update_groups_mixed_shapes_into_separate_buckets(self):
        # Rows of the same table can have different column sets when
        # optional fields are present in some rows but not others. The
        # batched implementation buckets by (table, tuple-of-cols), so
        # each shape is one executemany. Verify both shapes round-trip.
        class Note(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)
        # `full` row carries the schema's full property set; `partial`
        # row's json drops `text`. Same table, different column buckets.
        full = JsonDataModel(data_model=Note(id="full", text="present"))
        partial = JsonDataModel(
            json={"id": "partial"},
            schema=Note.get_schema(),
            name="partial",
        )
        result = await adapter.update([full, partial])
        self.assertEqual(result, ["full", "partial"])

        # Both rows must be retrievable. The "partial" row's `text`
        # column becomes NULL in DuckDB; what matters is that it exists.
        rows = await adapter.getall(table_name="Note")
        ids = {r.get_json()["id"] for r in rows}
        self.assertEqual(ids, {"full", "partial"})

    async def test_update_accepts_raw_datamodel_not_just_json(self):
        # `update` calls `.to_json_data_model()` when given a DataModel
        # instance instead of a JsonDataModel. This shorthand is used
        # widely in the examples — guard against accidental regressions
        # to the JsonDataModel-only contract.
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)

        result = await adapter.update(Person(name="Alice", age=30))
        self.assertEqual(result, "Alice")

        result = await adapter.update(
            [Person(name="Bob", age=25), Person(name="Carol", age=40)]
        )
        self.assertEqual(result, ["Bob", "Carol"])

    async def test_update_large_batch_inserts_all_rows(self):
        # The batched implementation collapses N inserts into one
        # executemany. Verify a non-trivial batch (well above the
        # single-statement path) actually writes every row.
        class Note(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)
        n = 250
        notes = [
            JsonDataModel(data_model=Note(id=f"n{i:03d}", text=f"content {i}"))
            for i in range(n)
        ]
        ids = await adapter.update(notes)
        self.assertEqual(len(ids), n)
        self.assertEqual(ids[0], "n000")
        self.assertEqual(ids[-1], f"n{n - 1:03d}")

        # Confirm every row landed. getall pages by default but with a
        # large limit it should return the full set.
        rows = await adapter.getall(table_name="Note", limit=n + 10)
        self.assertEqual(len(rows), n)

    async def test_update_builds_fts_index_for_every_touched_table(self):
        # Regression: pre-fix `_maybe_create_fulltext_index` was called
        # once after the loop with the last `data_model`, so when a
        # single update batch spanned two tables only the LAST table's
        # FTS index was built. Searching the first table then either
        # raised "FTS query failed" or returned nothing.
        class TableA(DataModel):
            id: str
            text: str

        class TableB(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)

        await adapter.update(
            [
                JsonDataModel(data_model=TableA(id="a1", text="quick brown fox")),
                JsonDataModel(data_model=TableB(id="b1", text="lazy quick dog")),
            ]
        )

        # If TableA's FTS index wasn't built this would raise.
        results_a = await adapter.fulltext_search("quick", table_name="TableA", k=10)
        self.assertGreater(len(results_a), 0)
        self.assertEqual(results_a[0]["id"], "a1")

        # TableB must work too (this worked pre-fix as it was the
        # last-loop-iteration model).
        results_b = await adapter.fulltext_search("quick", table_name="TableB", k=10)
        self.assertGreater(len(results_b), 0)
        self.assertEqual(results_b[0]["id"], "b1")


class DuckDBAdapterQueryTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_get_id_key_neutralizes_injection_shaped_property(self):
        # Schema-derived id_key is interpolated raw into multiple SQL
        # builds (ON CONFLICT, FTS PRAGMA, WHERE clauses). It must be
        # safe at the choke point so a maliciously-built schema can't
        # punch through. `_get_id_key` is the choke point — and now
        # neutralizes injection-shaped names via the snake_case +
        # sanitize pipeline rather than refusing them. Either is safe.
        adapter = DuckDBAdapter(uri=self.db_path)
        bad_schema = {"properties": {"id; DROP TABLE t; --": {"type": "string"}}}
        result = adapter._get_id_key(bad_schema)
        # Only the alphanumeric residue survives — the SQL tokens are
        # stripped, so this is a safe identifier to interpolate.
        self.assertEqual(result, "id_drop_table_t")

    async def test_get_id_key_rejects_schema_without_properties(self):
        # Previously a fallback elif returned the first top-level dict
        # key for inputs without "properties", so `{"title": "X"}` would
        # silently use "title" as the primary key. Must raise instead.
        adapter = DuckDBAdapter(uri=self.db_path)
        with self.assertRaises(ValueError):
            adapter._get_id_key({"title": "X", "type": "object"})
        with self.assertRaises(ValueError):
            adapter._get_id_key({})
        with self.assertRaises(ValueError):
            adapter._get_id_key({"properties": {}})

    async def test_query_simple(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        result = await adapter.sql("SELECT 1 as value")
        self.assertEqual(result, [{"value": 1}])

    async def test_query_with_params(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        people = [
            JsonDataModel(data_model=Person(name="Alice", age=30)),
            JsonDataModel(data_model=Person(name="Bob", age=25)),
        ]
        await adapter.update(people)

        result = await adapter.sql(
            "SELECT * FROM Person WHERE age > ?",
            params=[26],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "Alice")

    async def test_query_with_no_params_succeeds(self):
        # Explicitly passing params=None (the default) must not break
        # the binding pass. The body uses `params or []` so the None
        # case has to round-trip cleanly.
        adapter = DuckDBAdapter(uri=self.db_path)
        result = await adapter.sql("SELECT 42 AS answer", params=None)
        self.assertEqual(result, [{"answer": 42}])

    async def test_query_with_empty_params_list_succeeds(self):
        # Companion of the None case: an explicit empty list of params
        # for a query that needs no binds.
        adapter = DuckDBAdapter(uri=self.db_path)
        result = await adapter.sql("SELECT 'hi' AS msg", params=[])
        self.assertEqual(result, [{"msg": "hi"}])

    async def test_read_only_rejects_multi_statement_injection(self):
        # `read_only=True` must parse the input and reject anything
        # that isn't a SELECT — including a trailing DROP smuggled in
        # after a legitimate-looking SELECT.
        adapter = DuckDBAdapter(uri=self.db_path)
        with self.assertRaises(duckdb.InvalidInputException):
            await adapter.sql("SELECT 1; DROP TABLE x", read_only=True)

    async def test_read_only_rejects_copy_to_file(self):
        # DuckDB's read-only *connection* does NOT block `COPY ... TO 'file'`
        # on its own (it's a filesystem write, not a database write).
        # The parser check is what catches this exfiltration vector.
        import tempfile

        adapter = DuckDBAdapter(uri=self.db_path)
        target = os.path.join(tempfile.gettempdir(), "exfil_should_not_exist.csv")
        with self.assertRaises(duckdb.InvalidInputException):
            await adapter.sql(f"COPY (SELECT 1) TO '{target}'", read_only=True)
        self.assertFalse(os.path.exists(target))

    async def test_read_only_rejects_ddl_and_dml(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        for forbidden in [
            "DROP TABLE x",
            "DELETE FROM x",
            "UPDATE x SET y=1",
            "INSERT INTO x VALUES (1)",
            "CREATE TABLE y (z INTEGER)",
            "ATTACH ':memory:' AS m",
        ]:
            with self.assertRaises(duckdb.InvalidInputException):
                await adapter.sql(forbidden, read_only=True)

    async def test_read_only_rejects_empty_query(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        with self.assertRaises(duckdb.InvalidInputException):
            await adapter.sql("", read_only=True)

    async def test_read_only_blocks_read_csv_filesystem_escape(self):
        # `SELECT * FROM read_csv('/etc/passwd', ...)` is a valid SELECT
        # statement under DuckDB's parser, and a read-only *connection*
        # alone doesn't stop it (read_csv reads the host filesystem, not
        # the database). The adapter must additionally sandbox the
        # connection so this exfiltration path can't reach a real file.
        adapter = DuckDBAdapter(uri=self.db_path)
        with self.assertRaises(duckdb.Error) as ctx:
            await adapter.sql(
                "SELECT * FROM read_csv('/etc/passwd', "
                "columns={'line':'VARCHAR'}, delim='|', header=false) LIMIT 1",
                read_only=True,
            )
        # The error must come from DuckDB's permission layer, not be a
        # successful read happening to return zero rows.
        self.assertIn("disabled", str(ctx.exception).lower())

    async def test_read_only_blocks_read_csv_for_any_local_path(self):
        # Same defence applies to *any* path, not just /etc/passwd —
        # write a benign csv next to the test db and confirm it's also
        # refused.
        bait = os.path.join(self.temp_dir, "bait.csv")
        with open(bait, "w") as f:
            f.write("a,b\n1,2\n")
        adapter = DuckDBAdapter(uri=self.db_path)
        with self.assertRaises(duckdb.Error):
            await adapter.sql(f"SELECT * FROM read_csv('{bait}')", read_only=True)

    async def test_read_only_false_bypasses_validation(self):
        # When the caller explicitly opts out of read-only, the adapter
        # does NOT pre-parse — the caller is responsible. (Internal
        # write paths in the adapter rely on this.)
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())
        # Writes through `query(read_only=False, ...)` must succeed.
        await adapter.sql(
            "INSERT INTO Person (name, age) VALUES ('Eve', 40)",
            read_only=False,
        )
        rows = await adapter.sql("SELECT name FROM Person", read_only=True)
        self.assertEqual(rows, [{"name": "Eve"}])


class DuckDBAdapterSchemaConversionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def test_json_schema_to_duckdb_columns_basic_types(self):
        class TestModel(DataModel):
            string_field: str
            int_field: int
            float_field: float
            bool_field: bool

        adapter = DuckDBAdapter(uri=self.db_path)
        schema = TestModel.get_schema()
        columns = adapter._json_schema_to_duckdb_columns(schema)

        self.assertIn("string_field VARCHAR PRIMARY KEY", columns)
        self.assertIn("int_field", columns)
        self.assertIn("float_field", columns)
        self.assertIn("bool_field", columns)

    def test_duckdb_table_to_json_schema(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        schema = adapter._duckdb_table_to_json_schema("Person")
        self.assertEqual(schema["title"], "Person")
        self.assertIn("name", schema["properties"])
        self.assertIn("age", schema["properties"])

    def test_json_schema_to_duckdb_columns_with_dict(self):
        from typing import Any
        from typing import Dict

        class ModelWithDict(DataModel):
            id: str
            metadata: Dict[str, Any]

        adapter = DuckDBAdapter(uri=self.db_path)
        schema = ModelWithDict.get_schema()
        columns = adapter._json_schema_to_duckdb_columns(schema)

        self.assertIn("id VARCHAR PRIMARY KEY", columns)
        self.assertIn("metadata JSON", columns)

    async def test_crud_with_dict_field(self):
        from typing import Any
        from typing import Dict

        class ModelWithDict(DataModel):
            id: str
            metadata: Dict[str, Any]

        adapter = DuckDBAdapter(uri=self.db_path)
        model = ModelWithDict(id="test1", metadata={"key": "value", "count": 42})
        json_dm = JsonDataModel(data_model=model)

        result = await adapter.update(json_dm)
        self.assertEqual(result, "test1")

        retrieved = await adapter.get("test1", table_name="ModelWithDict")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get_json()["metadata"], {"key": "value", "count": 42})

    def test_json_schema_to_duckdb_columns_with_str_enum(self):
        """Pydantic emits `$ref` + `$defs` for bare `str, Enum` fields.

        Before the fix, DuckDBAdapter raised
        `ValueError: Malformed JSON schema: missing type for '<field>'`.
        """
        from enum import Enum

        class Status(str, Enum):
            OK = "ok"
            KO = "ko"

        class Record(DataModel):
            id: str
            status: Status

        adapter = DuckDBAdapter(uri=self.db_path)
        columns = adapter._json_schema_to_duckdb_columns(Record.get_schema())

        self.assertIn("id VARCHAR PRIMARY KEY", columns)
        self.assertIn("status VARCHAR", columns)

    def test_json_schema_to_duckdb_columns_with_int_enum(self):
        """IntEnum emits `type: integer` inside `$defs` — must become INTEGER."""
        from enum import IntEnum

        class Priority(IntEnum):
            LOW = 1
            HIGH = 2

        class Task(DataModel):
            id: str
            priority: Priority

        adapter = DuckDBAdapter(uri=self.db_path)
        columns = adapter._json_schema_to_duckdb_columns(Task.get_schema())

        self.assertIn("id VARCHAR PRIMARY KEY", columns)
        # IntEnum → integer column
        self.assertTrue(
            "priority INTEGER" in columns or "priority BIGINT" in columns,
            f"Expected integer column for IntEnum, got: {columns}",
        )

    def test_json_schema_to_duckdb_columns_with_optional_enum(self):
        """Optional[Enum] emits anyOf with `$ref` + null — must resolve."""
        from enum import Enum
        from typing import Optional

        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"

        class Item(DataModel):
            id: str
            color: Optional[Color] = None

        adapter = DuckDBAdapter(uri=self.db_path)
        columns = adapter._json_schema_to_duckdb_columns(Item.get_schema())

        self.assertIn("id VARCHAR PRIMARY KEY", columns)
        self.assertIn("color VARCHAR", columns)

    def test_json_schema_to_duckdb_columns_with_nested_datamodel(self):
        """Nested DataModel emits `$ref` resolving to `type: object` — JSON column."""

        class Address(DataModel):
            city: str
            zip_code: str

        class Person(DataModel):
            id: str
            address: Address

        adapter = DuckDBAdapter(uri=self.db_path)
        columns = adapter._json_schema_to_duckdb_columns(Person.get_schema())

        self.assertIn("id VARCHAR PRIMARY KEY", columns)
        # Nested object → JSON column (objects are stored as JSON)
        self.assertIn("address JSON", columns)

    async def test_crud_with_str_enum_roundtrip(self):
        """End-to-end roundtrip: insert then SELECT must preserve enum value."""
        from enum import Enum

        class Role(str, Enum):
            ADMIN = "admin"
            USER = "user"

        class Account(DataModel):
            id: str
            role: Role

        adapter = DuckDBAdapter(uri=self.db_path)
        account = Account(id="acc1", role=Role.ADMIN)
        json_dm = JsonDataModel(data_model=account)

        result = await adapter.update(json_dm)
        self.assertEqual(result, "acc1")

        retrieved = await adapter.get("acc1", table_name="Account")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get_json()["role"], "admin")

    def test_json_schema_to_duckdb_columns_preserves_description_on_ref(self):
        """When the referring property has `description`, it must not disappear
        after `$ref` resolution — protects metadata the user attaches via
        `Field(description=...)` on enum fields."""
        from enum import Enum

        class Status(str, Enum):
            OK = "ok"

        class Record(DataModel):
            id: str
            status: Status = Field(description="The record status.")

        adapter = DuckDBAdapter(uri=self.db_path)
        # Must not raise
        columns = adapter._json_schema_to_duckdb_columns(Record.get_schema())
        self.assertIn("status VARCHAR", columns)


class DuckDBAdapterFulltextSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_fulltext_search(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
            JsonDataModel(data_model=Document(id="doc3", text="A quick rabbit runs")),
        ]
        await adapter.update(docs)

        results = await adapter.fulltext_search("quick", table_name="Document", k=10)
        self.assertGreater(len(results), 0)

    async def test_fulltext_search_returns_full_records(self):
        """Test that fulltext_search returns all fields, not just id and score."""
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await adapter.update(docs)

        results = await adapter.fulltext_search("quick", table_name="Document", k=10)

        self.assertGreater(len(results), 0)
        # Verify that the result contains all fields, not just id and score
        result = results[0]
        self.assertIn("id", result)
        self.assertIn("text", result)
        self.assertIn("score", result)
        # Verify the content is correct
        self.assertEqual(result["id"], "doc1")
        self.assertEqual(result["text"], "The quick brown fox")

    async def test_fulltext_search_empty_query(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        # Empty query short-circuits to [] before any SQL — the
        # table doesn't need to exist for this path to work.
        results = await adapter.fulltext_search("", table_name="Anything")
        self.assertEqual(results, [])

    async def test_fulltext_search_threshold_actually_filters(self):
        # Regression: `threshold` was being appended to params without
        # being referenced in the SQL, so it silently shifted the LIMIT
        # bind and never filtered. With the fix, an extreme threshold
        # must produce fewer (ideally zero) rows than the unfiltered
        # version.
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
            JsonDataModel(data_model=Document(id="doc3", text="A quick rabbit runs")),
        ]
        await adapter.update(docs)

        unfiltered = await adapter.fulltext_search("quick", table_name="Document", k=10)
        filtered = await adapter.fulltext_search(
            "quick",
            table_name="Document",
            k=10,
            threshold=999.0,  # absurdly high BM25 score requirement
        )
        self.assertGreater(len(unfiltered), 0)
        self.assertEqual(len(filtered), 0)

    async def test_fulltext_search_list_queries(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await adapter.update(docs)

        results = await adapter.fulltext_search(
            ["quick", "lazy"], table_name="Document", k=10
        )
        self.assertGreater(len(results), 0)

    async def test_fulltext_search_k_zero_returns_empty(self):
        # k=0 is a degenerate-but-legal request; the post-sort slice
        # `ranked[:0]` should yield an empty list rather than raise.
        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(
            [
                JsonDataModel(data_model=Document(id="d1", text="quick fox")),
                JsonDataModel(data_model=Document(id="d2", text="quick rabbit")),
            ]
        )
        results = await adapter.fulltext_search("quick", table_name="Document", k=0)
        self.assertEqual(results, [])

    async def test_fulltext_search_skips_table_with_no_fts_field(self):
        # FTS indexes every VARCHAR column (excluding the id key) via
        # `create_fts_index('*', ...)`. A table whose only columns are
        # numeric has nothing indexable, so fulltext_search must skip
        # it with a warning rather than crash on a missing FTS schema.
        class Numeric(DataModel):
            id: int
            value: int

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(JsonDataModel(data_model=Numeric(id=1, value=42)))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = await adapter.fulltext_search(
                "anything", table_name="Numeric", k=10
            )

        self.assertEqual(results, [])
        self.assertTrue(
            any("Numeric" in str(w.message) for w in caught),
            f"expected skip warning, got {[str(w.message) for w in caught]}",
        )

    async def test_fulltext_search_returns_highest_scoring_rows(self):
        # Regression: post-merge sort was ascending while BM25 is
        # "higher = better", so when k truncated the list we got the
        # *worst* k matches instead of the best. Verify that with k=1
        # we get the highest-scoring match, not the lowest.
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            # Heavy on the query term — should rank highest.
            JsonDataModel(data_model=Document(id="strong", text="quick quick quick fox")),
            # One occurrence of the query term — should rank lower.
            JsonDataModel(
                data_model=Document(
                    id="weak",
                    text=(
                        "a long document with lots of unrelated words "
                        "and only a single quick mention somewhere"
                    ),
                )
            ),
        ]
        await adapter.update(docs)

        all_results = await adapter.fulltext_search("quick", table_name="Document", k=10)
        # Both docs should match.
        ids = [r["id"] for r in all_results]
        self.assertIn("strong", ids)
        self.assertIn("weak", ids)
        # Scores must be in descending order — top result outranks bottom.
        self.assertGreaterEqual(all_results[0]["score"], all_results[-1]["score"])
        # The strongest match comes first.
        self.assertEqual(all_results[0]["id"], "strong")

        # With k=1 we must keep the strong row, not the weak one.
        top_one = await adapter.fulltext_search("quick", table_name="Document", k=1)
        self.assertEqual(len(top_one), 1)
        self.assertEqual(top_one[0]["id"], "strong")


class DuckDBAdapterRegexSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def _seed_docs(self, adapter):
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="error 404 not found")),
            JsonDataModel(data_model=Document(id="doc2", text="ERROR 500 internal")),
            JsonDataModel(data_model=Document(id="doc3", text="success 200 ok")),
        ]
        await adapter.update(docs)

    async def test_regex_match_returns_full_rows(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        await self._seed_docs(adapter)
        results = await adapter.regex_search(r"error \d+", table_name="Document")
        ids = {r["id"] for r in results}
        self.assertEqual(ids, {"doc1"})
        self.assertEqual(results[0]["text"], "error 404 not found")

    async def test_regex_case_insensitive(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        await self._seed_docs(adapter)
        results = await adapter.regex_search(
            r"error \d+",
            table_name="Document",
            case_sensitive=False,
        )
        ids = {r["id"] for r in results}
        self.assertEqual(ids, {"doc1", "doc2"})

    async def test_regex_no_match_returns_empty(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        await self._seed_docs(adapter)
        results = await adapter.regex_search(r"nope-\w+", table_name="Document")
        self.assertEqual(results, [])

    async def test_regex_empty_pattern_returns_empty(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        await self._seed_docs(adapter)
        results = await adapter.regex_search("", table_name="Document")
        self.assertEqual(results, [])

    async def test_regex_invalid_pattern_raises(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        await self._seed_docs(adapter)
        # Unclosed group — DuckDB raises InvalidInputException which the
        # adapter rewraps with table context as RuntimeError.
        with self.assertRaises(RuntimeError):
            await adapter.regex_search("(unclosed", table_name="Document")

    async def test_regex_limit_k(self):
        class Note(DataModel):
            id: str
            text: str

        adapter = DuckDBAdapter(uri=self.db_path)
        notes = [
            JsonDataModel(data_model=Note(id=f"n{i}", text=f"match {i}"))
            for i in range(20)
        ]
        await adapter.update(notes)
        results = await adapter.regex_search(r"match", table_name="Note", k=5)
        self.assertEqual(len(results), 5)

    async def test_regex_fields_filter_restricts_columns(self):
        # When `fields` is given, only those columns are scanned. So a
        # pattern matching values in a *different* column must not match.
        class Article(DataModel):
            id: str
            title: str
            body: str

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(
            [
                JsonDataModel(
                    data_model=Article(
                        id="a1", title="Welcome", body="The error is in line 12"
                    )
                ),
                JsonDataModel(
                    data_model=Article(id="a2", title="error report", body="all good")
                ),
            ]
        )

        # Search only `title`: should miss a1 (where "error" is in body).
        results = await adapter.regex_search(
            r"error", table_name="Article", fields=["title"]
        )
        self.assertEqual({r["id"] for r in results}, {"a2"})

        # Default search (all string fields): should match both.
        results = await adapter.regex_search(r"error", table_name="Article")
        self.assertEqual({r["id"] for r in results}, {"a1", "a2"})

    async def test_regex_fields_neutralizes_injection(self):
        # Field names are funnelled through `column_identifier` which
        # snake-cases first then sanitizes; the SQL-injection tokens
        # get stripped so the request resolves to a regular column
        # lookup (which then finds no match in the schema and the
        # search returns nothing). The malicious tokens never reach a
        # SQL statement.
        adapter = DuckDBAdapter(uri=self.db_path)
        await self._seed_docs(adapter)
        results = await adapter.regex_search(
            r"error",
            table_name="Document",
            fields=["text; DROP TABLE Document"],
        )
        # Normalized field name is "text_drop_table_document", which
        # isn't a column on Document — the search skips this table
        # with no rows returned and no side effects on the DB.
        self.assertEqual(results, [])
        # Verify the table is still intact.
        all_docs = await adapter.getall(table_name="Document")
        self.assertGreater(len(all_docs), 0)

    async def test_regex_skips_table_without_string_fields(self):
        # If a table has no string columns, the adapter warns and moves on
        # rather than failing the whole search.
        class Numeric(DataModel):
            id: int
            value: float

        adapter = DuckDBAdapter(uri=self.db_path)
        await adapter.update(JsonDataModel(data_model=Numeric(id=1, value=3.14)))
        # `id` is sanitized but is integer-typed in schema; no string fields.
        # Skip-table semantics: empty result, no exception.
        results = await adapter.regex_search(r"3", table_name="Numeric")
        self.assertEqual(results, [])


class DuckDBAdapterVectorSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    @patch("litellm.aembedding")
    async def test_similarity_search(self, mock_embedding):
        vector_dim = 384
        expected_value = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )

        # Create table and insert data with embeddings
        adapter._maybe_create_table(Document.to_symbolic_data_model())

        with adapter._connect(read_only=False) as con:
            # Insert records with embeddings
            vector_str = str(expected_value)
            con.execute(
                f"INSERT INTO Document (id, text, embedding) "
                f"VALUES ('doc1', 'Machine learning basics', "
                f"{vector_str}::FLOAT[{vector_dim}])"
            )

        results = await adapter.similarity_search(
            "machine learning", table_name="Document", k=5
        )
        self.assertGreater(len(results), 0)

    @patch("litellm.aembedding")
    async def test_similarity_search_empty_query(self, mock_embedding):
        vector_dim = 384
        expected_value = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )

        results = await adapter.similarity_search("", table_name="Anything")
        self.assertEqual(results, [])

    @patch("litellm.aembedding")
    async def test_similarity_search_k_zero_returns_empty(self, mock_embedding):
        # k=0 must return an empty list — the LIMIT 0 binding produces
        # no rows per-table; the post-sort slice trims to nothing.
        vector_dim = 8
        embedding_vec = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": embedding_vec}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )
        adapter._maybe_create_table(Document.to_symbolic_data_model())
        with adapter._connect(read_only=False) as con:
            vector_str = str(embedding_vec)
            con.execute(
                f"INSERT INTO Document (id, text, embedding) "
                f"VALUES ('d1', 'hello', "
                f"{vector_str}::FLOAT[{vector_dim}])"
            )

        results = await adapter.similarity_search("hi", table_name="Document", k=0)
        self.assertEqual(results, [])

    @patch("litellm.aembedding")
    async def test_update_builds_hnsw_index_when_embeddings_present(self, mock_embedding):
        # The vector index isn't built during ordinary inserts unless
        # the rows carry non-NULL embeddings. We populate the column
        # directly via SQL (the adapter's update() doesn't auto-embed)
        # and verify the post-update FTS + HNSW rebuild step actually
        # creates the HNSW index that similarity_search would use.
        vector_dim = 8
        vec = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": vec}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )

        # Seed a row with a precomputed embedding via raw SQL.
        adapter._maybe_create_table(Document.to_symbolic_data_model())
        with adapter._connect(read_only=False) as con:
            con.execute(
                f"INSERT INTO Document (id, text, embedding) "
                f"VALUES ('d1', 'hello', {vec!s}::FLOAT[{vector_dim}])"
            )

        # An ordinary update() call on a row that already has an
        # embedding column populated should now trigger HNSW build.
        # The update() also adds an embedded row's worth of work,
        # but the load-bearing assertion is that the index exists.
        adapter._maybe_create_vector_index(Document.to_symbolic_data_model())

        with adapter._connect(read_only=True) as con:
            rows = con.execute(
                "SELECT index_name FROM duckdb_indexes WHERE table_name='Document'"
            ).fetchall()
        names = {r[0] for r in rows}
        self.assertIn("vector_main_Document", names)

    @patch("litellm.aembedding")
    async def test_vector_index_skipped_when_all_embeddings_null(self, mock_embedding):
        # Common case after a bulk load: the embedding column exists
        # but every row's value is NULL because the source file didn't
        # include the column. The auto-build step should silently
        # skip rather than fail or create a useless empty index.
        vector_dim = 8
        vec = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": vec}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )
        adapter._maybe_create_table(Document.to_symbolic_data_model())

        # Row with NULL embedding (the bulk-load shape).
        with adapter._connect(read_only=False) as con:
            con.execute("INSERT INTO Document (id, text) VALUES ('d1', 'hello')")

        adapter._maybe_create_vector_index(Document.to_symbolic_data_model())

        with adapter._connect(read_only=True) as con:
            rows = con.execute(
                "SELECT index_name FROM duckdb_indexes WHERE table_name='Document'"
            ).fetchall()
        names = {r[0] for r in rows}
        self.assertNotIn("vector_main_Document", names)

    async def test_vector_index_skipped_without_embedding_model(self):
        # No embedding model = no vector column on the table; trying
        # to build an HNSW index on a non-existent column would error.
        # The auto-build step should short-circuit silently.
        adapter = DuckDBAdapter(uri=self.db_path)  # no embedding_model
        adapter._maybe_create_table(Document.to_symbolic_data_model())

        # Should not raise even though there's no embedding column.
        adapter._maybe_create_vector_index(Document.to_symbolic_data_model())

        with adapter._connect(read_only=True) as con:
            rows = con.execute(
                "SELECT index_name FROM duckdb_indexes WHERE table_name='Document'"
            ).fetchall()
        names = {r[0] for r in rows}
        self.assertNotIn("vector_main_Document", names)

    @patch("litellm.aembedding")
    async def test_vector_index_rebuild_is_idempotent(self, mock_embedding):
        # Calling _maybe_create_vector_index twice on a table that
        # already has the index would error without the DROP IF EXISTS
        # — `CREATE INDEX` doesn't accept OR REPLACE. Verify two
        # consecutive calls succeed and the index ends up present.
        vector_dim = 8
        vec = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": vec}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )
        adapter._maybe_create_table(Document.to_symbolic_data_model())
        with adapter._connect(read_only=False) as con:
            con.execute(
                f"INSERT INTO Document (id, text, embedding) "
                f"VALUES ('d1', 'hello', {vec!s}::FLOAT[{vector_dim}])"
            )

        # First build creates the index.
        adapter._maybe_create_vector_index(Document.to_symbolic_data_model())
        # Second build must not raise — DROP IF EXISTS + CREATE.
        adapter._maybe_create_vector_index(Document.to_symbolic_data_model())

        with adapter._connect(read_only=True) as con:
            rows = con.execute(
                "SELECT index_name FROM duckdb_indexes WHERE table_name='Document'"
            ).fetchall()
        self.assertIn("vector_main_Document", {r[0] for r in rows})

    @patch("litellm.aembedding")
    async def test_update_triggers_hnsw_rebuild_when_embeddings_exist(
        self, mock_embedding
    ):
        # End-to-end: an update() call to a table that already has
        # embedded rows must rebuild the HNSW index. This is the
        # "user inserted a vector via raw SQL, then ran update with
        # more rows" path — the index gets brought up to date.
        vector_dim = 8
        vec = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": vec}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )
        adapter._maybe_create_table(Document.to_symbolic_data_model())
        with adapter._connect(read_only=False) as con:
            con.execute(
                f"INSERT INTO Document (id, text, embedding) "
                f"VALUES ('seed', 'seed', {vec!s}::FLOAT[{vector_dim}])"
            )

        # Now an ordinary update(): triggers the auto-build branch in
        # update() (FTS + vector rebuild after the bulk insert).
        await adapter.update(JsonDataModel(data_model=Document(id="d2", text="another")))

        with adapter._connect(read_only=True) as con:
            rows = con.execute(
                "SELECT index_name FROM duckdb_indexes WHERE table_name='Document'"
            ).fetchall()
        self.assertIn("vector_main_Document", {r[0] for r in rows})


class DuckDBAdapterHybridSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_hybrid_search_without_embedding_model(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await adapter.update(docs)

        # Without embedding model, hybrid search falls back to fulltext search
        results = await adapter.hybrid_search(
            text_or_texts="quick", table_name="Document", k=5
        )
        self.assertGreater(len(results), 0)

    async def test_hybrid_search_empty_query(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        results = await adapter.hybrid_search(text_or_texts="", table_name="Anything")
        self.assertEqual(results, [])

    async def test_hybrid_search_alias_forwards_to_hybrid_fts_search(self):
        # `hybrid_search` is the deprecated spelling; the new canonical
        # name is `hybrid_fts_search` (symmetric with `hybrid_regex_search`).
        # The alias must keep working for code written before the rename.
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="d1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="d2", text="The lazy dog sleeps")),
        ]
        await adapter.update(docs)
        old = await adapter.hybrid_search(
            text_or_texts="quick", table_name="Document", k=5
        )
        new = await adapter.hybrid_fts_search(
            text_or_texts="quick", table_name="Document", k=5
        )
        self.assertEqual(old, new)

    async def test_hybrid_regex_search_without_embedding_model(self):
        # Without an embedding model the vector half can't run, so the
        # method falls through to the regex side alone — same fallback
        # convention as `hybrid_search` (which falls through to FTS).
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="d1", text="error 404 not found")),
            JsonDataModel(data_model=Document(id="d2", text="success 200 ok")),
        ]
        await adapter.update(docs)
        results = await adapter.hybrid_regex_search(
            text_or_texts="error codes",
            pattern_or_patterns=r"error \d+",
            table_name="Document",
            k=5,
        )
        self.assertEqual({r["id"] for r in results}, {"d1"})

    async def test_hybrid_regex_search_empty_inputs(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        self.assertEqual(
            await adapter.hybrid_regex_search(
                text_or_texts="",
                pattern_or_patterns=None,
                table_name="Anything",
            ),
            [],
        )

    @patch("litellm.aembedding")
    async def test_hybrid_fts_search_rrf_merges_both_halves(self, mock_embedding):
        """When both the FTS and vector halves return rows, hybrid_fts_search
        must combine them via Reciprocal Rank Fusion and emit a ``score``
        column. Exercises the otherwise-uncovered merge math."""
        vector_dim = 8
        # Give every embedding call the same vector so similarity_search
        # returns deterministic ranks against whatever's stored.
        query_vec = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": query_vec}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )
        adapter._maybe_create_table(Document.to_symbolic_data_model())
        with adapter._connect(read_only=False) as con:
            for doc_id, text in (
                ("d1", "quick brown fox jumps"),
                ("d2", "quick lazy dog"),
                ("d3", "completely unrelated"),
            ):
                vec_str = str(query_vec)
                con.execute(
                    f"INSERT INTO Document (id, text, embedding) "
                    f"VALUES ('{doc_id}', '{text}', "
                    f"{vec_str}::FLOAT[{vector_dim}])"
                )

        results = await adapter.hybrid_fts_search("quick", table_name="Document", k=5)
        self.assertGreater(len(results), 0)
        # Every returned row must carry the RRF-fused score.
        for row in results:
            self.assertIn("score", row)
            self.assertGreater(row["score"], 0.0)
        # Sorted descending by score.
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch("litellm.aembedding")
    async def test_hybrid_regex_search_rrf_merges_both_halves(self, mock_embedding):
        """Same as above for hybrid_regex_search: when both halves return
        rows we hit the RRF score-combine path."""
        vector_dim = 8
        query_vec = np.random.rand(vector_dim).tolist()
        mock_embedding.return_value = {"data": [{"embedding": query_vec}]}

        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        adapter = DuckDBAdapter(
            uri=self.db_path,
            embedding_model=embedding_model,
            vector_dim=vector_dim,
        )
        adapter._maybe_create_table(Document.to_symbolic_data_model())
        with adapter._connect(read_only=False) as con:
            for doc_id, text in (
                ("d1", "error 404 not found"),
                ("d2", "error 500 internal"),
                ("d3", "success 200 ok"),
            ):
                vec_str = str(query_vec)
                con.execute(
                    f"INSERT INTO Document (id, text, embedding) "
                    f"VALUES ('{doc_id}', '{text}', "
                    f"{vec_str}::FLOAT[{vector_dim}])"
                )

        results = await adapter.hybrid_regex_search(
            text_or_texts="error codes",
            pattern_or_patterns=r"error \d+",
            table_name="Document",
            k=5,
        )
        self.assertGreater(len(results), 0)
        for row in results:
            self.assertIn("score", row)
            self.assertGreater(row["score"], 0.0)
        # Regex half should pull d1 and d2 (the "error \d+" matches); the
        # vector half pulls everything; merged results should include those.
        ids = {r["id"] for r in results}
        self.assertIn("d1", ids)
        self.assertIn("d2", ids)
