# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile
import uuid
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import DuckDBAdapter
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import (
    sanitize_identifier,
)
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import (
    sanitize_properties,
)


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
        props = {"valid_key": "value", "AnotherKey": 123}
        result = sanitize_properties(props)
        self.assertEqual(result, {"valid_key": "value", "AnotherKey": 123})

    def test_sanitize_properties_invalid_key(self):
        props = {"invalid-key": "value"}
        with self.assertRaises(ValueError):
            sanitize_properties(props)


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

    def test_init_with_duckdb_uri_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            adapter = DuckDBAdapter(uri=f"duckdb://{db_path}")
            self.assertEqual(adapter.uri, db_path)

    @patch("litellm.aembedding")
    def test_init_with_embedding_model(self, mock_embedding):
        expected_value = np.random.rand(384).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            embedding_model = EmbeddingModel(model="ollama/all-minilm")
            adapter = DuckDBAdapter(
                uri=db_path,
                embedding_model=embedding_model,
            )
            self.assertEqual(adapter.vector_dim, 384)
            self.assertEqual(adapter.embedding_model, embedding_model)


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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DuckDBAdapterInstallExtensionsTest(testing.TestCase):
    """Verify _install_extensions skips INSTALL when the extension is already
    present in duckdb_extensions(). Prevents unnecessary network calls to
    extensions.duckdb.org on every KnowledgeBase construction."""

    def _make_adapter(self, installed, embedding_model=None):
        executed = []
        adapter = DuckDBAdapter.__new__(DuckDBAdapter)
        adapter.embedding_model = embedding_model
        adapter._connect = lambda read_only=False: _FakeConn(installed, executed)
        adapter._install_extensions()
        return executed

    def test_skips_install_fts_when_already_installed(self):
        executed = self._make_adapter(installed=["fts"])
        install_calls = [s for s in executed if "INSTALL fts" in s]
        load_calls = [s for s in executed if "LOAD fts" in s]
        self.assertEqual(len(install_calls), 0)
        self.assertEqual(len(load_calls), 1)

    def test_runs_install_fts_when_not_installed(self):
        executed = self._make_adapter(installed=[])
        install_calls = [s for s in executed if "INSTALL fts" in s]
        load_calls = [s for s in executed if "LOAD fts" in s]
        self.assertEqual(len(install_calls), 1)
        self.assertEqual(len(load_calls), 1)

    def test_skips_install_vss_when_already_installed(self):
        executed = self._make_adapter(installed=["fts", "vss"], embedding_model=object())
        install_fts = [s for s in executed if "INSTALL fts" in s]
        install_vss = [s for s in executed if "INSTALL vss" in s]
        load_vss = [s for s in executed if "LOAD vss" in s]
        self.assertEqual(len(install_fts), 0)
        self.assertEqual(len(install_vss), 0)
        self.assertEqual(len(load_vss), 1)

    def test_runs_install_vss_when_not_installed(self):
        executed = self._make_adapter(installed=["fts"], embedding_model=object())
        install_vss = [s for s in executed if "INSTALL vss" in s]
        load_vss = [s for s in executed if "LOAD vss" in s]
        self.assertEqual(len(install_vss), 1)
        self.assertEqual(len(load_vss), 1)

    def test_skips_vss_entirely_without_embedding_model(self):
        executed = self._make_adapter(installed=[], embedding_model=None)
        vss_calls = [s for s in executed if "vss" in s]
        self.assertEqual(len(vss_calls), 0)


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
        result = await adapter.get("Alice", [Person.to_symbolic_data_model()])
        self.assertEqual(result.get_json()["age"], 35)

    async def test_get_record(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        person = JsonDataModel(data_model=Person(name="Alice", age=30))
        await adapter.update(person)

        result = await adapter.get("Alice", [Person.to_symbolic_data_model()])
        self.assertIsNotNone(result)
        self.assertEqual(result.get_json()["name"], "Alice")
        self.assertEqual(result.get_json()["age"], 30)

    async def test_get_nonexistent_record(self):
        class Person(DataModel):
            name: str
            age: int

        adapter = DuckDBAdapter(uri=self.db_path)
        adapter._maybe_create_table(Person.to_symbolic_data_model())

        result = await adapter.get("NonExistent", [Person.to_symbolic_data_model()])
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

        results = await adapter.getall(Person.to_symbolic_data_model())
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

        results = await adapter.getall(Person.to_symbolic_data_model(), limit=3, offset=2)
        self.assertEqual(len(results), 3)


class DuckDBAdapterQueryTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_query_simple(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        result = await adapter.query("SELECT 1 as value")
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

        result = await adapter.query(
            "SELECT * FROM Person WHERE age > ?",
            params=[26],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "Alice")


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

        retrieved = await adapter.get("test1", [ModelWithDict.to_symbolic_data_model()])
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

        retrieved = await adapter.get("acc1", [Account.to_symbolic_data_model()])
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

        results = await adapter.fulltext_search(
            "quick", [Document.to_symbolic_data_model()], k=10
        )
        self.assertGreater(len(results), 0)

    async def test_fulltext_search_returns_full_records(self):
        """Test that fulltext_search returns all fields, not just id and score."""
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await adapter.update(docs)

        results = await adapter.fulltext_search(
            "quick", [Document.to_symbolic_data_model()], k=10
        )

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
        results = await adapter.fulltext_search("")
        self.assertEqual(results, [])

    async def test_fulltext_search_list_queries(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await adapter.update(docs)

        results = await adapter.fulltext_search(
            ["quick", "lazy"], [Document.to_symbolic_data_model()], k=10
        )
        self.assertGreater(len(results), 0)


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
            "machine learning", [Document.to_symbolic_data_model()], k=5
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

        results = await adapter.similarity_search("")
        self.assertEqual(results, [])


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
            "quick", [Document.to_symbolic_data_model()], k=5
        )
        self.assertGreater(len(results), 0)

    async def test_hybrid_search_empty_query(self):
        adapter = DuckDBAdapter(uri=self.db_path)
        results = await adapter.hybrid_search("")
        self.assertEqual(results, [])
