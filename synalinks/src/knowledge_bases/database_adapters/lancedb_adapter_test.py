# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import shutil
import tempfile
from typing import List
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.knowledge_bases import database_adapters
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import DuckDBAdapter
from synalinks.src.knowledge_bases.database_adapters.lancedb_adapter import LanceDBAdapter
from synalinks.src.modules.embedding_models import EmbeddingModel


class Customer(DataModel):
    id: str = Field(description="Customer ID")
    name: str = Field(description="Customer name")
    country: str = Field(description="Country")


class Document(DataModel):
    id: str = Field(description="Document ID")
    text: str = Field(description="Document text")
    embedding: List[float] = Field(default=[], description="Vector")


class Note(DataModel):
    id: str = Field(description="Note ID")
    title: str = Field(description="Title")
    tags: List[str] = Field(default=[], description="Tags")
    meta: dict = Field(default={}, description="Arbitrary metadata")


class LanceDBAdapterTest(testing.TestCase):
    def _uri(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        return f"lancedb://{os.path.join(d, 'kb.lance')}"

    def _adapter(self, **kwargs):
        return LanceDBAdapter(uri=self._uri(), **kwargs)

    # -- routing ---------------------------------------------------------------

    def test_scheme_routing(self):
        self.assertIs(database_adapters.get("lancedb://x"), LanceDBAdapter)
        self.assertIs(database_adapters.get("duckdb://x"), DuckDBAdapter)
        self.assertIs(database_adapters.get(None), DuckDBAdapter)

    # -- data ops --------------------------------------------------------------

    async def test_update_get_getall(self):
        a = self._adapter(data_models=[Customer])
        ids = await a.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
            ]
        )
        self.assertEqual(ids, ["C1", "C2"])

        got = await a.get("C1", table_name="Customer")
        self.assertEqual(got.get("name"), "Alice")

        missing = await a.get("nope", table_name="Customer")
        self.assertIsNone(missing)

        rows = await a.getall(table_name="Customer", limit=10)
        self.assertEqual(len(rows), 2)

    async def test_update_is_upsert(self):
        a = self._adapter(data_models=[Customer])
        await a.update(
            Customer(id="C1", name="Alice", country="USA").to_json_data_model()
        )
        await a.update(
            Customer(id="C1", name="Alice Smith", country="USA").to_json_data_model()
        )
        rows = await a.getall(table_name="Customer", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("name"), "Alice Smith")

    async def test_update_requires_primary_key(self):
        from synalinks.src.backend import JsonDataModel

        a = self._adapter(data_models=[Customer])
        bad = JsonDataModel(
            json={"name": "X", "country": "Y"}, schema=Customer.get_schema()
        )
        with self.assertRaises(ValueError):
            await a.update(bad)

    async def test_delete(self):
        a = self._adapter(data_models=[Customer])
        await a.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
            ]
        )
        n = await a.delete("C1", table_name="Customer")
        self.assertEqual(n, 1)
        self.assertEqual(len(await a.getall(table_name="Customer", limit=10)), 1)

    async def test_drop_table(self):
        a = self._adapter(data_models=[Customer])
        await a.update(
            Customer(id="C1", name="Alice", country="USA").to_json_data_model()
        )
        self.assertTrue(await a.drop_table("Customer"))
        self.assertFalse(await a.drop_table("Customer"))

    async def test_get_symbolic_data_models(self):
        a = self._adapter(data_models=[Customer])
        models = a.get_symbolic_data_models()
        titles = {m.get_schema()["title"] for m in models}
        self.assertIn("Customer", titles)

    async def test_json_columns_round_trip(self):
        """object / array-of-... columns survive a store/load round-trip."""
        a = self._adapter(data_models=[Note])
        await a.update(
            Note(
                id="N1",
                title="hi",
                tags=["a", "b"],
                meta={"k": 1, "nested": {"x": True}},
            ).to_json_data_model()
        )
        got = await a.get("N1", table_name="Note")
        self.assertEqual(got.get("tags"), ["a", "b"])
        self.assertEqual(got.get("meta"), {"k": 1, "nested": {"x": True}})

    async def test_wipe_on_start(self):
        uri = self._uri()
        a = LanceDBAdapter(uri=uri, data_models=[Customer])
        await a.update(Customer(id="C1", name="A", country="U").to_json_data_model())
        a2 = LanceDBAdapter(uri=uri, wipe_on_start=True)
        self.assertEqual(a2._db.table_names(), [])

    # -- sql (via DuckDB) ------------------------------------------------------

    async def test_sql_via_duckdb(self):
        a = self._adapter(data_models=[Customer])
        await a.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                Customer(id="C3", name="Carol", country="USA").to_json_data_model(),
            ]
        )
        rows = await a.sql(
            "SELECT country, count(*) AS n FROM Customer GROUP BY country ORDER BY country"
        )
        self.assertEqual(rows, [{"country": "UK", "n": 1}, {"country": "USA", "n": 2}])

    async def test_sql_csv_output(self):
        a = self._adapter(data_models=[Customer])
        await a.update(
            Customer(id="C1", name="Alice", country="USA").to_json_data_model()
        )
        out = await a.sql("SELECT id FROM Customer", output_format="csv")
        self.assertIsInstance(out, str)
        self.assertIn("C1", out)

    # -- fulltext / regex (no embedding) ---------------------------------------

    async def test_fulltext_search(self):
        a = self._adapter(data_models=[Customer])
        await a.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob Quick", country="UK").to_json_data_model(),
            ]
        )
        results = await a.fulltext_search("Quick", table_name="Customer", k=5)
        self.assertEqual([r["id"] for r in results], ["C2"])
        self.assertIn("score", results[0])

    async def test_regex_search(self):
        a = self._adapter(data_models=[Customer])
        await a.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
            ]
        )
        results = await a.regex_search(
            "^Bob$", table_name="Customer", fields=["name"], k=5
        )
        self.assertEqual([r["id"] for r in results], ["C2"])

    # -- file loaders ----------------------------------------------------------

    async def test_from_json(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "people.json")
        with open(path, "w") as f:
            f.write('[{"id": "p1", "name": "Ada"}, {"id": "p2", "name": "Linus"}]')

        a = self._adapter()
        model = await a.from_json(path, table_name="People")
        self.assertEqual(model.get_schema()["title"], "People")
        rows = await a.getall(table_name="People", limit=10)
        self.assertEqual(len(rows), 2)

    # -- vector / hybrid (mocked embeddings) -----------------------------------

    @patch("litellm.aembedding")
    async def test_similarity_search(self, mock_embedding):
        mock_embedding.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        em = EmbeddingModel(model="ollama/all-minilm")
        a = self._adapter(embedding_model=em, vector_dim=3)
        await a.update(
            [
                Document(
                    id="D0", text="exact", embedding=[1.0, 0.0, 0.0]
                ).to_json_data_model(),
                Document(
                    id="D1", text="orthogonal", embedding=[0.0, 1.0, 0.0]
                ).to_json_data_model(),
                Document(
                    id="D2", text="close", embedding=[0.9, 0.1, 0.0]
                ).to_json_data_model(),
            ]
        )
        results = await a.similarity_search("q", table_name="Document", k=3)
        self.assertEqual(results[0]["id"], "D0")  # nearest to [1,0,0]
        self.assertIn("score", results[0])
        # results are ranked by ascending distance
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores))

    @patch("litellm.aembedding")
    async def test_similarity_requires_embedding_model(self, mock_embedding):
        a = self._adapter(data_models=[Document])  # no embedding model
        with self.assertRaises(ValueError):
            await a.similarity_search("q", table_name="Document", k=3)

    @patch("litellm.aembedding")
    async def test_hybrid_fts_search(self, mock_embedding):
        mock_embedding.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        em = EmbeddingModel(model="ollama/all-minilm")
        a = self._adapter(embedding_model=em, vector_dim=3)
        await a.update(
            [
                Document(
                    id="D0", text="quick brown fox", embedding=[1.0, 0.0, 0.0]
                ).to_json_data_model(),
                Document(
                    id="D1", text="lazy dog", embedding=[0.0, 1.0, 0.0]
                ).to_json_data_model(),
            ]
        )
        results = await a.hybrid_fts_search(
            "fox", keywords="fox", table_name="Document", k=5
        )
        ids = {r["id"] for r in results}
        self.assertIn("D0", ids)
        self.assertIn("score", results[0])
