# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Cross-adapter parity suite.

One set of contract tests, run against **every** SQL/vector ``DatabaseAdapter``
listed in ``ADAPTERS``, so the concrete adapters (DuckDB, LanceDB, ...) stay
behavior-compatible. Each test loops over the registry with ``subTest`` and uses
only the public adapter API. Add a new adapter by appending one entry to
``ADAPTERS``.
"""

import os
import shutil
import tempfile
from typing import List
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import database_adapters
from synalinks.src.knowledge_bases.database_adapters.duckdb_adapter import DuckDBAdapter
from synalinks.src.knowledge_bases.database_adapters.lancedb_adapter import LanceDBAdapter
from synalinks.src.modules.embedding_models import EmbeddingModel

# (name, concrete adapter class, uri scheme, file extension)
ADAPTERS = [
    ("DuckDBAdapter", DuckDBAdapter, "duckdb", "db"),
    ("LanceDBAdapter", LanceDBAdapter, "lancedb", "lance"),
]


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


class Article(DataModel):
    id: str = Field(description="Article ID")
    text: str = Field(description="Body text")


class DatabaseAdapterParityTest(testing.TestCase):
    """Run the same contract against each concrete adapter in ``ADAPTERS``."""

    def _make(self, cls, scheme, ext, **kwargs):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        adapter = cls(uri=f"{scheme}://{os.path.join(d, 'kb.' + ext)}", **kwargs)
        # DuckDB holds an exclusive file lock for the connection's lifetime.
        if hasattr(adapter, "close"):
            self.addCleanup(adapter.close)
        return adapter

    # -- data ops --------------------------------------------------------------

    async def test_update_get_getall(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                ids = await a.update(
                    [
                        Customer(
                            id="C1", name="Alice", country="USA"
                        ).to_json_data_model(),
                        Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                    ]
                )
                self.assertEqual(ids, ["C1", "C2"])
                got = await a.get("C1", table_name="Customer")
                self.assertEqual(got.get("name"), "Alice")
                self.assertIsNone(await a.get("nope", table_name="Customer"))
                self.assertEqual(len(await a.getall(table_name="Customer", limit=10)), 2)

    async def test_get_list_preserves_order_and_gaps(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    [
                        Customer(
                            id="C1", name="Alice", country="USA"
                        ).to_json_data_model(),
                        Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                    ]
                )
                got = await a.get(["C2", "missing", "C1"], table_name="Customer")
                self.assertEqual(got[0].get("name"), "Bob")
                self.assertIsNone(got[1])
                self.assertEqual(got[2].get("name"), "Alice")

    async def test_update_is_upsert(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    Customer(id="C1", name="Alice", country="USA").to_json_data_model()
                )
                await a.update(
                    Customer(
                        id="C1", name="Alice Smith", country="USA"
                    ).to_json_data_model()
                )
                rows = await a.getall(table_name="Customer", limit=10)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].get("name"), "Alice Smith")

    async def test_update_requires_primary_key(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                bad = JsonDataModel(
                    json={"name": "X", "country": "Y"}, schema=Customer.get_schema()
                )
                with self.assertRaises(ValueError):
                    await a.update(bad)

    async def test_delete(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    [
                        Customer(
                            id="C1", name="Alice", country="USA"
                        ).to_json_data_model(),
                        Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                    ]
                )
                self.assertEqual(await a.delete("C1", table_name="Customer"), 1)
                self.assertEqual(len(await a.getall(table_name="Customer", limit=10)), 1)

    async def test_drop_table(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    Customer(id="C1", name="Alice", country="USA").to_json_data_model()
                )
                self.assertTrue(await a.drop_table("Customer"))
                self.assertFalse(await a.drop_table("Customer"))

    async def test_get_symbolic_data_models(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                titles = {m.get_schema()["title"] for m in a.get_symbolic_data_models()}
                self.assertIn("Customer", titles)

    async def test_metric_vocabulary_is_shared(self):
        # Both adapters accept the same canonical metric names and reject
        # anything else, so swapping adapters never changes the `metric=` value.
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                for metric in ("l2sq", "cosine", "ip"):
                    a = self._make(cls, scheme, ext, metric=metric)
                    self.assertEqual(a.metric, metric)
                with self.assertRaises(ValueError):
                    self._make(cls, scheme, ext, metric="not-a-metric")

    async def test_wipe_database(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    Customer(id="C1", name="Alice", country="USA").to_json_data_model()
                )
                a.wipe_database()
                self.assertEqual(a.get_symbolic_data_models(), [])

    async def test_json_columns_round_trip(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Note])
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

    # -- sql -------------------------------------------------------------------

    async def test_sql_aggregate(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    [
                        Customer(
                            id="C1", name="Alice", country="USA"
                        ).to_json_data_model(),
                        Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                        Customer(
                            id="C3", name="Carol", country="USA"
                        ).to_json_data_model(),
                    ]
                )
                rows = await a.sql(
                    "SELECT country, count(*) AS n FROM Customer GROUP BY country "
                    "ORDER BY country"
                )
                self.assertEqual(
                    rows, [{"country": "UK", "n": 1}, {"country": "USA", "n": 2}]
                )

    async def test_sql_csv_output(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    Customer(id="C1", name="Alice", country="USA").to_json_data_model()
                )
                out = await a.sql("SELECT id FROM Customer", output_format="csv")
                self.assertIsInstance(out, str)
                self.assertIn("C1", out)

    # -- fulltext / regex ------------------------------------------------------

    async def test_fulltext_search(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    [
                        Customer(
                            id="C1", name="Alice", country="USA"
                        ).to_json_data_model(),
                        Customer(
                            id="C2", name="Bob Quick", country="UK"
                        ).to_json_data_model(),
                    ]
                )
                results = await a.fulltext_search("Quick", table_name="Customer", k=5)
                self.assertEqual([r["id"] for r in results], ["C2"])
                self.assertIn("score", results[0])

    async def test_fulltext_scores_are_normalized(self):
        # BM25 magnitudes differ across engines, so fulltext_search rescales each
        # result set to [0, 1]: best hit -> 1.0, worst -> 0.0. The bounds and the
        # ranking must agree across adapters (middle values stay engine-specific).
        docs = [
            Article(id="D1", text="the quick brown fox jumps"),
            Article(id="D2", text="the fox the fox the fox runs fast"),
            Article(id="D3", text="a lazy dog sleeps all day"),
            Article(id="D4", text="quick quick rabbits hop"),
        ]
        per_adapter = {}
        for name, cls, scheme, ext in ADAPTERS:
            a = self._make(cls, scheme, ext, data_models=[Article])
            await a.update([d.to_json_data_model() for d in docs])
            results = await a.fulltext_search("quick fox", table_name="Article", k=5)
            per_adapter[name] = [(r["id"], r["score"]) for r in results]

        for name, ranked in per_adapter.items():
            with self.subTest(adapter=name):
                scores = [s for _, s in ranked]
                self.assertGreater(len(ranked), 1)
                self.assertTrue(all(0.0 <= s <= 1.0 for s in scores), scores)
                self.assertEqual(scores[0], 1.0)  # best hit
                self.assertEqual(scores[-1], 0.0)  # worst hit
                self.assertEqual(scores, sorted(scores, reverse=True))  # descending

        rankings = {name: [i for i, _ in r] for name, r in per_adapter.items()}
        ref = rankings["DuckDBAdapter"]
        for name, ids in rankings.items():
            with self.subTest(adapter=name):
                self.assertEqual(ids, ref)  # identical ranking across engines

    async def test_minmax_normalize_scores_edges(self):
        from synalinks.src.knowledge_bases.adapters_utils import minmax_normalize_scores

        # Spread maps to [0, 1] endpoints; single row / all-equal -> 1.0; None -> 0.0.
        spread = minmax_normalize_scores([{"score": 4.0}, {"score": 2.0}, {"score": 3.0}])
        self.assertEqual([r["score"] for r in spread], [1.0, 0.0, 0.5])

        single = minmax_normalize_scores([{"score": 7.0}])
        self.assertEqual(single[0]["score"], 1.0)

        all_equal = minmax_normalize_scores([{"score": 5.0}, {"score": 5.0}])
        self.assertEqual([r["score"] for r in all_equal], [1.0, 1.0])

        with_none = minmax_normalize_scores([{"score": 2.0}, {"score": None}])
        self.assertEqual([r["score"] for r in with_none], [1.0, 0.0])

    async def test_regex_search(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Customer])
                await a.update(
                    [
                        Customer(
                            id="C1", name="Alice", country="USA"
                        ).to_json_data_model(),
                        Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                    ]
                )
                results = await a.regex_search(
                    "^Bob$", table_name="Customer", fields=["name"], k=5
                )
                self.assertEqual([r["id"] for r in results], ["C2"])

    # -- file loaders ----------------------------------------------------------

    async def test_from_json(self):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                d = tempfile.mkdtemp()
                self.addCleanup(shutil.rmtree, d, True)
                path = os.path.join(d, "people.json")
                with open(path, "w") as f:
                    f.write(
                        '[{"id": "p1", "name": "Ada"}, {"id": "p2", "name": "Linus"}]'
                    )

                a = self._make(cls, scheme, ext)
                model = await a.from_json(path, table_name="People")
                self.assertEqual(model.get_schema()["title"], "People")
                self.assertEqual(len(await a.getall(table_name="People", limit=10)), 2)

    # -- vector / hybrid (mocked embeddings) -----------------------------------

    @patch("litellm.aembedding")
    async def test_similarity_search(self, mock_embedding):
        mock_embedding.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                em = EmbeddingModel(model="ollama/all-minilm")
                a = self._make(cls, scheme, ext, embedding_model=em, vector_dim=3)
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
                self.assertEqual(results[0]["id"], "D0")  # nearest to [1, 0, 0]
                self.assertIn("score", results[0])
                scores = [r["score"] for r in results]
                self.assertEqual(scores, sorted(scores))  # ascending distance

    @patch("litellm.aembedding")
    async def test_similarity_requires_embedding_model(self, mock_embedding):
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                a = self._make(cls, scheme, ext, data_models=[Document])
                with self.assertRaises(ValueError):
                    await a.similarity_search("q", table_name="Document", k=3)

    @patch("litellm.aembedding")
    async def test_similarity_search_honours_metric(self, mock_embedding):
        # Same data + query, three metrics, two adapters: the configured metric
        # must drive the ranking identically on both. ``best_dir`` wins on
        # distance/angle metrics; ``best_mag`` wins on inner product because its
        # larger magnitude maximises the dot product with the query.
        mock_embedding.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        docs = [
            Document(id="best_dir", text="a", embedding=[1.0, 0.0, 0.0]),
            Document(id="best_mag", text="b", embedding=[2.0, 2.0, 2.0]),
            Document(id="mid", text="c", embedding=[0.0, 1.0, 0.0]),
        ]
        expected_top = {"l2sq": "best_dir", "cosine": "best_dir", "ip": "best_mag"}
        for name, cls, scheme, ext in ADAPTERS:
            for metric, top in expected_top.items():
                with self.subTest(adapter=name, metric=metric):
                    em = EmbeddingModel(model="ollama/all-minilm")
                    a = self._make(
                        cls, scheme, ext, embedding_model=em, vector_dim=3, metric=metric
                    )
                    await a.update([d.to_json_data_model() for d in docs])
                    results = await a.similarity_search("q", table_name="Document", k=3)
                    self.assertEqual(results[0]["id"], top)

    @patch("litellm.aembedding")
    async def test_similarity_scores_match_across_adapters(self, mock_embedding):
        # Not just the ranking — the actual per-row score must agree numerically
        # across adapters for every metric (canonical units: squared-L2 /
        # cosine-distance / negative inner product).
        mock_embedding.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        docs = [
            Document(id="A", text="a", embedding=[1.0, 0.0, 0.0]),
            Document(id="B", text="b", embedding=[2.0, 2.0, 2.0]),
            Document(id="C", text="c", embedding=[0.0, 1.0, 0.0]),
        ]
        # Canonical scores the two adapters must both produce.
        expected = {
            "l2sq": {"A": 0.0, "C": 2.0, "B": 9.0},
            "cosine": {"A": 0.0, "B": 0.42265, "C": 1.0},
            "ip": {"A": -1.0, "B": -2.0, "C": 0.0},
        }
        for metric, want in expected.items():
            with self.subTest(metric=metric):
                per_adapter = {}
                for name, cls, scheme, ext in ADAPTERS:
                    em = EmbeddingModel(model="ollama/all-minilm")
                    a = self._make(
                        cls, scheme, ext, embedding_model=em, vector_dim=3, metric=metric
                    )
                    await a.update([d.to_json_data_model() for d in docs])
                    results = await a.similarity_search("q", table_name="Document", k=3)
                    per_adapter[name] = {r["id"]: round(r["score"], 5) for r in results}
                want = {k: round(v, 5) for k, v in want.items()}
                for name, scored in per_adapter.items():
                    self.assertEqual(scored, want, f"{name} metric={metric}")

    @patch("litellm.aembedding")
    async def test_hybrid_fts_search(self, mock_embedding):
        mock_embedding.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        for name, cls, scheme, ext in ADAPTERS:
            with self.subTest(adapter=name):
                em = EmbeddingModel(model="ollama/all-minilm")
                a = self._make(cls, scheme, ext, embedding_model=em, vector_dim=3)
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
                self.assertIn("D0", {r["id"] for r in results})
                self.assertIn("score", results[0])


class AdapterRoutingTest(testing.TestCase):
    def test_scheme_routing(self):
        self.assertIs(database_adapters.get("lancedb://x"), LanceDBAdapter)
        self.assertIs(database_adapters.get("duckdb://x"), DuckDBAdapter)
        self.assertIs(database_adapters.get(None), DuckDBAdapter)
