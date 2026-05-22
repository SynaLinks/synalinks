# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import GenericResult
from synalinks.src.backend import JsonDataModel
from synalinks.src.modules.rerankers.rrf_reranker import RRFReranker


def _result(rows):
    return JsonDataModel(json={"result": rows}, schema=GenericResult.get_schema())


class RRFRerankerTest(testing.TestCase):
    async def test_fuses_two_lists_by_id(self):
        # doc "b" is ranked highly in both lists -> should win.
        list_a = _result([{"id": "a"}, {"id": "b"}, {"id": "c"}])
        list_b = _result([{"id": "b"}, {"id": "d"}, {"id": "a"}])
        out = await RRFReranker(id_key="id")([list_a, list_b])
        rows = out.get("result")
        ids = [r["id"] for r in rows]
        self.assertEqual(ids[0], "b")
        # every fused row carries an rrf_score
        self.assertTrue(all("rrf_score" in r for r in rows))
        # b: 1/(60+2) + 1/(60+1); a: 1/(60+1) + 1/(60+3)
        b_score = 1.0 / 62 + 1.0 / 61
        self.assertAlmostEqual(rows[0]["rrf_score"], b_score, places=9)

    async def test_truncates_to_k(self):
        list_a = _result([{"id": str(i)} for i in range(5)])
        out = await RRFReranker(k=2, id_key="id")([list_a])
        self.assertEqual(len(out.get("result")), 2)

    async def test_ignores_none_inputs(self):
        list_a = _result([{"id": "a"}])
        out = await RRFReranker(id_key="id")([None, list_a, None])
        self.assertEqual([r["id"] for r in out.get("result")], ["a"])

    async def test_all_none_returns_none(self):
        out = await RRFReranker()([None, None])
        self.assertIsNone(out)

    async def test_signature_identity_without_id_key(self):
        # No id_key -> identical rows are matched by full-row signature.
        list_a = _result([{"title": "x"}, {"title": "y"}])
        list_b = _result([{"title": "x"}])
        out = await RRFReranker()([list_a, list_b])
        rows = out.get("result")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "x")  # matched in both lists

    async def test_symbolic_trace(self):
        r0 = _result([{"id": "a"}]).to_symbolic_data_model()
        out = await RRFReranker(id_key="id").symbolic_call([r0, r0])
        self.assertIn("result", out.get_schema().get("properties", {}))

    def test_config_round_trip(self):
        mod = RRFReranker(k_rank=42, k=5, id_key="id", name="rrf")
        cfg = mod.get_config()
        mod2 = RRFReranker.from_config(cfg)
        self.assertEqual(mod2.k_rank, 42)
        self.assertEqual(mod2.k, 5)
        self.assertEqual(mod2.id_key, "id")

    def test_invalid_k_rank(self):
        with self.assertRaises(ValueError):
            RRFReranker(k_rank=0)


class RRFRerankerIntegrationTest(testing.TestCase):
    """End-to-end: merge two real searches over a real DuckDB store.

    Only the LM (query generation) is mocked; DuckDB BM25 + regex are
    real. This is the canonical use case — fusing heterogeneous
    retrievers into one ranked list.
    """

    def setUp(self):
        import os
        import tempfile

        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "rrf.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_merges_fulltext_and_regex_results(self):
        from unittest.mock import patch

        import synalinks
        from synalinks.src.backend import DataModel
        from synalinks.src.backend import Field
        from synalinks.src.knowledge_bases import KnowledgeBase
        from synalinks.src.modules.language_models import LanguageModel

        class Document(DataModel):
            id: str = Field(description="doc id")
            text: str = Field(description="doc text")

        class Query(DataModel):
            question: str = Field(description="the user question")

        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        await kb.update(
            [
                JsonDataModel(
                    data_model=Document(id="d1", text="The quick brown fox jumps")
                ),
                JsonDataModel(data_model=Document(id="d2", text="A lazy dog sleeps")),
                JsonDataModel(data_model=Document(id="d3", text="The fox and the hound")),
            ]
        )
        lm = LanguageModel(model="ollama/mistral")

        # One reply carries both query fields; each generator reads its own.
        content = '{"fulltext_search": ["fox"], "regex_search": "fox"}'

        def _fake(*args, **kwargs):
            return {"choices": [{"message": {"content": content}}]}

        inputs = synalinks.Input(data_model=Query)
        fts = await synalinks.FullTextSearch(
            knowledge_base=kb, language_model=lm, data_model=Document, k=5, name="fts"
        )(inputs)
        rgx = await synalinks.RegexSearch(
            knowledge_base=kb, language_model=lm, data_model=Document, k=5, name="rgx"
        )(inputs)
        fused = await RRFReranker(k=10, id_key="id", name="rrf")([fts, rgx])
        program = synalinks.Program(inputs=inputs, outputs=fused, name="rag")

        with patch("litellm.acompletion", side_effect=_fake):
            result = await program(Query(question="tell me about the fox"))

        rows = result.get("result")
        ids = {r["id"] for r in rows}
        self.assertEqual(ids, {"d1", "d3"})  # both fox docs, dog excluded
        self.assertTrue(all("rrf_score" in r for r in rows))
        scores = [r["rrf_score"] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))
