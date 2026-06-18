# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Tests for `RelationSimilaritySearch`, focused on infer-the-label behavior.

The KB's relation enumeration (`get_symbolic_relations`) and search
(`relation_similarity_search`) are stubbed on a real KB instance, so the tests
are deterministic and backend-independent; only ``litellm.acompletion`` is
mocked, to return the generated query (including the inferred ``relation_label``).
"""

import os
import tempfile
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.modules.retrievers.relation_similarity_search import (
    RelationSimilaritySearch,
)
from synalinks.src.programs import Program


class Doc(DataModel):
    id: str = Field(description="id")
    text: str = Field(description="text")


class Query(DataModel):
    question: str = Field(description="The user question")


class _FakeModel:
    def __init__(self, title):
        self._title = title

    def get_schema(self):
        return {"title": self._title}


def _lm_returns(content):
    def _fake(*args, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    return _fake


class RelationSimilaritySearchInferTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "r.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _kb(self, captured=None):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Doc])
        kb.get_symbolic_relations = lambda: [_FakeModel("LivesIn"), _FakeModel("Knows")]

        async def _search(queries, label=None, **kwargs):
            if captured is not None:
                captured["label"] = label
                captured["queries"] = queries
            return [{"subj": "Alice", "obj": "Paris", "label": label}]

        kb.relation_similarity_search = _search
        return kb

    def test_infers_label_enum_from_kb(self):
        kb = self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = RelationSimilaritySearch(knowledge_base=kb, language_model=lm, name="r")
        self.assertIsNone(mod.label)
        props = mod.query_generator.schema["properties"]
        self.assertIn("relation_label", props)
        self.assertEqual(props["relation_label"]["enum"], ["Knows", "LivesIn"])

    def test_provided_label_skips_inference(self):
        kb = self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = RelationSimilaritySearch(
            knowledge_base=kb, language_model=lm, label="LivesIn", name="r"
        )
        self.assertEqual(mod.label, "LivesIn")
        self.assertNotIn("relation_label", mod.query_generator.schema["properties"])

    async def test_call_uses_inferred_label(self):
        captured = {}
        kb = self._kb(captured)
        lm = LanguageModel(model="ollama/mistral")
        mod = RelationSimilaritySearch(knowledge_base=kb, language_model=lm, name="r")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns(
                '{"similarity_search": ["lives"], "relation_label": "LivesIn"}'
            ),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="who lives where")
            )
        self.assertEqual(captured["label"], "LivesIn")
        self.assertEqual(result.get("result")[0]["subj"], "Alice")
