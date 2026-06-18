# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Tests for `EntityRegexSearch`, focused on the infer-the-label behavior.

The knowledge base's graph enumeration (`get_symbolic_entities`) and search
(`entity_regex_search`) are stubbed on a real KB instance, so the tests are
deterministic and independent of the graph backend; only ``litellm.acompletion``
is mocked, to return the generated query (including the inferred ``entity_label``).
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
from synalinks.src.modules.retrievers.entity_regex_search import EntityRegexSearch
from synalinks.src.programs import Program


class Doc(DataModel):
    id: str = Field(description="id")
    text: str = Field(description="text")


class Query(DataModel):
    question: str = Field(description="The user question")


class _FakeModel:
    """Minimal stand-in for a symbolic model exposing ``get_schema().title``."""

    def __init__(self, title):
        self._title = title

    def get_schema(self):
        return {"title": self._title}


def _lm_returns(content):
    def _fake(*args, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    return _fake


class EntityRegexSearchInferTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "e.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _kb(self, captured=None):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Doc])
        # Stub the graph enumeration + search so no real graph adapter is needed.
        kb.get_symbolic_entities = lambda: [_FakeModel("Person"), _FakeModel("City")]

        async def _search(*args, label=None, **kwargs):
            if captured is not None:
                captured["label"] = label
                captured["args"] = args
            return [{"id": "p1", "name": "Alice", "label": label}]

        kb.entity_regex_search = _search
        return kb

    def test_infers_label_enum_from_kb(self):
        kb = self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = EntityRegexSearch(knowledge_base=kb, language_model=lm, name="e")
        self.assertIsNone(mod.label)
        props = mod.query_generator.schema["properties"]
        self.assertIn("entity_label", props)
        self.assertEqual(props["entity_label"]["enum"], ["City", "Person"])

    def test_provided_label_skips_inference(self):
        kb = self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = EntityRegexSearch(
            knowledge_base=kb, language_model=lm, label="Person", name="e"
        )
        self.assertEqual(mod.label, "Person")
        self.assertNotIn("entity_label", mod.query_generator.schema["properties"])

    async def test_call_uses_inferred_label(self):
        captured = {}
        kb = self._kb(captured)
        lm = LanguageModel(model="ollama/mistral")
        mod = EntityRegexSearch(knowledge_base=kb, language_model=lm, name="e")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"regex_search": "alice", "entity_label": "City"}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="where does alice live")
            )
        # the LM-inferred label was the one passed to the KB search
        self.assertEqual(captured["label"], "City")
        self.assertEqual(result.get("result")[0]["id"], "p1")
