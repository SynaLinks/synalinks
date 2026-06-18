# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Tests for `PathRegexSearch`, focused on inferring the endpoint labels.

Both endpoint labels (``subj_label`` / ``obj_label``) are independently
inferred from the KB's entity labels when not provided. The KB's entity
enumeration (`get_symbolic_entities`) and search (`path_regex_search`) are
stubbed on a real KB instance; only ``litellm.acompletion`` is mocked.
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
from synalinks.src.modules.retrievers.path_regex_search import PathRegexSearch
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


class PathRegexSearchInferTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "p.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _kb(self, captured=None):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Doc])
        kb.get_symbolic_entities = lambda: [_FakeModel("Person"), _FakeModel("City")]

        async def _search(*args, subj_label=None, obj_label=None, **kwargs):
            if captured is not None:
                captured["subj_label"] = subj_label
                captured["obj_label"] = obj_label
            return [{"path": [subj_label, obj_label]}]

        kb.path_regex_search = _search
        return kb

    def test_infers_both_endpoint_labels(self):
        kb = self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = PathRegexSearch(knowledge_base=kb, language_model=lm, name="p")
        self.assertIsNone(mod.subj_label)
        self.assertIsNone(mod.obj_label)
        props = mod.query_generator.schema["properties"]
        self.assertEqual(props["subj_label"]["enum"], ["City", "Person"])
        self.assertEqual(props["obj_label"]["enum"], ["City", "Person"])

    def test_partial_inference_only_missing_endpoint(self):
        kb = self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = PathRegexSearch(
            knowledge_base=kb, language_model=lm, subj_label="Person", name="p"
        )
        self.assertEqual(mod.subj_label, "Person")
        self.assertIsNone(mod.obj_label)
        props = mod.query_generator.schema["properties"]
        self.assertNotIn("subj_label", props)  # fixed -> not inferred
        self.assertIn("obj_label", props)  # missing -> inferred

    async def test_call_uses_inferred_labels(self):
        captured = {}
        kb = self._kb(captured)
        lm = LanguageModel(model="ollama/mistral")
        mod = PathRegexSearch(knowledge_base=kb, language_model=lm, name="p")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns(
                '{"subj_regex_search": "^alice$", '
                '"obj_regex_search": "^paris$", '
                '"subj_label": "Person", "obj_label": "City"}'
            ),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="how is alice connected to paris")
            )
        self.assertEqual(captured["subj_label"], "Person")
        self.assertEqual(captured["obj_label"], "City")
        self.assertEqual(result.get("result")[0]["path"], ["Person", "City"])
