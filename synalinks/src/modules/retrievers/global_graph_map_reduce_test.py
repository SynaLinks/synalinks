# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Tests for `GlobalGraphMapReduce`.

`KnowledgeBase.community_graph_search` is stubbed on a real KB instance so the
tests are deterministic and independent of the graph backend; only
``litellm.acompletion`` is mocked. The mock routes on the rendered prompt: map
calls carry a community subgraph (its entity labels), the reduce call carries
the ``community_answers`` list, so the parallel map order never matters.
"""

import json
import os
import tempfile
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend.pydantic.knowledge import Entity
from synalinks.src.backend.pydantic.knowledge import KnowledgeGraph
from synalinks.src.backend.pydantic.knowledge import KnowledgeGraphs
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.modules.retrievers.global_graph_map_reduce import GlobalGraphMapReduce
from synalinks.src.programs import Program


class Doc(DataModel):
    id: str = Field(description="id")
    text: str = Field(description="text")


class Query(DataModel):
    question: str = Field(description="The user question")


class FinalAnswer(DataModel):
    answer: str = Field(description="The final answer")


class FinalReport(DataModel):
    report: str = Field(description="The final report")


def _lm_router(scores, reduce_prompts=None, reduce_content='{"answer": "final"}'):
    """Mock LM: score map calls by the entity label found in the prompt,
    answer the reduce call with a fixed final answer."""

    def _fake(*args, **kwargs):
        text = json.dumps(kwargs.get("messages", []))
        if "community_answers" in text:
            if reduce_prompts is not None:
                reduce_prompts.append(text)
            return {"choices": [{"message": {"content": reduce_content}}]}
        for label, score in scores.items():
            if label in text:
                content = json.dumps(
                    {"partial_answer": f"about {label}", "relevance_score": score}
                )
                return {"choices": [{"message": {"content": content}}]}
        raise AssertionError(f"unexpected map prompt: {text}")

    return _fake


class GlobalGraphMapReduceTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "m.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _kb(self, community_labels=("Alice", "Xenia")):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Doc])

        async def _search(**kwargs):
            return KnowledgeGraphs(
                knowledge_graphs=[
                    KnowledgeGraph(entities=[Entity(label=label)], relations=[])
                    for label in community_labels
                ]
            )

        kb.community_graph_search = _search
        return kb

    def _module(self, kb, **kwargs):
        # Most tests exercise the structured path; the default (no
        # schema/data_model → ChatMessage) has its own dedicated tests.
        kwargs.setdefault("data_model", FinalAnswer)
        lm = LanguageModel(model="ollama/mistral")
        return GlobalGraphMapReduce(
            knowledge_base=kb, language_model=lm, name="m", **kwargs
        )

    def test_final_schema_defaults_to_none_chat_message(self):
        kb = self._kb()
        mod = self._module(kb, data_model=None)
        self.assertIsNone(mod.schema)
        self.assertIsNone(mod.reduce_generator.schema)
        lm = LanguageModel(model="ollama/mistral")
        mod2 = GlobalGraphMapReduce(
            knowledge_base=kb, language_model=lm, data_model=FinalReport, name="m2"
        )
        self.assertIn("report", mod2.reduce_generator.schema["properties"])

    def test_validates_score_threshold(self):
        kb = self._kb()
        with self.assertRaisesRegex(ValueError, "score_threshold"):
            self._module(kb, score_threshold=-1.0)

    async def test_map_reduce_end_to_end_orders_best_first(self):
        kb = self._kb()
        reduce_prompts = []
        mod = self._module(kb, return_inputs=True)
        with patch(
            "litellm.acompletion",
            side_effect=_lm_router({"Alice": 90, "Xenia": 10}, reduce_prompts),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="what are the main themes")
            )
        self.assertEqual(result.get("answer"), "final")
        # return_inputs=True: the inputs flow through.
        self.assertEqual(result.get("question"), "what are the main themes")
        # Both partial answers reached the reduce step, best first.
        self.assertEqual(len(reduce_prompts), 1)
        text = reduce_prompts[0]
        self.assertIn("about Alice", text)
        self.assertIn("about Xenia", text)
        self.assertLess(text.index("about Alice"), text.index("about Xenia"))

    async def test_score_threshold_filters_partial_answers(self):
        kb = self._kb()
        reduce_prompts = []
        mod = self._module(kb, score_threshold=50.0)
        with patch(
            "litellm.acompletion",
            side_effect=_lm_router({"Alice": 90, "Xenia": 10}, reduce_prompts),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="themes?")
            )
        self.assertEqual(result.get("answer"), "final")
        self.assertIn("about Alice", reduce_prompts[0])
        self.assertNotIn("about Xenia", reduce_prompts[0])

    async def test_return_community_answers(self):
        kb = self._kb()
        mod = self._module(kb, return_community_answers=True)
        with patch(
            "litellm.acompletion",
            side_effect=_lm_router({"Alice": 90, "Xenia": 10}),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="themes?")
            )
        answers = result.get("community_answers")
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[0]["partial_answer"], "about Alice")
        self.assertEqual(answers[0]["relevance_score"], 90)

    async def test_default_reduce_returns_chat_message(self):
        kb = self._kb()
        mod = self._module(kb, data_model=None)
        with patch(
            "litellm.acompletion",
            side_effect=_lm_router(
                {"Alice": 90, "Xenia": 10}, reduce_content="the final answer"
            ),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="themes?")
            )
        self.assertEqual(result.get("content"), "the final answer")

    async def test_no_communities_returns_none(self):
        kb = self._kb(community_labels=())
        mod = self._module(kb)
        with patch(
            "litellm.acompletion",
            side_effect=AssertionError("no LM call expected"),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="themes?")
            )
        self.assertIsNone(result)
