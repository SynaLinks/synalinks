# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Tests for `GlobalGraphSearch`.

The module runs no language model: it reads the communities materialized at
index time. `KnowledgeBase.community_graph_search` is stubbed on a real KB
instance so the tests are deterministic and independent of the graph backend.
"""

import os
import tempfile

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend.pydantic.knowledge import Entity
from synalinks.src.backend.pydantic.knowledge import KnowledgeGraph
from synalinks.src.backend.pydantic.knowledge import KnowledgeGraphs
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.retrievers.global_graph_search import GlobalGraphSearch
from synalinks.src.programs import Program


class Doc(DataModel):
    id: str = Field(description="id")
    text: str = Field(description="text")


class Query(DataModel):
    question: str = Field(description="The user question")


class GlobalGraphSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "g.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _kb(self, captured=None):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Doc])

        async def _search(**kwargs):
            if captured is not None:
                captured.update(kwargs)
            return KnowledgeGraphs(
                knowledge_graphs=[
                    KnowledgeGraph(
                        entities=[Entity(label="Person")],
                        relations=[],
                    )
                ]
            )

        kb.community_graph_search = _search
        return kb

    def test_validates_k(self):
        kb = self._kb()
        with self.assertRaisesRegex(ValueError, "`k`"):
            GlobalGraphSearch(knowledge_base=kb, k=0, name="g")

    def test_validates_members_per_community(self):
        kb = self._kb()
        with self.assertRaisesRegex(ValueError, "members_per_community"):
            GlobalGraphSearch(knowledge_base=kb, members_per_community=0, name="g")

    async def test_call_passes_params_and_returns_graphs(self):
        captured = {}
        kb = self._kb(captured)
        mod = GlobalGraphSearch(
            knowledge_base=kb,
            node_labels=["Person"],
            rel_labels=["Knows"],
            k=3,
            members_per_community=5,
            name="g",
        )
        inputs = Input(data_model=Query)
        outputs = await mod(inputs)
        result = await Program(inputs=inputs, outputs=outputs)(
            Query(question="what are the main themes")
        )
        self.assertEqual(captured["node_labels"], ["Person"])
        self.assertEqual(captured["rel_labels"], ["Knows"])
        self.assertEqual(captured["k"], 3)
        self.assertEqual(captured["members_per_community"], 5)
        graphs = result.get("knowledge_graphs")
        self.assertEqual(len(graphs), 1)
        self.assertEqual(graphs[0]["entities"][0]["label"], "Person")
        # return_inputs=True (default): the inputs flow through.
        self.assertEqual(result.get("question"), "what are the main themes")

    async def test_call_without_return_inputs(self):
        kb = self._kb()
        mod = GlobalGraphSearch(knowledge_base=kb, return_inputs=False, name="g")
        inputs = Input(data_model=Query)
        outputs = await mod(inputs)
        result = await Program(inputs=inputs, outputs=outputs)(Query(question="themes?"))
        self.assertIsNone(result.get("question"))
        self.assertEqual(len(result.get("knowledge_graphs")), 1)
