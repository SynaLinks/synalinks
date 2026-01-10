# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os
import tempfile
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.knowledge.retrieve_knowledge import RetrieveKnowledge


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The content of the document")


class Query(DataModel):
    question: str = Field(description="The user question")


class RetrieveKnowledgeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    @patch("litellm.acompletion")
    async def test_retrieve_knowledge(self, mock_completion):
        # Mock the LLM response to return a search query
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tables": ["Document"], "search": ["quick brown fox"]}'
                        )
                    }
                }
            ]
        }

        knowledge_base = KnowledgeBase(
            uri="duckdb://" + self.db_path,
            data_models=[Document],
        )

        # Store some documents
        docs = [
            JsonDataModel(
                data_model=Document(id="doc1", text="The quick brown fox jumps")
            ),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await knowledge_base.update(docs)

        language_model = LanguageModel(model="openai/gpt-4o-mini")

        retrieve_module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            k=5,
            name="test_retriever",
        )

        query = JsonDataModel(data_model=Query(question="Find documents about foxes"))
        result = await retrieve_module(query)

        self.assertIsNotNone(result)
        self.assertIn("result", result.get_json())

    async def test_retrieve_knowledge_none_input(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )

        language_model = LanguageModel(model="openai/gpt-4o-mini")

        retrieve_module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            name="test_retriever",
        )

        result = await retrieve_module(None)
        self.assertIsNone(result)

    def test_retrieve_knowledge_default_instructions(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )

        language_model = LanguageModel(model="openai/gpt-4o-mini")

        retrieve_module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            name="test_retriever",
        )

        self.assertIn("Document", retrieve_module.instructions)
        self.assertIn("search", retrieve_module.instructions)
