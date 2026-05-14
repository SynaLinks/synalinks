# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules.knowledge.retrieve_knowledge import RetrieveKnowledge
from synalinks.src.modules.language_models import LanguageModel


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

        language_model = LanguageModel(model="ollama/mistral")

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

        language_model = LanguageModel(model="ollama/mistral")

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

        language_model = LanguageModel(model="ollama/mistral")

        retrieve_module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            name="test_retriever",
        )

        self.assertIn("Document", retrieve_module.instructions)
        self.assertIn("search", retrieve_module.instructions)

    def test_retrieve_knowledge_regex_instructions_differ(self):
        # Regex mode must instruct the LM to produce patterns, not
        # natural-language phrases — wrong guidance here silently
        # degrades retrieval quality.
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )
        language_model = LanguageModel(model="ollama/mistral")

        nl_module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            search_type="hybrid",
            name="nl_retriever",
        )
        regex_module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            search_type="regex",
            name="regex_retriever",
        )
        self.assertIn("natural language", nl_module.instructions)
        self.assertNotIn("regular-expression", nl_module.instructions)
        self.assertIn("regular-expression", regex_module.instructions)
        self.assertNotIn("natural language", regex_module.instructions)

    @patch("litellm.acompletion")
    async def test_retrieve_knowledge_regex(self, mock_completion):
        # LM emits a regex pattern in the `search` field; the module
        # dispatches to `kb.regex_search` and returns rows that match.
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tables": ["Document"], "search": ["error \\\\d+"]}'
                        )
                    }
                }
            ]
        }

        knowledge_base = KnowledgeBase(
            uri="duckdb://" + self.db_path,
            data_models=[Document],
        )
        await knowledge_base.update(
            [
                JsonDataModel(
                    data_model=Document(id="d1", text="error 404 not found")
                ),
                JsonDataModel(
                    data_model=Document(id="d2", text="ERROR 500 internal")
                ),
                JsonDataModel(
                    data_model=Document(id="d3", text="success 200 ok")
                ),
            ]
        )

        language_model = LanguageModel(model="ollama/mistral")

        # Case-sensitive: matches lowercase "error 404" only.
        retrieve_module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            search_type="regex",
            k=5,
            name="regex_retriever",
        )
        query = JsonDataModel(data_model=Query(question="any error codes?"))
        result = await retrieve_module(query)
        rows = result.get_json().get("result", [])
        ids = {r["id"] for r in rows}
        self.assertEqual(ids, {"d1"})

        # Case-insensitive: also matches the uppercase ERROR 500.
        retrieve_module_ci = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            search_type="regex",
            case_sensitive=False,
            k=5,
            name="regex_retriever_ci",
        )
        result_ci = await retrieve_module_ci(query)
        rows_ci = result_ci.get_json().get("result", [])
        ids_ci = {r["id"] for r in rows_ci}
        self.assertEqual(ids_ci, {"d1", "d2"})

    def test_retrieve_knowledge_legacy_hybrid_alias(self):
        # `"hybrid"` is the historical spelling of `"hybrid_fts"`. It must
        # keep working: the module silently translates to the canonical
        # name so old code (and saved configs) deserialise unchanged.
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )
        language_model = LanguageModel(model="ollama/mistral")
        module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            search_type="hybrid",
            name="legacy_retriever",
        )
        self.assertEqual(module.search_type, "hybrid_fts")

    def test_retrieve_knowledge_invalid_search_type(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )
        language_model = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            RetrieveKnowledge(
                knowledge_base=knowledge_base,
                language_model=language_model,
                data_models=[Document.to_symbolic_data_model()],
                search_type="unknown",
                name="bad_retriever",
            )

    def test_retrieve_knowledge_hybrid_regex_instructions(self):
        # The hybrid_regex mode must mention BOTH the search list (vector)
        # and the patterns list (regex) in the LM instructions — otherwise
        # the LM has no idea it needs to populate both.
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )
        language_model = LanguageModel(model="ollama/mistral")
        module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            search_type="hybrid_regex",
            name="hr_retriever",
        )
        self.assertIn("`search`", module.instructions)
        self.assertIn("`patterns`", module.instructions)
        self.assertIn("Reciprocal Rank Fusion", module.instructions)

    @patch("litellm.acompletion")
    async def test_retrieve_knowledge_hybrid_regex(self, mock_completion):
        # The LM emits BOTH a `search` list (vector queries) and a
        # `patterns` list (regex patterns); the module dispatches to
        # `kb.hybrid_regex_search` and returns rows from the regex side
        # (no embedding model is configured, so the vector half is
        # skipped and we still get the regex matches back).
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tables": ["Document"], '
                            '"search": ["any error codes"], '
                            '"patterns": ["error \\\\d+"]}'
                        )
                    }
                }
            ]
        }

        knowledge_base = KnowledgeBase(
            uri="duckdb://" + self.db_path,
            data_models=[Document],
        )
        await knowledge_base.update(
            [
                JsonDataModel(
                    data_model=Document(id="d1", text="error 404 not found")
                ),
                JsonDataModel(
                    data_model=Document(id="d2", text="success 200 ok")
                ),
            ]
        )

        language_model = LanguageModel(model="ollama/mistral")

        module = RetrieveKnowledge(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_models=[Document.to_symbolic_data_model()],
            search_type="hybrid_regex",
            k=5,
            name="hr_retriever",
        )
        query = JsonDataModel(data_model=Query(question="any error codes?"))
        result = await module(query)
        rows = result.get_json().get("result", [])
        self.assertEqual({r["id"] for r in rows}, {"d1"})
