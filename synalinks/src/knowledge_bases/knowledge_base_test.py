# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os
import tempfile
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.knowledge_bases import KnowledgeBase


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The content of the document")


class Chunk(DataModel):
    id: str = Field(description="The chunk id")
    text: str = Field(description="The content of the chunk")
    document_id: str = Field(description="The parent document id")


class KnowledgeBaseTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_knowledge_base(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document, Chunk],
            metric="cosine",
            wipe_on_start=False,
        )

        result = await knowledge_base.query("SELECT 1 as value")
        self.assertEqual(result, [{"value": 1}])

    async def test_knowledge_base_crud(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
            wipe_on_start=False,
        )

        # Insert a document
        doc = Document(id="doc1", text="Hello World")
        result = await knowledge_base.update(JsonDataModel(data_model=doc))
        self.assertEqual(result, "doc1")

        # Retrieve the document
        retrieved = await knowledge_base.get("doc1", [Document.to_symbolic_data_model()])
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get_json()["text"], "Hello World")

    async def test_knowledge_base_fulltext_search(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
            wipe_on_start=False,
        )

        # Insert documents
        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="The quick brown fox")),
            JsonDataModel(data_model=Document(id="doc2", text="The lazy dog sleeps")),
        ]
        await knowledge_base.update(docs)

        # Full-text search
        results = await knowledge_base.fulltext_search(
            "quick", [Document.to_symbolic_data_model()], k=10
        )
        self.assertGreater(len(results), 0)

    def test_knowledge_base_serialization(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document, Chunk],
            metric="cosine",
            wipe_on_start=False,
        )

        config = knowledge_base.get_config()
        cloned_knowledge_base = KnowledgeBase.from_config(config)
        self.assertEqual(
            cloned_knowledge_base.get_config(),
            knowledge_base.get_config(),
        )

    @patch("litellm.aembedding")
    def test_knowledge_base_serialization_with_embedding(self, mock_embedding):
        expected_value = np.random.rand(384).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document, Chunk],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=False,
        )

        config = knowledge_base.get_config()
        cloned_knowledge_base = KnowledgeBase.from_config(config)
        self.assertEqual(
            cloned_knowledge_base.get_config(),
            knowledge_base.get_config(),
        )
