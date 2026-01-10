# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.modules import Input
from synalinks.src.modules.knowledge.embed_knowledge import EmbedKnowledge
from synalinks.src.programs import Program


class Document(DataModel):
    title: str = Field(description="The document title")
    text: str = Field(description="The document content")


class EmbedKnowledgeTest(testing.TestCase):
    @patch("litellm.aembedding")
    async def test_embed_knowledge_single_document(self, mock_embedding):
        expected_value = np.random.rand(1024).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="openai/text-embedding-3-small",
        )

        i0 = Input(data_model=Document)
        x0 = await EmbedKnowledge(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
            name="test_embed_knowledge",
            description="test_embed_knowledge",
        )

        input_doc = Document(
            title="My Title",
            text="test document content",
        )

        result = await program(input_doc)
        self.assertTrue(result is not None)
        self.assertTrue(len(result.get("embedding")) > 0)

    @patch("litellm.aembedding")
    async def test_embed_knowledge_with_out_mask(self, mock_embedding):
        expected_value = np.random.rand(1024).tolist()
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="openai/text-embedding-3-small",
        )

        i0 = Input(data_model=Document)
        x0 = await EmbedKnowledge(
            embedding_model=embedding_model,
            out_mask=["title"],  # Remove title, embed only text
        )(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
            name="test_embed_knowledge",
            description="test_embed_knowledge",
        )

        input_doc = Document(
            title="My Title",
            text="test document content",
        )

        result = await program(input_doc)
        self.assertTrue(result is not None)
        self.assertTrue(len(result.get("embedding")) > 0)
