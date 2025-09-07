# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Literal
from typing import Union
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import Entities
from synalinks.src.backend import Entity
from synalinks.src.backend import Relation
from synalinks.src.backend import Relations
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.knowledge_bases.knowledge_base import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.knowledge.embedding import Embedding
from synalinks.src.modules.knowledge.update_knowledge import UpdateKnowledge
from synalinks.src.programs import Program


class Document(Entity):
    label: Literal["Document"]
    text: str


class Chunk(Entity):
    label: Literal["Chunk"]
    text: str


class DocumentsAndChunks(Entities):
    entities: List[Union[Document, Chunk]]


class IsPartOf(Relation):
    subj: Document
    label: Literal["IsPartOf"]
    obj: Union[Chunk, Document]


class DocumentRelations(Relations):
    relations: List[IsPartOf]


class UpdateKnowledgeTest(testing.TestCase):
    @patch("litellm.aembedding")
    async def test_update_knowledge_single_entity(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")

        knowledge_base = KnowledgeBase(
            uri="neo4j://localhost:7687",
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=True,
        )

        x0 = Input(data_model=Document)
        x1 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(x0)
        x2 = await UpdateKnowledge(
            knowledge_base=knowledge_base,
        )(x1)

        program = Program(
            inputs=x0,
            outputs=x2,
            name="test_update_knowledge",
            description="test_update_knowledge",
        )

        input_doc = Document(
            label="Document",
            text="test document",
        )

        result = await program(input_doc)
        self.assertNotEqual(result, None)

    @patch("litellm.aembedding")
    async def test_update_knowledge_single_relation(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        knowledge_base = KnowledgeBase(
            uri="neo4j://localhost:7687",
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=True,
        )

        x0 = Input(data_model=IsPartOf)
        x1 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(x0)
        x2 = await UpdateKnowledge(
            knowledge_base=knowledge_base,
        )(x1)

        program = Program(
            inputs=x0,
            outputs=x2,
            name="test_update_knowledge",
            description="test_update_knowledge",
        )

        doc = Document(
            label="Document",
            text="test document",
        )

        chunk = Chunk(
            label="Chunk",
            text="test chunk",
        )

        input_rel = IsPartOf(
            subj=doc,
            label="IsPartOf",
            obj=chunk,
        )
        result = await program(input_rel)
        self.assertNotEqual(result, None)

    @patch("litellm.aembedding")
    async def test_update_knowledge_entities(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        knowledge_base = KnowledgeBase(
            uri="neo4j://localhost:7687",
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=True,
        )

        i0 = Input(data_model=DocumentsAndChunks)

        x1 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(i0)
        x2 = await UpdateKnowledge(
            knowledge_base=knowledge_base,
        )(x1)

        program = Program(
            inputs=i0,
            outputs=x2,
            name="test_update_knowledge",
            description="test_update_knowledge",
        )

        inputs = DocumentsAndChunks(
            entities=[
                Document(label="Document", text="test document 1"),
                Document(label="Document", text="test document 2"),
            ]
        )

        result = await program(inputs)
        self.assertNotEqual(result, None)

    @patch("litellm.aembedding")
    async def test_update_knowledge_relations(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        knowledge_base = KnowledgeBase(
            uri="neo4j://localhost:7687",
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=True,
        )

        x0 = Input(data_model=DocumentRelations)
        x1 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(x0)
        x2 = await UpdateKnowledge(
            knowledge_base=knowledge_base,
        )(x1)

        program = Program(
            inputs=x0,
            outputs=x2,
            name="test_update_knowledge",
            description="test_update_knowledge",
        )

        rel1 = IsPartOf(
            subj=Document(
                label="Document",
                text="test document 1",
            ),
            label="IsPartOf",
            obj=Document(
                label="Document",
                text="test document 2",
            ),
        )

        rel2 = IsPartOf(
            subj=Document(
                label="Document",
                text="test document 3",
            ),
            label="IsPartOf",
            obj=Document(
                label="Document",
                text="test document 4",
            ),
        )

        inputs = DocumentRelations(relations=[rel1, rel2])

        result = await program(inputs)
        self.assertNotEqual(result, None)
