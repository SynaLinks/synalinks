# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile
from typing import List
from typing import Literal

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend.pydantic.knowledge import Entities
from synalinks.src.backend.pydantic.knowledge import Entity
from synalinks.src.backend.pydantic.knowledge import Relation
from synalinks.src.backend.pydantic.knowledge import Relations
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.knowledge.update_knowledge import UpdateKnowledge
from synalinks.src.programs import Program


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The content of the document")


class Person(Entity):
    label: Literal["Person"]
    name: str = Field(description="The person's name")


class Knows(Relation):
    label: Literal["Knows"]
    subj: Person
    obj: Person


class People(Entities):
    entities: List[Person] = Field(description="The people")


class Friendships(Relations):
    relations: List[Knows] = Field(description="The friendships")


class UpdateKnowledgeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_update_knowledge_single(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )

        x0 = Input(data_model=Document)
        x1 = await UpdateKnowledge(knowledge_base=knowledge_base)(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="test_update_knowledge",
            description="test_update_knowledge",
        )

        input_doc = Document(id="doc1", text="test document")
        result = await program(input_doc)

        self.assertIsNotNone(result)

        # Verify document was stored
        retrieved = await knowledge_base.get("doc1", table_name="Document")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get_json()["text"], "test document")

    async def test_update_knowledge_multiple(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )

        update_module = UpdateKnowledge(knowledge_base=knowledge_base)

        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="first document")),
            JsonDataModel(data_model=Document(id="doc2", text="second document")),
        ]

        results = await update_module(docs)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)

        # Verify both documents were stored
        retrieved1 = await knowledge_base.get("doc1", table_name="Document")
        retrieved2 = await knowledge_base.get("doc2", table_name="Document")
        self.assertIsNotNone(retrieved1)
        self.assertIsNotNone(retrieved2)

    async def test_update_knowledge_upsert(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )

        update_module = UpdateKnowledge(knowledge_base=knowledge_base)

        # Insert first version
        doc1 = JsonDataModel(data_model=Document(id="doc1", text="original text"))
        await update_module(doc1)

        # Update with new text
        doc1_updated = JsonDataModel(data_model=Document(id="doc1", text="updated text"))
        await update_module(doc1_updated)

        # Verify update
        retrieved = await knowledge_base.get("doc1", table_name="Document")
        self.assertEqual(retrieved.get_json()["text"], "updated text")

    async def test_update_knowledge_none_input(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )

        update_module = UpdateKnowledge(knowledge_base=knowledge_base)
        result = await update_module(None)
        self.assertIsNone(result)

    def test_update_knowledge_serialization(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
        )

        update_module = UpdateKnowledge(
            knowledge_base=knowledge_base,
            name="test_update",
            description="Test update module",
        )

        config = update_module.get_config()
        cloned_module = UpdateKnowledge.from_config(config)

        self.assertEqual(cloned_module.name, "test_update")
        self.assertEqual(cloned_module.description, "Test update module")


class GraphWrapperUpdateTest(testing.TestCase):
    """The `Entities` / `Relations` wrapper branches must hand the KB typed
    instances (via `get_nested_entity_list`), not raw JSON items."""

    async def test_update_relations_wrapper(self):
        knowledge_base = KnowledgeBase(
            graph_uri="ladybug://:memory:",
            entity_models=[Person],
            relation_models=[Knows],
        )

        x0 = Input(data_model=Friendships)
        x1 = await UpdateKnowledge(knowledge_base=knowledge_base)(x0)
        program = Program(inputs=x0, outputs=x1, name="rel_wrapper_update")

        result = await program(
            Friendships(
                relations=[
                    Knows(
                        label="Knows",
                        subj=Person(label="Person", name="Alice"),
                        obj=Person(label="Person", name="Bob"),
                    )
                ]
            )
        )
        self.assertIsNotNone(result)
        rows = await knowledge_base.cypher(
            "MATCH (a:Person)-[:Knows]->(b:Person) RETURN a.name AS a, b.name AS b"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["a"], rows[0]["b"]), ("Alice", "Bob"))

    async def test_update_entities_wrapper(self):
        knowledge_base = KnowledgeBase(
            graph_uri="ladybug://:memory:",
            entity_models=[Person],
            relation_models=[Knows],
        )

        x0 = Input(data_model=People)
        x1 = await UpdateKnowledge(knowledge_base=knowledge_base)(x0)
        program = Program(inputs=x0, outputs=x1, name="ent_wrapper_update")

        result = await program(
            People(
                entities=[
                    Person(label="Person", name="Alice"),
                    Person(label="Person", name="Bob"),
                ]
            )
        )
        self.assertIsNotNone(result)
        rows = await knowledge_base.cypher("MATCH (p:Person) RETURN p.name AS name")
        self.assertEqual(sorted(r["name"] for r in rows), ["Alice", "Bob"])

    async def test_relations_wrapper_with_embeddings_upserts_endpoints(self):
        """With an embedding model the adapter resolves endpoints against
        EXISTING nodes and silently no-ops when they're absent; the module
        must upsert the endpoint entities first (relations-only seeding)."""
        from unittest.mock import patch

        from synalinks.src.modules.embedding_models import EmbeddingModel
        from synalinks.src.modules.knowledge.embed_knowledge import EmbedKnowledge

        # Distinct (near-orthogonal) vectors per text; identical vectors
        # would make the adapter's near-duplicate detection collapse the
        # two endpoints onto one node.
        basis = {"Alice": [1.0, 0.0, 0.0], "Bob": [0.0, 1.0, 0.0]}

        async def fake_embed(*args, **kwargs):
            texts = kwargs.get("input") or args[1]
            return {
                "data": [
                    {"embedding": basis.get(text, [0.0, 0.0, 1.0])} for text in texts
                ]
            }

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")
        knowledge_base = KnowledgeBase(
            graph_uri="ladybug://:memory:",
            entity_models=[Person],
            relation_models=[Knows],
            embedding_model=embedding_model,
            metric="cosine",
        )

        x0 = Input(data_model=Friendships)
        x1 = await EmbedKnowledge(
            embedding_model=embedding_model,
            in_mask=["name"],
        )(x0)
        x2 = await UpdateKnowledge(knowledge_base=knowledge_base)(x1)
        program = Program(inputs=x0, outputs=x2, name="embedded_rel_update")

        with patch("litellm.aembedding", side_effect=fake_embed):
            result = await program(
                Friendships(
                    relations=[
                        Knows(
                            label="Knows",
                            subj=Person(label="Person", name="Alice"),
                            obj=Person(label="Person", name="Bob"),
                        )
                    ]
                )
            )
        self.assertIsNotNone(result)
        rows = await knowledge_base.cypher("MATCH (p:Person) RETURN p.name AS name")
        self.assertEqual(sorted(r["name"] for r in rows), ["Alice", "Bob"])
        edges = await knowledge_base.cypher(
            "MATCH (a:Person)-[:Knows]->(b:Person) RETURN a.name AS a, b.name AS b"
        )
        self.assertEqual(len(edges), 1)
