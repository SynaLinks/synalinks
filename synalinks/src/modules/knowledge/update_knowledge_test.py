# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os
import tempfile

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.knowledge.update_knowledge import UpdateKnowledge
from synalinks.src.programs import Program


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The content of the document")


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
        retrieved = await knowledge_base.get("doc1", [Document.to_symbolic_data_model()])
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
        retrieved1 = await knowledge_base.get("doc1", [Document.to_symbolic_data_model()])
        retrieved2 = await knowledge_base.get("doc2", [Document.to_symbolic_data_model()])
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
        retrieved = await knowledge_base.get("doc1", [Document.to_symbolic_data_model()])
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
