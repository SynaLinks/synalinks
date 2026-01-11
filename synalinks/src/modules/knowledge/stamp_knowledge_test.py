# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os
import tempfile
from datetime import datetime

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.knowledge.stamp_knowledge import StampKnowledge
from synalinks.src.modules.knowledge.update_knowledge import UpdateKnowledge
from synalinks.src.programs import Program


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The content of the document")


class StampedDocument(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The content of the document")
    created_at: datetime = Field(description="The creation timestamp")


class StampKnowledgeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def test_stamp_knowledge_single(self):
        x0 = Input(data_model=Document)
        x1 = await StampKnowledge()(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="test_stamp_knowledge",
            description="test_stamp_knowledge",
        )

        input_doc = Document(id="doc1", text="test document")
        result = await program(input_doc)

        self.assertIsNotNone(result)
        self.assertTrue("created_at" in result)
        self.assertTrue("id" in result)
        self.assertTrue("text" in result)
        self.assertEqual(result.get("id"), "doc1")
        self.assertEqual(result.get("text"), "test document")

    async def test_stamp_knowledge_multiple(self):
        stamp_module = StampKnowledge()

        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="first document")),
            JsonDataModel(data_model=Document(id="doc2", text="second document")),
        ]

        results = await stamp_module(docs)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)

        for result in results:
            self.assertTrue("created_at" in result)

    async def test_stamp_knowledge_none_input(self):
        stamp_module = StampKnowledge()
        result = await stamp_module(None)
        self.assertIsNone(result)

    def test_stamp_knowledge_serialization(self):
        stamp_module = StampKnowledge(
            name="test_stamp",
            description="Test stamp module",
        )

        config = stamp_module.get_config()
        cloned_module = StampKnowledge.from_config(config)

        self.assertEqual(cloned_module.name, "test_stamp")
        self.assertEqual(cloned_module.description, "Test stamp module")

    async def test_stamp_knowledge_contains_created_at_key(self):
        x0 = Input(data_model=Document)
        x1 = await StampKnowledge()(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="test_stamp_knowledge",
            description="test_stamp_knowledge",
        )

        input_doc = Document(id="doc1", text="test document")
        result = await program(input_doc)

        # Test the new string key contains functionality
        self.assertTrue("created_at" in result)
        self.assertFalse("nonexistent_key" in result)

    async def test_stamp_and_store_knowledge(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[StampedDocument],
        )

        x0 = Input(data_model=Document)
        x1 = await StampKnowledge()(x0)
        x2 = await UpdateKnowledge(knowledge_base=knowledge_base)(x1)

        program = Program(
            inputs=x0,
            outputs=x2,
            name="test_stamp_and_store",
            description="test_stamp_and_store",
        )

        input_doc = Document(id="doc1", text="test document")
        result = await program(input_doc)

        # Verify the stamped and stored result has all expected fields
        self.assertIsNotNone(result)
        self.assertTrue("created_at" in result)
        self.assertTrue("id" in result)
        self.assertTrue("text" in result)
        self.assertEqual(result.get("id"), "doc1")
        self.assertEqual(result.get("text"), "test document")
        self.assertIsNotNone(result.get("created_at"))

        # Retrieve stored data using get method
        symbolic_models = knowledge_base.get_symbolic_data_models()
        self.assertTrue(len(symbolic_models) > 0)

        retrieved = await knowledge_base.get("doc1", symbolic_models)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.get_json()["id"], "doc1")
        self.assertEqual(retrieved.get_json()["text"], "test document")
        self.assertIn("created_at", retrieved.get_json())

    async def test_stamp_and_store_multiple(self):
        knowledge_base = KnowledgeBase(
            uri=self.db_path,
            data_models=[StampedDocument],
        )

        stamp_module = StampKnowledge()
        update_module = UpdateKnowledge(knowledge_base=knowledge_base)

        docs = [
            JsonDataModel(data_model=Document(id="doc1", text="first document")),
            JsonDataModel(data_model=Document(id="doc2", text="second document")),
        ]

        stamped = await stamp_module(docs)
        results = await update_module(stamped)

        # Verify the results have all expected fields
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)

        for i, result in enumerate(results):
            self.assertTrue("created_at" in result)
            self.assertTrue("id" in result)
            self.assertTrue("text" in result)
            self.assertIsNotNone(result.get("created_at"))

        # Retrieve stored data using get method
        symbolic_models = knowledge_base.get_symbolic_data_models()
        self.assertTrue(len(symbolic_models) > 0)

        retrieved1 = await knowledge_base.get("doc1", symbolic_models)
        retrieved2 = await knowledge_base.get("doc2", symbolic_models)
        self.assertIsNotNone(retrieved1)
        self.assertIsNotNone(retrieved2)
        self.assertIn("created_at", retrieved1.get_json())
        self.assertIn("created_at", retrieved2.get_json())
