# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.knowledge_bases.graph_database_adapters import GraphDatabaseAdapter


class GraphDatabaseAdapterTest(testing.TestCase):
    """The base class is abstract: every method must raise
    NotImplementedError so concrete adapters are forced to provide one,
    and so the failure message names the offending subclass.
    """

    def _adapter(self):
        return GraphDatabaseAdapter()

    def test_init_stores_attributes(self):
        adapter = GraphDatabaseAdapter(
            uri="ladybug://test",
            metric="cosine",
            name="kg",
        )
        self.assertEqual(adapter.uri, "ladybug://test")
        self.assertEqual(adapter.metric, "cosine")
        self.assertEqual(adapter.name, "kg")
        self.assertEqual(adapter.entity_models, [])
        self.assertEqual(adapter.relation_models, [])

    def test_init_splits_entity_and_relation_models(self):
        from synalinks.src.backend.pydantic.knowledge import Entity
        from synalinks.src.backend.pydantic.knowledge import Relation

        class Doc(Entity):
            pass

        class Knows(Relation):
            pass

        adapter = GraphDatabaseAdapter(
            entity_models=[Doc],
            relation_models=[Knows],
        )
        self.assertEqual(adapter.entity_models, [Doc])
        self.assertEqual(adapter.relation_models, [Knows])

    def test_wipe_database_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "wipe_database"):
            self._adapter().wipe_database()

    async def test_update_entities_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "update_entities"):
            await self._adapter().update_entities(object())

    async def test_update_relations_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "update_relations"):
            await self._adapter().update_relations(object())

    async def test_update_knowledge_graph_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "update_knowledge_graph"):
            await self._adapter().update_knowledge_graph(object())

    async def test_get_entity_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "get_entity"):
            await self._adapter().get_entity("id-1", label="Doc")

    async def test_delete_entity_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "delete_entity"):
            await self._adapter().delete_entity("id-1", label="Doc")

    async def test_delete_relation_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "delete_relation"):
            await self._adapter().delete_relation(
                label="knows", source_id="a", target_id="b"
            )

    async def test_cypher_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "cypher"):
            await self._adapter().cypher("MATCH (n) RETURN n")

    async def test_entity_similarity_search_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "entity_similarity_search"):
            await self._adapter().entity_similarity_search("query", label="Doc")

    async def test_entity_fulltext_search_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "entity_fulltext_search"):
            await self._adapter().entity_fulltext_search("query", label="Doc")

    def test_repr_includes_class_name_and_uri(self):
        adapter = GraphDatabaseAdapter(uri="ladybug://x")
        self.assertIn("GraphDatabaseAdapter", repr(adapter))
        self.assertIn("ladybug://x", repr(adapter))

    def test_error_message_names_subclass(self):
        """Subclasses should appear by name in the NotImplementedError so
        the failure is actionable."""

        class CustomGraphAdapter(GraphDatabaseAdapter):
            pass

        with self.assertRaisesRegex(NotImplementedError, "CustomGraphAdapter"):
            CustomGraphAdapter().wipe_database()
