# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Literal

from synalinks.src import testing
from synalinks.src.backend.pydantic.base import EmbeddedEntity
from synalinks.src.backend.pydantic.base import Embedding
from synalinks.src.backend.pydantic.base import Entities
from synalinks.src.backend.pydantic.base import Entity
from synalinks.src.backend.pydantic.base import KnowledgeGraph
from synalinks.src.backend.pydantic.base import Relation
from synalinks.src.backend.pydantic.base import Relations
from synalinks.src.backend.pydantic.base import is_embedded_entity
from synalinks.src.backend.pydantic.base import is_entities
from synalinks.src.backend.pydantic.base import is_entity
from synalinks.src.backend.pydantic.base import is_knowledge_graph
from synalinks.src.backend.pydantic.base import is_relation
from synalinks.src.backend.pydantic.base import is_relations
from synalinks.src.backend.pydantic.core import DataModel


class EntityTest(testing.TestCase):
    def test_entity_has_label_field(self):
        entity = Entity(label="Alice")
        self.assertEqual(entity.get_json(), {"label": "Alice"})

    def test_is_entity_true(self):
        self.assertTrue(is_entity(Entity))
        self.assertTrue(is_entity(Entity(label="Alice")))

    def test_is_entity_false_when_no_label(self):
        class NotAnEntity(DataModel):
            name: str

        self.assertFalse(is_entity(NotAnEntity))

    def test_entity_subclass_with_literal_label(self):
        class Author(Entity):
            label: Literal["Author"]
            full_name: str

        author = Author(label="Author", full_name="Ada Lovelace")
        self.assertEqual(
            author.get_json(),
            {"label": "Author", "full_name": "Ada Lovelace"},
        )
        self.assertTrue(is_entity(Author))


class EmbeddedEntityTest(testing.TestCase):
    def test_embedded_entity_has_label_and_embedding(self):
        ee = EmbeddedEntity(label="Doc", embedding=[0.1, 0.2, 0.3])
        self.assertEqual(
            ee.get_json(),
            {"label": "Doc", "embedding": [0.1, 0.2, 0.3]},
        )

    def test_is_embedded_entity_true(self):
        self.assertTrue(is_embedded_entity(EmbeddedEntity))
        self.assertTrue(
            is_embedded_entity(EmbeddedEntity(label="Doc", embedding=[1.0]))
        )

    def test_is_embedded_entity_false_for_plain_entity(self):
        self.assertFalse(is_embedded_entity(Entity))

    def test_is_embedded_entity_false_for_plain_embedding(self):
        self.assertFalse(is_embedded_entity(Embedding))


class RelationTest(testing.TestCase):
    def test_relation_has_subj_label_obj(self):
        rel = Relation(
            subj=Entity(label="Alice"),
            label="knows",
            obj=Entity(label="Bob"),
        )
        json = rel.get_json()
        self.assertEqual(json["label"], "knows")
        self.assertEqual(json["subj"], {"label": "Alice"})
        self.assertEqual(json["obj"], {"label": "Bob"})

    def test_is_relation_true(self):
        self.assertTrue(is_relation(Relation))

    def test_is_relation_false_when_missing_fields(self):
        class IncompleteRelation(DataModel):
            label: str
            subj: Entity

        self.assertFalse(is_relation(IncompleteRelation))


class EntitiesTest(testing.TestCase):
    def test_entities_holds_a_list(self):
        es = Entities(entities=[Entity(label="A"), Entity(label="B")])
        self.assertEqual(len(es.entities), 2)

    def test_is_entities_true(self):
        self.assertTrue(is_entities(Entities))

    def test_is_entities_false_for_relation(self):
        self.assertFalse(is_entities(Relation))


class RelationsTest(testing.TestCase):
    def test_relations_holds_a_list(self):
        rs = Relations(
            relations=[
                Relation(
                    subj=Entity(label="A"),
                    label="r",
                    obj=Entity(label="B"),
                ),
            ]
        )
        self.assertEqual(len(rs.relations), 1)

    def test_is_relations_true(self):
        self.assertTrue(is_relations(Relations))

    def test_is_relations_false_for_entities(self):
        self.assertFalse(is_relations(Entities))


class KnowledgeGraphTest(testing.TestCase):
    def test_knowledge_graph_has_entities_and_relations(self):
        kg = KnowledgeGraph(
            entities=[Entity(label="Alice"), Entity(label="Bob")],
            relations=[
                Relation(
                    subj=Entity(label="Alice"),
                    label="knows",
                    obj=Entity(label="Bob"),
                ),
            ],
        )
        self.assertEqual(len(kg.entities), 2)
        self.assertEqual(len(kg.relations), 1)

    def test_is_knowledge_graph_true(self):
        self.assertTrue(is_knowledge_graph(KnowledgeGraph))

    def test_is_knowledge_graph_false_without_relations(self):
        class OnlyEntities(DataModel):
            entities: List[Entity]

        self.assertFalse(is_knowledge_graph(OnlyEntities))

    def test_is_knowledge_graph_false_without_entities(self):
        class OnlyRelations(DataModel):
            relations: List[Relation]

        self.assertFalse(is_knowledge_graph(OnlyRelations))
