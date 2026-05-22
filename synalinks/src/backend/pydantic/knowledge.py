"""Knowledge-oriented `DataModel` primitives.

labelled entities, typed relations, knowledge graphs,
used by the knowledge base layer.

The `is_*` predicates work for any data-model flavour
(`SymbolicDataModel`, `JsonDataModel`, `DataModel`, `Variable`) by
inspecting the JSON schema.
"""

from typing import List

from pydantic import Field

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.pydantic.base import Embedding
from synalinks.src.backend.pydantic.core import DataModel


@synalinks_export(
    [
        "synalinks.backend.Entity",
        "synalinks.Entity",
    ]
)
class Entity(DataModel):
    """An entity data model"""

    label: str = Field(
        description="The entity label",
    )


@synalinks_export(
    [
        "synalinks.backend.is_entity",
        "synalinks.is_entity",
    ]
)
def is_entity(x):
    """Checks if the given data model is an entity

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("label", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.EmbeddedEntity",
        "synalinks.EmbeddedEntity",
    ]
)
class EmbeddedEntity(Embedding, Entity):
    """An entity with an embedding vector"""

    pass


@synalinks_export(
    [
        "synalinks.backend.is_embedded_entity",
        "synalinks.is_embedded_entity",
    ]
)
def is_embedded_entity(x):
    """Checks if the given data model is an embedded entity

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("label", None) and properties.get("embedding", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Relation",
        "synalinks.Relation",
    ]
)
class Relation(DataModel):
    """A relation model"""

    subj: Entity = Field(
        description="The subject entity",
    )
    label: str = Field(
        description="The relation label",
    )
    obj: Entity = Field(
        description="The object entity",
    )


@synalinks_export(
    [
        "synalinks.backend.is_relation",
        "synalinks.is_relation",
    ]
)
def is_relation(x):
    """Checks if is a relation model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if (
            properties.get("subj", None)
            and properties.get("label", None)
            and properties.get("obj", None)
        ):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Entities",
        "synalinks.Entities",
    ]
)
class Entities(DataModel):
    """A list of entities"""

    entities: List[Entity] = Field(
        description="The entities",
    )


@synalinks_export(
    [
        "synalinks.backend.is_entities",
        "synalinks.is_entities",
    ]
)
def is_entities(x):
    """Checks if is an entities model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("entities", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Relations",
        "synalinks.Relations",
    ]
)
class Relations(DataModel):
    """A list of relations"""

    relations: List[Relation] = Field(
        description="The relations",
    )


@synalinks_export(
    [
        "synalinks.backend.is_relations",
        "synalinks.is_relations",
    ]
)
def is_relations(x):
    """Checks if is an relations model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("relations", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.KnowledgeGraph",
        "synalinks.KnowledgeGraph",
    ]
)
class KnowledgeGraph(DataModel):
    """A knowledge graph (entities + relations)"""

    entities: List[Entity] = Field(
        description="The entities",
    )
    relations: List[Relation] = Field(
        description="The relations",
    )


@synalinks_export(
    [
        "synalinks.backend.is_knowledge_graph",
        "synalinks.is_knowledge_graph",
    ]
)
def is_knowledge_graph(x):
    """Checks if is a knowledge graph model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("entities", None) and properties.get("relations", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.KnowledgeGraphs",
        "synalinks.KnowledgeGraphs",
    ]
)
class KnowledgeGraphs(DataModel):
    """A list of knowledge graphs"""

    knowledge_graphs: List[KnowledgeGraph] = Field(
        description="The knowledge graphs",
    )


@synalinks_export(
    [
        "synalinks.backend.is_knowledge_graphs",
        "synalinks.is_knowledge_graphs",
    ]
)
def is_knowledge_graphs(x):
    """Checks if is a knowledge graphs model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("knowledge_graphs", None):
            return True
    return False
