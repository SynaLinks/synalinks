# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Dict

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.knowledge_bases import database_adapters
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.KnowledgeBase")
class KnowledgeBase(SynalinksSaveable):
    """A generic graph knowledge base.

    ### Using Neo4j graph database

    ```python
    import synalinks
    import os

    class Document(synalinks.Entity):
        title: str
        content: str

    class Chunk(synalinks.Entity):
        content: str

    class IsPartOf(synalinks.Relation):
        source: Chunk
        target: Document

    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large"
    )

    os.environ["NEO4J_DATABASE"] = "your-neo4j-db" # (Default to "neo4j")
    os.environ["NEO4J_USERNAME"] = "your-neo4j-username" # (Default to "neo4j")
    os.environ["NEO4J_PASSWORD"] = "your-neo4j-password" # (Default to "neo4j")

    knowledge_base = synalinks.KnowledgeBase(
        uri="neo4j://localhost:7687",
        entity_models=[Document, Chunk],
        relation_models=[IsPartOf],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=False,
    )
    ```

    Learn more about Neo4J in their documentation **[here](https://neo4j.com/docs/)**

    ### Using MemGraph graph database

    ```python
    os.environ["MEMGRAPH_DATABASE"] = "your-memgraph-db" # (Default to "memgraph")
    os.environ["MEMGRAPH_USERNAME"] = "your-memgraph-username" # (Default to "memgraph")
    os.environ["MEMGRAPH_PASSWORD"] = "your-memgraph-password" # (Default to "memgraph")

    knowledge_base = synalinks.KnowledgeBase(
        uri="memgraph://localhost:7687",
        entity_models=[Document, Chunk],
        relation_models=[IsPartOf],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=False,
    )
    ```

    Learn more about MemGraph in their documentation **[here](https://memgraph.com/docs)**

    **Note**: Obviously, use an `.env` file and `.gitignore` to avoid putting
    your username and password in the code or a config file that can lead to
    leackage when pushing it into repositories.

    Args:
        uri (str): The index name/url of the database.
        entity_models (list): The entity models being a list of `Entity`.
        relation_models (list): The relation models being a list of `Relation`.
        embedding_model (EmbeddingModel): The embedding model.
        metric (str): The metric to use for the vector index (`cosine` or `euclidean`).
        wipe_on_start (bool): Wether or not to wipe the graph database at start
            (Default to False).
        name (str): Optional. The name of the knowledge base used for serialization.
    """

    def __init__(
        self,
        uri=None,
        entity_models=None,
        relation_models=None,
        embedding_model=None,
        metric="cosine",
        wipe_on_start=False,
        name=None,
    ):
        self.adapter = database_adapters.get(uri)(
            uri=uri,
            entity_models=entity_models,
            relation_models=relation_models,
            embedding_model=embedding_model,
            metric=metric,
            wipe_on_start=wipe_on_start,
        )
        self.uri = uri
        self.entity_models = entity_models
        self.relation_models = relation_models
        self.embedding_model = embedding_model
        self.metric = metric
        self.wipe_on_start = wipe_on_start
        if not name:
            self.name = auto_name("knowledge_base")
        else:
            self.name = name

    async def update(
        self,
        data_model,
        threshold=0.8,
    ):
        """Update the knowledge base with new data.

        Adds or updates entities and relationships in the knowledge graph based on
        the provided data model. Perform alignment operations to
        merge similar entities.

        Args:
            data_model (JsonDataModel | DataModel): The data model containing entities
                and relations to be added or updated in the knowledge base.
                Should conform to the entity or relation models defined during
                initialization.
            threshold (float): Similarity threshold for entity alignment.
                Entities with similarity above this threshold will be merged.
                Should be between 0.0 and 1.0 (Defaults to 0.8).
        """
        return await self.adapter.update(data_model)

    async def query(self, query: str, params: Dict[str, Any] = None, **kwargs):
        """Execute a query against the knowledge base.

        Args:
            query (str): The Cypher query to execute. The format depends on the
                underlying database adapter (e.g., Cypher for Neo4j).

        Returns:
            (GenericResult): the query results
        """
        return await self.adapter.query(query, params=params, **kwargs)

    async def similarity_search(
        self,
        similarity_search,
        k=10,
        threshold=0.8,
    ):
        """Perform similarity search to find entities similar to the given text.

        Uses vector embeddings to find entities in the knowledge base that are
        semantically similar to the provided text query.

        Args:
            similarity_search (JsonDataModel): The `SimilaritySearch` data model.
            k (int): Maximum number of similar entities to return.
                Defaults to 10.
            threshold (float): Minimum similarity score for results.
                Entities with similarity below this threshold are excluded.
                Should be between 0.0 and 1.0 (Defaults to 0.8).
        """
        return await self.adapter.similarity_search(
            similarity_search,
            k=k,
            threshold=threshold,
        )

    async def triplet_search(
        self,
        triplet_search,
        k=10,
        threshold=0.8,
    ):
        """Search for triplets in the knowledge graph.

        Finds relationship triplets in the knowledge base that match or are similar
        to the provided triplet pattern.

        Args:
            triplet_search (JsonDataModel): The `TripletSearch` data model.
            k (int): Maximum number of matching triplets to return.
                (Defaults to 10).
            threshold (float, optional): Minimum similarity score for triplet matches.
                Triplets with similarity below this threshold are excluded.
                Should be between 0.0 and 1.0. (Defaults to 0.8).
        """
        return await self.adapter.triplet_search(
            triplet_search,
            k=k,
            threshold=threshold,
        )

    def get_config(self):
        config = {
            "uri": self.uri,
            "name": self.name,
            "metric": self.metric,
            "wipe_on_start": self.wipe_on_start,
        }
        entity_models_config = {
            "entity_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        entity_model.to_symbolic_data_model(
                            name="entity_model" + (f"_{i}_" if i > 0 else "_") + self.name
                        )
                    )
                    if not is_symbolic_data_model(entity_model)
                    else serialization_lib.serialize_synalinks_object(entity_model)
                )
                for i, entity_model in enumerate(self.entity_models)
            ]
        }
        relation_models_config = {
            "relation_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        relation_model.to_symbolic_data_model(
                            name="relation_model" + (f"_{i}_" if i > 0 else "_") + self.name
                        )
                    )
                    if not is_symbolic_data_model(relation_model)
                    else serialization_lib.serialize_synalinks_object(relation_model)
                )
                for i, relation_model in enumerate(self.relation_models)
            ]
        }
        embedding_model_config = {
            "embedding_model": serialization_lib.serialize_synalinks_object(
                self.embedding_model,
            )
        }
        return {
            **entity_models_config,
            **relation_models_config,
            **embedding_model_config,
            **config,
        }

    @classmethod
    def from_config(cls, config):
        entity_models_config = config.pop("entity_models")
        entity_models = [
            serialization_lib.deserialize_synalinks_object(entity_model)
            for entity_model in entity_models_config
        ]
        relation_models_config = config.pop("relation_models")
        relation_models = [
            serialization_lib.deserialize_synalinks_object(relation_model)
            for relation_model in relation_models_config
        ]
        embedding_model = serialization_lib.deserialize_synalinks_object(
            config.pop("embedding_model"),
        )
        return cls(
            entity_models=entity_models,
            relation_models=relation_models,
            embedding_model=embedding_model,
            **config,
        )
