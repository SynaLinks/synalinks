# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import copy
import re
from typing import Any
from typing import Dict

from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.utils.async_utils import run_maybe_nested
from synalinks.src.utils.naming import to_snake_case


class DatabaseAdapter:
    def __init__(
        self,
        uri=None,
        embedding_model=None,
        entity_models=None,
        relation_models=None,
        metric="cosine",
        wipe_on_start=False,
        **kwargs,
    ):
        self.uri = uri
        if not embedding_model:
            raise ValueError("KnowledgeBase requires `embedding_model` argument")
        if not isinstance(embedding_model, EmbeddingModel):
            raise ValueError(
                "KnowledgeBase `embedding_model` argument to be an `EmbeddingModel`"
            )

        self.embedding_model = embedding_model
        self.embedding_dim = len(
            run_maybe_nested(embedding_model(texts=["test"]))["embeddings"][0]
        )

        if metric not in ("cosine", "euclidean"):
            raise ValueError(
                "KnowledgeBase `metric` argument should between `cosine` or `euclidean`"
            )
        self.metric = metric

        if not entity_models and not relation_models:
            raise ValueError(
                "KnowledgeBase requires `entity_models` and `relation_models` argument"
            )
        if not entity_models:
            raise ValueError("KnowledgeBase requires `entity_models` argument")

        if not relation_models:
            relation_models = []

        self.entity_models = entity_models
        self.relation_models = relation_models

        if wipe_on_start:
            self.wipe_database()

        self.create_vector_index()

    def sanitize(self, string):
        """Prevent Cypher injections.

        Args:
            string (str): The string to sanitize

        Returns:
            (str): The sanitized string
        """
        string = re.sub(r"//.*?(\n|$)", "", string)
        string = re.sub(r"/\*.*?\*/", "", string, flags=re.DOTALL)
        return string

    def sanitize_label(self, label):
        """Prevent Cypher injections.

        Args:
            label (str): The label to sanitize

        Returns:
            (str): The sanitized string
        """
        label = self.sanitize(label)
        label = re.sub(r"[^a-zA-Z0-9]", "", label)
        return label

    def sanitize_property_name(self, property_name):
        """Prevent Cypher injections.

        Args:
            property_name (str): The property name to sanitize

        Returns:
            (str): The sanitized property name
        """
        property_name = self.sanitize(property_name)
        property_name = re.sub(r"[^a-zA-Z0-9_]", "", property_name)
        return to_snake_case(property_name)

    def sanitize_properties(self, properties):
        """Prevent Cypher injections.

        Args:
            properties (dict): The properties to sanitize

        Returns:
            (dict): The sanitized properties
        """
        properties = copy.deepcopy(properties)
        return {self.sanitize_property_name(k): v for k, v in properties.items()}

    def wipe_database(self):
        raise NotImplementedError(
            f"{self.__class__} should implement the `wipe_database()` method"
        )

    def create_vector_index(self):
        raise NotImplementedError(
            f"{self.__class__} should implement the `create_vector_index()` method"
        )

    async def update(self, data_model, threshold=0.8):
        raise NotImplementedError(
            f"{self.__class__} should implement the `update()` method"
        )

    async def query(self, query: str, params: Dict[str, Any] = None, **kwargs):
        raise NotImplementedError(
            f"{self.__class__} should implement the `query()` method"
        )

    async def similarity_search(self, text, data_model=None, k=10, threshold=0.8):
        raise NotImplementedError(
            f"{self.__class__} should implement the `similarity_search()` method"
        )
        
    async def fulltext_search(self, keywords, data_model=None, k=10, threshold=0.8):
        raise NotImplementedError(
            f"{self.__class__} should implement the `fulltext_search()` method"
        )
        
    async def chain_search(self, chain_search, data_model=None, k=10, threshold=0.8):
        raise NotImplementedError(
            f"{self.__class__} should implement the `chain_search()` method"
        )

    async def triplet_search(self, triplet_search, k=10, threshold=0.8):
        raise NotImplementedError(
            f"{self.__class__} should implement the `triplet_search()` method"
        )

    async def __repr__(self):
        return f"<DatabaseAdapter index={self.uri}>"
