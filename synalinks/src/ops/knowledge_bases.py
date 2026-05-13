# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import GenericResult
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.ops.operation import Operation
from synalinks.src.saving import serialization_lib


class UpdateKnowledge(Operation):
    def __init__(
        self,
        knowledge_base=None,
        threshold=0.8,
        name=None,
        description=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.knowledge_base = knowledge_base
        self.threshold = 0.8

    async def call(self, x):
        await self.knowledge_base.update(
            x,
            threshold=self.threshold,
        )
        return JsonDataModel(
            json=x.get_json(),
            schema=x.get_schema(),
            name=self.name,
        )

    async def compute_output_spec(self, x):
        return SymbolicDataModel(
            schema=x.get_schema(),
            name=self.name,
        )

    def get_config(self):
        config = {
            "name": self.name,
            "description": self.description,
        }
        knowledge_base_config = serialization_lib.serialize_synalinks_object(
            self.knowledge_base
        )
        return {"knowledge_base": knowledge_base_config, **config}

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        return cls(knowledge_base=knowledge_base, **config)


@synalinks_export(
    [
        "synalinks.ops.update_knowledge",
        "synalinks.ops.knowledge_bases.update_knowledge",
    ]
)
async def update_knowledge(
    x,
    knowledge_base=None,
    threshold=0.8,
    name=None,
    description=None,
):
    """Update the knowledge base with new data.

    This function updates the specified knowledge base with the
    data model provided in the input.

    Args:
        x (JsonDataModel | SymbolicDataModel): The data model to update the
            knowledge base with.
        knowledge_base (KnowledgeBase): The knowledge base to update. Required.
        align (bool): Wether of not to perform automatic alignment (Default to True).
        threshold (float): The alignment threshold (Default to 0.7).
        name (str): Optional name for the operation.
        description (str): Optional description for the operation.

    Returns:
        (JsonDataModel| SymbolicDataModel): The resulting json data model.

    Raises:
        ValueError: If `knowledge_base` is not provided.
    """
    if knowledge_base is None:
        raise ValueError("You should provide the `knowledge_base` argument")
    if any_symbolic_data_models(x):
        return await UpdateKnowledge(
            knowledge_base=knowledge_base,
            name=name,
            threshold=threshold,
            description=description,
        ).symbolic_call(x)
    return await UpdateKnowledge(
        knowledge_base=knowledge_base,
        name=name,
        description=description,
    )(x)


class SimilaritySearch(Operation):
    def __init__(
        self,
        knowledge_base=None,
        k=10,
        threshold=0.7,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.knowledge_base = knowledge_base
        self.k = k
        self.threshold = threshold

    async def call(self, x):
        result = await self.knowledge_base.similarity_search(
            x,
            k=self.k,
            threshold=self.threshold,
        )
        return JsonDataModel(
            json={"result": result},
            schema=GenericResult.get_schema(),
            name=self.name,
        )

    async def compute_output_spec(self, x):
        return SymbolicDataModel(
            schema=GenericResult.get_schema(),
            name=self.name,
        )

    def get_config(self):
        config = {
            "k": self.k,
            "threshold": self.threshold,
            "name": self.name,
            "description": self.description,
        }
        knowledge_base_config = serialization_lib.serialize_synalinks_object(
            self.knowledge_base
        )
        return {"knowledge_base": knowledge_base_config, **config}

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        return cls(knowledge_base=knowledge_base, **config)


@synalinks_export(
    [
        "synalinks.ops.similarity_search",
        "synalinks.ops.knowledge_bases.similarity_search",
    ]
)
async def similarity_search(
    x,
    knowledge_base=None,
    k=10,
    threshold=0.7,
    name=None,
    description=None,
):
    """Search for similar entities in the given knowledge base.

    This function performs a similarity search in the specified knowledge base
    using the input data as the query.

    Args:
        x (JsonDataModel | SymbolicDataModel): The query for the similarity search.
        knowledge_base (KnowledgeBase): The knowledge base to search in.
        k (int): Maximum number of results to return (Defaults to 10).
        threshold (float): Similarity threshold for filtering results. Only results with
            similarity scores above this threshold will be returned (Defaults to 0.7).
        name (str): Optional name for the operation.
        description (str): Optional description for the operation.

    Returns:
        (JsonDataModel): A data model containing the search results.

    Raises:
        ValueError: If knowledge_base is not provided.
    """
    if knowledge_base is None:
        raise ValueError("You should provide the `knowledge_base` argument")
    if any_symbolic_data_models(x):
        return await SimilaritySearch(
            knowledge_base=knowledge_base,
            k=k,
            threshold=threshold,
            name=name,
            description=description,
        ).symbolic_call(x)
    return await SimilaritySearch(
        knowledge_base=knowledge_base,
        k=k,
        threshold=threshold,
        name=name,
        description=description,
    )(x)
