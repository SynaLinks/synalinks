# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import Embeddings
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.modules.embedding_models import get as _get_em
from synalinks.src.ops.operation import Operation
from synalinks.src.saving import serialization_lib


class Embedding(Operation):
    """Extract the embedding vectors from a data model using an `EmbeddingModel`.

    Args:
        embedding_model (EmbeddingModel): The embedding model to use.
        name (str): Optional. The name of the operation.
        description (str): Optional. Description of the operation.
        **kwargs (keyword warguments): Additional keyword arguments send to the
            embedding model.
    """

    def __init__(
        self,
        embedding_model=None,
        name=None,
        description=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.embedding_model = _get_em(embedding_model)
        self.em_kwargs = kwargs

    async def call(self, x):
        texts = tree.flatten(tree.map_structure(lambda field: str(field), x.get_json()))
        embeddings = await self.embedding_model(texts, **self.em_kwargs)
        return JsonDataModel(data_model=Embeddings(**embeddings), name=self.name)

    async def compute_output_spec(self, x):
        return SymbolicDataModel(schema=Embeddings.get_schema(), name=self.name)

    def get_config(self):
        config = {
            "name": self.name,
            "description": self.description,
        }
        embedding_model_config = serialization_lib.serialize_synalinks_object(
            self.embedding_model
        )
        config.update({"em_kwargs": self.em_kwargs})
        return {"embedding_model": embedding_model_config, **config}

    @classmethod
    def from_config(cls, config):
        embedding_model = serialization_lib.deserialize_synalinks_object(
            config.pop("embedding_model")
        )
        em_kwargs = config.pop("em_kwargs")
        return cls(embedding_model=embedding_model, **config, **em_kwargs)


@synalinks_export(["synalinks.ops.embedding", "synalinks.ops.embedding_models.embedding"])
async def embedding(x, embedding_model=None, name=None, description=None, **kwargs):
    """Extract the embedding vectors from a data model using an `EmbeddingModel`.

    Embedding consist in converting the given data_model into a vector representation.
    This function always output a data model that uses `Embeddings` schema.

    If the input data model have multiple fields, each one is embedded.

    Args:
        x (JsonDataModel | SymbolicDataModel): the input data_model
        embedding_model (EmbeddingModel): The embedding model to use
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.
        **kwargs (keyword arguments): Additional keywords forwarded to the
            EmbeddingModel call.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data_model
    """
    if any_symbolic_data_models(x):
        return await Embedding(
            embedding_model=embedding_model,
            name=name,
            description=description,
        ).symbolic_call(x)
    return await Embedding(
        embedding_model=embedding_model,
        name=name,
        description=description,
        **kwargs,
    )(x)
