# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
We provide different backend-dependent `DataModel`s to use.

These data models provide I/O for chatbots, agents, rags etc.

The user can build new data models by inheriting from these base models.

The check functions works for every type of data models (by checking the schema)
e.g. `SymbolicDataModel`, `JsonDataModel`, `DataModel` or `Variable`.
"""

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import Field

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_schema_utils import contains_schema
from synalinks.src.backend.pydantic.core import DataModel


@synalinks_export(
    [
        "synalinks.backend.GenericOutputs",
        "synalinks.GenericOutputs",
    ]
)
class GenericOutputs(DataModel):
    """A generic outputs"""

    outputs: Dict[str, Any] = Field(
        description="The outputs",
    )


@synalinks_export(
    [
        "synalinks.backend.GenericInputs",
        "synalinks.GenericInputs",
    ]
)
class GenericInputs(DataModel):
    """A generic inputs"""

    inputs: Dict[str, Any] = Field(
        description="The inputs",
    )


@synalinks_export(
    [
        "synalinks.backend.Stamp",
        "synalinks.Stamp",
    ]
)
class Stamp(DataModel):
    created_at: Optional[datetime] = Field(
        description="The creation time",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.is_stamped",
        "synalinks.is_stamped",
    ]
)
def is_stamped(x):
    """Checks if the given data model is stamped

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("created_at", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.GenericIO",
        "synalinks.GenericIO",
    ]
)
class GenericIO(DataModel):
    """A pair of generic inputs/outputs"""

    inputs: Dict[str, Any] = Field(
        description="The inputs",
    )
    outputs: Dict[str, Any] = Field(
        description="The outputs",
    )


@synalinks_export(
    [
        "synalinks.backend.GenericResult",
        "synalinks.GenericResult",
    ]
)
class GenericResult(DataModel):
    """A generic result"""

    result: List[Any] = Field(
        description="The result",
    )


@synalinks_export(
    [
        "synalinks.backend.EmbeddingRequest",
        "synalinks.EmbeddingRequest",
    ]
)
class EmbeddingRequest(DataModel):
    """Input for an embedding model: a single text or a batch."""

    texts: Union[str, List[str]] = Field(
        description="A text or list of texts to embed",
    )


@synalinks_export(
    [
        "synalinks.backend.Embedding",
    ]
)
class Embedding(DataModel):
    """An embedding vector"""

    embedding: List[float] = Field(
        description="The embedding vector",
        default=[],
    )


@synalinks_export(
    [
        "synalinks.backend.is_embedding",
        "synalinks.is_embedding",
    ]
)
def is_embedding(x):
    """Checks if the given data model is an embedding

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Embedding.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Embeddings",
        "synalinks.Embeddings",
    ]
)
class Embeddings(DataModel):
    """A list of embeddings"""

    embeddings: List[List[float]] = Field(
        description="The list of embedding vectors",
        default=[],
    )


@synalinks_export(
    [
        "synalinks.backend.is_embeddings",
        "synalinks.is_embeddings",
    ]
)
def is_embeddings(x):
    """Checks if the given data model are embeddings

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Embeddings.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.is_embedded",
        "synalinks.is_embedded",
    ]
)
def is_embedded(x):
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
        if properties.get("embedding", None):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Prediction",
        "synalinks.Prediction",
    ]
)
class Prediction(GenericIO):
    reward: Optional[float] = Field(
        description="The prediction's reward",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.is_prediction",
        "synalinks.is_prediction",
    ]
)
def is_prediction(x):
    """Checks if the given data model is a prediction

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Prediction.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Trainable",
        "synalinks.Trainable",
    ]
)
class Trainable(DataModel):
    examples: List[Prediction] = Field(
        description="The examples for few-shot learning",
        default=[],
    )
    current_predictions: List[Prediction] = Field(
        description="The current predictions store",
        default=[],
    )
    predictions: List[Prediction] = Field(
        description="The predictions store",
        default=[],
    )
    seed_candidates: List[Any] = Field(
        description="The seed candidates",
        default=[],
    )
    candidates: List[Any] = Field(
        description="The candidates",
        default=[],
    )
    best_candidates: List[Any] = Field(
        description="The best candidates",
        default=[],
    )
    history: List[Any] = Field(
        description="The candidates history",
        default=[],
    )
    nb_visit: int = Field(
        description=(
            "Number of scored predictions for this variable in the most recent "
            "batch (reset each batch, not accumulated). With `cumulative_reward` "
            "it forms the per-batch mean reward used to pick the struggling "
            "module to optimize. 0 means unvisited."
        ),
        default=0,
    )
    cumulative_reward: float = Field(
        description=(
            "Sum of rewards for this variable's predictions in the most recent "
            "batch (reset each batch). `cumulative_reward / nb_visit` is the "
            "per-batch mean reward."
        ),
        default=0.0,
    )


@synalinks_export(
    [
        "synalinks.backend.is_trainable",
        "synalinks.is_trainable",
    ]
)
def is_trainable(x):
    """Checks if the given data model is Trainable

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Trainable.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.Instructions",
        "synalinks.Instructions",
    ]
)
class Instructions(Trainable):
    """The instructions for the language model"""

    instructions: Optional[str] = Field(
        description="The instructions for the language model",
    )


@synalinks_export(
    [
        "synalinks.backend.is_instructions",
        "synalinks.is_instructions",
    ]
)
def is_instructions(x):
    """Checks if the given data model is an instructions data model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), Instructions.get_schema()):
        return True
    return False
