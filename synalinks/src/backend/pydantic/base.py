# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

"""
We provide different backend-dependent `DataModel`s to use.

These data models provide I/O for chatbots, agents, rags etc.

The user can build new data models by inheriting from these base models.

The check functions works for every type of data models (by checking the schema)
e.g. `SymbolicDataModel`, `JsonDataModel`, `DataModel` or `Variable`.
"""

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import Field
from pydantic import field_validator

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_schema_utils import contains_schema
from synalinks.src.backend.pydantic.core import DataModel


@synalinks_export(
    [
        "synalinks.backend.Score",
        "synalinks.Score",
    ]
)
class Score(float, Enum):
    VERY_BAD = 0.0
    POOR = 0.1
    BELOW_AVERAGE = 0.2
    LOW_AVERAGE = 0.3
    MEDIUM_LOW = 0.4
    MEDIUM = 0.5
    MEDIUM_HIGH = 0.6
    ABOVE_AVERAGE = 0.7
    HIGH_AVERAGE = 0.8
    GOOD = 0.9
    VERY_GOOD = 1.0


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
        "synalinks.backend.ChatRole",
        "synalinks.ChatRole",
    ]
)
class ChatRole(str, Enum):
    """The chat message roles"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@synalinks_export(
    [
        "synalinks.backend.ToolCalling",
        "synalinks.ToolCalling",
        "synalinks.ToolCall",
        "synalinks.backend.ToollCall",
    ]
)
class ToolCall(DataModel):
    id: str = Field(
        description="The id of the tool call",
    )
    name: str = Field(
        description="The name of the function called",
    )
    arguments: Dict[str, Any] = Field(
        description="The arguments of the tool call",
    )


@synalinks_export(
    [
        "synalinks.backend.ChatMessage",
        "synalinks.ChatMessage",
    ]
)
class ChatMessage(DataModel):
    """A chat message"""

    role: ChatRole = Field(
        description="The chat message role",
    )
    content: Union[str, Dict[str, Any]] = Field(
        description="The content of the message",
        default="",
    )
    tool_call_id: Optional[str] = Field(
        description="The id of the tool call if role is `tool`",
        default=None,
    )
    tool_calls: List[ToolCall] = Field(
        description="The tool calls of the agent",
        default=[],
    )


@synalinks_export(
    [
        "synalinks.backend.is_chat_message",
        "synalinks.is_chat_message",
    ]
)
def is_chat_message(x):
    """Checks if the given data model is a chat message

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), ChatMessage.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.ChatMessages",
        "synalinks.ChatMessages",
    ]
)
class ChatMessages(DataModel):
    """A list of chat messages"""

    messages: List[ChatMessage] = Field(
        description="The list of chat messages",
        default=[],
    )

    @field_validator("messages", mode="before")
    @classmethod
    def convert_dicts_to_chat_messages(cls, v):
        """Convert dict messages to ChatMessage objects."""
        if isinstance(v, list):
            return [
                ChatMessage(**msg) if isinstance(msg, dict) else msg
                for msg in v
            ]
        return v


@synalinks_export(
    [
        "synalinks.backend.is_chat_messages",
        "synalinks.is_chat_messages",
    ]
)
def is_chat_messages(x):
    """Checks if the given data model are chat messages

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), ChatMessages.get_schema()):
        return True
    return False


@synalinks_export(
    [
        "synalinks.backend.is_tool_call",
        "synalinks.is_tool_call",
    ]
)
def is_tool_call(x):
    """Checks if the given data model is a tool call

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    if contains_schema(x.get_schema(), ToolCall.get_schema()):
        return True
    return False


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
        description="The number of visits",
        default=0,
    )
    cumulative_reward: float = Field(
        description="The cumulative reward",
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


@synalinks_export(
    [
        "synalinks.backend.SimilaritySearch",
        "synalinks.SimilaritySearch",
    ]
)
class SimilaritySearch(DataModel):
    entity_label: str = Field(
        description=("The label of the entity to look for(use `*` to match them all)"),
    )
    similarity_search: str = Field(
        description=("The natural language similarity query to match specific entities"),
    )


@synalinks_export(
    [
        "synalinks.backend.is_similarity_search",
        "synalinks.is_similarity_search",
    ]
)
def is_similarity_search(x):
    """Checks if is a similarity search data model

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    schema = x.get_schema()
    properties = schema.get("properties", None)
    if properties:
        if properties.get("entity_label", None) and properties.get(
            "similarity_search", None
        ):
            return True
    return False


@synalinks_export(
    [
        "synalinks.backend.TripletSearch",
        "synalinks.TripletSearch",
    ]
)
class TripletSearch(DataModel):
    subject_label: str = Field(
        description=("The label of the subject entity (use `*` to match them all)"),
    )
    subject_similarity_search: str = Field(
        description=(
            "A short similarity query to match specific subjects"
            "(use `*` to match them all)",
        ),
    )
    relation_label: str = Field(
        description="The label of the relation to search for",
    )
    object_label: str = Field(
        description=("The label of the object entity (use `*` to match them all)"),
    )
    object_similarity_search: str = Field(
        description=(
            "A short similarity query to match specific objects"
            " (use `*` to match them all)"
        ),
    )


@synalinks_export(
    [
        "synalinks.backend.is_triplet_search",
        "synalinks.is_triplet_search",
    ]
)
def is_triplet_search(x):
    """Checks if is a triplet seach data model

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
            properties.get("subject_label", None)
            and properties.get("subject_similarity_search", None)
            and properties.get("relation_label", None)
            and properties.get("object_label", None)
            and properties.get("object_similarity_search", None)
        ):
            return True
    return False
