# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Common message data models.

Chat-message primitives shared across modules, agents, and the OpenAI
Chat Completions wire converter. Lives here (rather than in `base.py`)
because every chat-shaped path — generators, agents, datasets, tool
calling — depends on `ChatMessage` / `ChatMessages` / `ToolCall`.
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
        "synalinks.backend.ChatRole",
        "synalinks.ChatRole",
    ]
)
class ChatRole(str, Enum):
    """The chat message roles"""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


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
        "synalinks.backend.ChatMessage",
        "synalinks.ChatMessage",
    ]
)
class ChatMessage(DataModel):
    """A chat message"""

    role: ChatRole = Field(
        description="The chat message role",
    )
    thinking: Optional[str] = Field(
        description="The thinking/reasoning content of the message",
        default=None,
    )
    thinking_blocks: Optional[List[Dict[str, Any]]] = Field(
        description=(
            "Opaque provider-native thinking blocks (e.g. Anthropic's signed "
            "`thinking_blocks`). Carried through verbatim on assistant-message "
            "re-injection so multi-turn tool-use round-trips preserve signatures. "
            "None for providers that emit reasoning only as text."
        ),
        default=None,
    )
    content: Union[str, Union[List[Dict[str, Any]], Dict[str, Any]]] = Field(
        description="The content of the message",
        default="",
    )
    tool_call_id: Optional[str] = Field(
        description="The id of the tool call if role is `tool`",
        default=None,
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        description="The tool calls of the agent",
        default=None,
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
        "synalinks.backend.is_strictly_chat_message",
        "synalinks.is_strictly_chat_message",
    ]
)
def is_strictly_chat_message(x):
    """Checks if the given data model is strictly a chat message

    Unlike `is_chat_message`, this performs an equality check on the schema
    rather than a containment check.

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    return x.get_schema() == ChatMessage.get_schema()


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
            return [ChatMessage(**msg) if isinstance(msg, dict) else msg for msg in v]
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
        "synalinks.backend.is_strictly_chat_messages",
        "synalinks.is_strictly_chat_messages",
    ]
)
def is_strictly_chat_messages(x):
    """Checks if the given data model is strictly chat messages

    Unlike `is_chat_messages`, this performs an equality check on the schema
    rather than a containment check.

    Args:
        x (DataModel | JsonDataModel | SymbolicDataModel | Variable):
            The data model to check.

    Returns:
        (bool): True if the condition is met
    """
    return x.get_schema() == ChatMessages.get_schema()
