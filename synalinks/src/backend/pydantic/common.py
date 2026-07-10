# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Common message data models.

Chat-message primitives shared across modules, agents, and the OpenAI
Chat Completions wire converter. Lives here (rather than in `base.py`)
because every chat-shaped path — generators, agents, datasets, tool
calling — depends on `ChatMessage` / `ChatMessages` / `ToolCall`.
"""

import warnings
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

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
        "synalinks.backend.ToolCallFunction",
        "synalinks.ToolCallFunction",
    ]
)
class ToolCallFunction(DataModel):
    """The `function` payload of a tool call (name + arguments).

    `arguments` is a parsed dict by default, but the raw JSON-encoded
    string used on the Chat Completions wire is also accepted, for users
    who prefer to keep the wire encoding. The converters in
    `backend.pydantic.chat_completions` are type-driven: a dict is
    JSON-encoded at the wire edge while a string passes through verbatim,
    and a wire string is parsed back into a dict.
    """

    name: str = Field(
        description="The name of the function called",
    )
    arguments: Union[Dict[str, Any], str] = Field(
        description=(
            "The arguments of the tool call, as a parsed dict or as the "
            "raw JSON-encoded string (chat-completion wire form)"
        ),
    )


@synalinks_export(
    [
        "synalinks.backend.ToolCall",
        "synalinks.ToolCall",
        "synalinks.backend.ToolCalling",
        "synalinks.ToolCalling",
    ]
)
class ToolCall(DataModel):
    """A tool call, shaped like an OpenAI Chat Completions tool call.

    Mirrors the wire envelope (`{id, type, function: {name, arguments}}`)
    except that `arguments` is a parsed dict by default rather than a
    JSON-encoded string, so modules and agents can read it directly. The
    string form is also accepted; the wire converters in
    `backend.pydantic.chat_completions` handle either by type.
    """

    id: str = Field(
        description="The id of the tool call",
    )
    type: Literal["function"] = Field(
        description="The tool call type (always `function` today)",
        default="function",
    )
    function: ToolCallFunction = Field(
        description="The function invocation (name + arguments)",
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
    """A chat message.

    Its keys are in exact parity with the litellm-extended Chat Completions
    message (`backend.pydantic.chat_completions.ChatCompletionMessage`) —
    same names on both sides, enforced by test. Only value types may differ
    where synalinks is deliberately richer (e.g. parsed-dict tool-call
    arguments, dict tool-result content). When adding a field, add its wire
    twin in `chat_completions.py` too.
    """

    role: ChatRole = Field(
        description="The chat message role",
    )
    reasoning_content: Optional[str] = Field(
        description=(
            "The reasoning/thinking content of the message. Keyed to match the "
            "litellm/DeepSeek `reasoning_content` chat-completion field (a "
            "provider extension, not part of the base OpenAI spec), so the "
            "message API stays in exact key parity with the litellm-extended "
            "chat-completion message."
        ),
        default=None,
    )
    thinking_blocks: Optional[List[Dict[str, Any]]] = Field(
        description=(
            "Opaque provider-native thinking blocks (e.g. Anthropic's signed "
            "`thinking_blocks`; a litellm extension, not part of the base "
            "OpenAI spec). Carried through verbatim on assistant-message "
            "re-injection so multi-turn tool-use round-trips preserve "
            "signatures. None for providers that emit reasoning only as text."
        ),
        default=None,
    )
    content: Optional[Union[str, List[Dict[str, Any]], Dict[str, Any]]] = Field(
        description="The content of the message",
        default=None,
    )
    tool_call_id: Optional[str] = Field(
        description="The id of the tool call if role is `tool`",
        default=None,
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        description="The tool calls of the agent",
        default=None,
    )
    name: Optional[str] = Field(
        description=(
            "Optional author name for the message (chat-completion `name`). "
            "On the legacy `function` role it carries the function name."
        ),
        default=None,
    )
    refusal: Optional[str] = Field(
        description=(
            "The refusal message returned when the model declines to fulfill "
            "a structured-output request (chat-completion `refusal`)."
        ),
        default=None,
    )
    audio: Optional[Dict[str, Any]] = Field(
        description=(
            "Audio response payload when the audio output modality is used "
            "(chat-completion `audio`: id, data, transcript, expires_at). "
            "Distinct from audio *input*, which travels as an `input_audio` "
            "content part."
        ),
        default=None,
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_thinking(cls, data):
        """Accept the deprecated `thinking` key as an alias for `reasoning_content`.

        `thinking` was renamed to `reasoning_content` so `ChatMessage`'s keys
        stay in exact parity with the chat-completion message. Legacy construction
        (``ChatMessage(thinking=...)`` / ``ChatMessage(**{"thinking": ...})``)
        keeps working, mapping onto `reasoning_content`.
        """
        if isinstance(data, dict) and "thinking" in data:
            thinking = data.pop("thinking")
            data.setdefault("reasoning_content", thinking)
            warnings.warn(
                "`ChatMessage.thinking` is deprecated; use `reasoning_content` "
                "(the OpenAI/litellm chat-completion key). The `thinking` alias "
                "will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_content(cls, data):
        """Normalize a multimodal `content` list to its chat-completion shape.

        A `content` list may mix plain strings (text) and special media types
        (`Image`/`Audio`); each is mapped to its content part so the stored
        `content` is a strict list of chat-completion parts. Plain string or
        already-shaped dict content is left untouched.
        """
        if isinstance(data, dict) and isinstance(data.get("content"), list):
            from synalinks.src.backend.pydantic.media import normalize_content

            data["content"] = normalize_content(data["content"])
        return data

    @property
    def thinking(self) -> Optional[str]:
        """Deprecated read alias for `reasoning_content`."""
        warnings.warn(
            "`ChatMessage.thinking` is deprecated; read `reasoning_content` "
            "instead. The `thinking` alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.reasoning_content


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
