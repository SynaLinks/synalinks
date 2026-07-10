# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Data models for the OpenAI Chat Completions wire format.

These mirror the request/response shape of `POST /v1/chat/completions`
as faithfully as possible (e.g. `tool_calls[].function.arguments` stays
a JSON-encoded string, not a dict). For the synalinks-internal chat
vocabulary used inside modules, see `backend.pydantic.common`
(`ChatMessage`, `ChatMessages`, `ToolCall`).
"""

import json
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import ConfigDict
from pydantic import Field

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.pydantic.common import ChatMessage
from synalinks.src.backend.pydantic.common import ChatMessages
from synalinks.src.backend.pydantic.common import ChatRole
from synalinks.src.backend.pydantic.common import ToolCall
from synalinks.src.backend.pydantic.common import ToolCallFunction
from synalinks.src.backend.pydantic.core import DataModel


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionFunctionCall",
        "synalinks.ChatCompletionFunctionCall",
    ]
)
class ChatCompletionFunctionCall(DataModel):
    """The `function` payload inside a Chat Completions tool call."""

    name: str = Field(
        description="The name of the function to call",
    )
    arguments: Union[str, Dict[str, Any]] = Field(
        description=(
            "JSON-encoded arguments for the function call (a parsed dict "
            "is also accepted and kept as-is by the converters)"
        ),
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionToolCall",
        "synalinks.ChatCompletionToolCall",
    ]
)
class ChatCompletionToolCall(DataModel):
    """A tool call emitted by the assistant in Chat Completions."""

    id: str = Field(
        description="The id of the tool call",
    )
    type: Literal["function"] = Field(
        description="The tool call type (always `function` today)",
        default="function",
    )
    function: ChatCompletionFunctionCall = Field(
        description="The function invocation",
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionMessage",
        "synalinks.ChatCompletionMessage",
    ]
)
class ChatCompletionMessage(DataModel):
    """A single message in a Chat Completions request or response."""

    role: Literal[
        "developer",
        "system",
        "user",
        "assistant",
        "tool",
        "function",
    ] = Field(
        description="The message role",
    )
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        description=(
            "The message content. String for plain text, or a list of "
            "content parts for multimodal inputs. Null when the assistant "
            "returns only tool calls."
        ),
        default=None,
    )
    name: Optional[str] = Field(
        description="Optional author name for the message",
        default=None,
    )
    tool_calls: Optional[List[ChatCompletionToolCall]] = Field(
        description="Tool calls emitted by the assistant",
        default=None,
    )
    tool_call_id: Optional[str] = Field(
        description="The tool call id this message responds to (role=tool)",
        default=None,
    )
    refusal: Optional[str] = Field(
        description="The assistant refusal message, if any",
        default=None,
    )
    audio: Optional[Dict[str, Any]] = Field(
        description="Audio response payload when the audio modality is used",
        default=None,
    )
    reasoning_content: Optional[str] = Field(
        description=(
            "Provider reasoning text (litellm/DeepSeek-style extension, not "
            "part of the base OpenAI spec). Maps to "
            "`ChatMessage.reasoning_content`."
        ),
        default=None,
    )
    thinking_blocks: Optional[List[Dict[str, Any]]] = Field(
        description=(
            "Opaque provider-native thinking blocks (e.g. Anthropic's signed "
            "blocks; litellm extension). Carried through verbatim so multi-turn "
            "tool-use round-trips preserve signatures."
        ),
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionToolFunction",
        "synalinks.ChatCompletionToolFunction",
    ]
)
class ChatCompletionToolFunction(DataModel):
    """The `function` declaration inside a Chat Completions tool."""

    name: str = Field(
        description="The function name",
    )
    description: Optional[str] = Field(
        description="What the function does",
        default=None,
    )
    parameters: Optional[Dict[str, Any]] = Field(
        description="JSON Schema describing the function parameters",
        default=None,
    )
    strict: Optional[bool] = Field(
        description="Whether to enforce strict schema adherence",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionTool",
        "synalinks.ChatCompletionTool",
    ]
)
class ChatCompletionTool(DataModel):
    """A tool declaration sent in a Chat Completions request."""

    type: Literal["function"] = Field(
        description="The tool type (always `function` today)",
        default="function",
    )
    function: ChatCompletionToolFunction = Field(
        description="The function declaration",
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionJsonSchema",
        "synalinks.ChatCompletionJsonSchema",
    ]
)
class ChatCompletionJsonSchema(DataModel):
    """The `json_schema` payload of a `response_format` object."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    name: str = Field(
        description="The schema identifier (≤64 chars, [a-zA-Z0-9_-])",
    )
    description: Optional[str] = Field(
        description="What the schema represents — guides model selection",
        default=None,
    )
    schema_: Optional[Dict[str, Any]] = Field(
        alias="schema",
        description="The JSON Schema describing the response",
        default=None,
    )
    strict: Optional[bool] = Field(
        description="Whether to enforce strict schema adherence",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionResponseFormat",
        "synalinks.ChatCompletionResponseFormat",
    ]
)
class ChatCompletionResponseFormat(DataModel):
    """The `response_format` field of a Chat Completions request."""

    type: Literal["text", "json_object", "json_schema"] = Field(
        description="The response format type",
    )
    json_schema: Optional[ChatCompletionJsonSchema] = Field(
        description="The JSON schema spec when `type='json_schema'`",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionRequest",
        "synalinks.ChatCompletionRequest",
    ]
)
class ChatCompletionRequest(DataModel):
    """A Chat Completions request body."""

    model: str = Field(
        description="The model id to use",
    )
    messages: List[ChatCompletionMessage] = Field(
        description="The conversation messages",
    )
    tools: Optional[List[ChatCompletionTool]] = Field(
        description="Tools the model may call",
        default=None,
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        description="Controls which tool the model calls",
        default=None,
    )
    response_format: Optional[ChatCompletionResponseFormat] = Field(
        description="Constraint on the response format",
        default=None,
    )
    temperature: Optional[float] = Field(
        description="Sampling temperature",
        default=None,
    )
    top_p: Optional[float] = Field(
        description="Nucleus sampling probability mass",
        default=None,
    )
    n: Optional[int] = Field(
        description="Number of completions to generate",
        default=None,
    )
    max_tokens: Optional[int] = Field(
        description="Maximum tokens in the completion (deprecated)",
        default=None,
    )
    max_completion_tokens: Optional[int] = Field(
        description="Maximum tokens in the completion",
        default=None,
    )
    stop: Optional[Union[str, List[str]]] = Field(
        description="Stop sequences",
        default=None,
    )
    presence_penalty: Optional[float] = Field(
        description="Presence penalty",
        default=None,
    )
    frequency_penalty: Optional[float] = Field(
        description="Frequency penalty",
        default=None,
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        description="Token logit biases",
        default=None,
    )
    logprobs: Optional[bool] = Field(
        description="Whether to return log probabilities",
        default=None,
    )
    top_logprobs: Optional[int] = Field(
        description="Number of top log-probabilities to return",
        default=None,
    )
    seed: Optional[int] = Field(
        description="Sampling seed for best-effort determinism",
        default=None,
    )
    stream: Optional[bool] = Field(
        description="Whether to stream the response",
        default=None,
    )
    user: Optional[str] = Field(
        description="End-user identifier for abuse monitoring",
        default=None,
    )
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = Field(
        description="Reasoning effort hint for reasoning-capable models",
        default=None,
    )
    parallel_tool_calls: Optional[bool] = Field(
        description="Whether to allow the model to call multiple tools in parallel",
        default=None,
    )
    stream_options: Optional[Dict[str, Any]] = Field(
        description="Streaming configuration (e.g. {'include_usage': true})",
        default=None,
    )
    metadata: Optional[Dict[str, str]] = Field(
        description="Arbitrary key/value metadata attached to the request",
        default=None,
    )
    store: Optional[bool] = Field(
        description="Whether to store the completion for later retrieval",
        default=None,
    )
    service_tier: Optional[str] = Field(
        description="Service tier (e.g. 'auto', 'default', 'flex')",
        default=None,
    )
    modalities: Optional[List[str]] = Field(
        description="Output modalities requested (e.g. ['text', 'audio'])",
        default=None,
    )
    audio: Optional[Dict[str, Any]] = Field(
        description="Audio output configuration when 'audio' is in modalities",
        default=None,
    )
    prediction: Optional[Dict[str, Any]] = Field(
        description="Predicted output for latency optimization",
        default=None,
    )
    web_search_options: Optional[Dict[str, Any]] = Field(
        description="Configuration for the web_search built-in tool",
        default=None,
    )
    prompt_cache_key: Optional[str] = Field(
        description="Cache key to route similar prompts to the same cache",
        default=None,
    )
    safety_identifier: Optional[str] = Field(
        description="Stable end-user identifier for safety/abuse monitoring",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionPromptTokensDetails",
        "synalinks.ChatCompletionPromptTokensDetails",
    ]
)
class ChatCompletionPromptTokensDetails(DataModel):
    """Breakdown of tokens that made up the prompt."""

    cached_tokens: Optional[int] = Field(
        description="Tokens served from the prompt cache",
        default=None,
    )
    audio_tokens: Optional[int] = Field(
        description="Audio input tokens in the prompt",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionCompletionTokensDetails",
        "synalinks.ChatCompletionCompletionTokensDetails",
    ]
)
class ChatCompletionCompletionTokensDetails(DataModel):
    """Breakdown of tokens generated in the completion."""

    reasoning_tokens: Optional[int] = Field(
        description="Hidden reasoning tokens consumed by the model",
        default=None,
    )
    audio_tokens: Optional[int] = Field(
        description="Audio output tokens generated",
        default=None,
    )
    accepted_prediction_tokens: Optional[int] = Field(
        description="Predicted output tokens that matched the actual output",
        default=None,
    )
    rejected_prediction_tokens: Optional[int] = Field(
        description="Predicted output tokens that did not match",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionUsage",
        "synalinks.ChatCompletionUsage",
    ]
)
class ChatCompletionUsage(DataModel):
    """Token usage reported in a Chat Completions response."""

    prompt_tokens: int = Field(
        description="Tokens in the prompt",
    )
    completion_tokens: int = Field(
        description="Tokens in the completion",
    )
    total_tokens: int = Field(
        description="Total tokens consumed",
    )
    prompt_tokens_details: Optional[ChatCompletionPromptTokensDetails] = Field(
        description="Breakdown of prompt tokens",
        default=None,
    )
    completion_tokens_details: Optional[ChatCompletionCompletionTokensDetails] = Field(
        description="Breakdown of completion tokens",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionChoice",
        "synalinks.ChatCompletionChoice",
    ]
)
class ChatCompletionChoice(DataModel):
    """One completion choice in a Chat Completions response."""

    index: int = Field(
        description="The choice index",
    )
    message: ChatCompletionMessage = Field(
        description="The assistant message for this choice",
    )
    finish_reason: Optional[
        Literal[
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "function_call",
            "safety",
        ]
    ] = Field(
        description="Why the model stopped generating",
        default=None,
    )
    logprobs: Optional[Dict[str, Any]] = Field(
        description="Log-probability payload, when requested",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.ChatCompletionResponse",
        "synalinks.ChatCompletionResponse",
    ]
)
class ChatCompletionResponse(DataModel):
    """A Chat Completions response body."""

    id: str = Field(
        description="The completion id",
    )
    object: Literal["chat.completion"] = Field(
        description="The object type",
        default="chat.completion",
    )
    created: int = Field(
        description="Unix timestamp the completion was created",
    )
    model: str = Field(
        description="The model that produced the completion",
    )
    choices: List[ChatCompletionChoice] = Field(
        description="The completion choices",
    )
    usage: Optional[ChatCompletionUsage] = Field(
        description="Token usage for the request",
        default=None,
    )
    system_fingerprint: Optional[str] = Field(
        description="Backend configuration fingerprint",
        default=None,
    )
    service_tier: Optional[str] = Field(
        description="The service tier that fulfilled the request",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.backend.to_chat_completion_message",
        "synalinks.to_chat_completion_message",
    ]
)
def to_chat_completion_message(message):
    """Convert a synalinks `ChatMessage` to a `ChatCompletionMessage`.

    Performs the structural changes needed to match the OpenAI wire
    format: each `ToolCall`'s `function.arguments` dict is JSON-encoded
    into a string, and `Dict` content (the synalinks tool-result payload
    extension) is JSON-encoded to a string since OpenAI's `tool` role
    requires string content. The synalinks `reasoning_content` /
    `thinking_blocks` fields map onto the wire `reasoning_content` /
    `thinking_blocks` (litellm provider extensions) so reasoning survives a
    round-trip, and `name`/`refusal`/`audio` are carried through verbatim.
    A tool call whose `arguments` is already a JSON-encoded string (the
    optional wire form) is passed through without re-encoding.
    Role handling mirrors the language-model wire path: an assistant
    message with empty content emits `content=null`, and the legacy
    `function` role carries its name in `name` (falling back to
    `tool_call_id` for messages built before `name` existed).

    Args:
        message (ChatMessage): The synalinks message to convert.

    Returns:
        (ChatCompletionMessage): The OpenAI-shaped message.
    """
    role = message.role
    if isinstance(role, ChatRole):
        role = role.value
    content = message.content
    if isinstance(content, dict):
        content = json.dumps(content)

    tool_calls = None
    if message.tool_calls:
        tool_calls = [
            ChatCompletionToolCall(
                id=tc.id,
                type="function",
                function=ChatCompletionFunctionCall(
                    name=tc.function.name,
                    arguments=(
                        tc.function.arguments
                        if isinstance(tc.function.arguments, str)
                        else json.dumps(tc.function.arguments)
                    ),
                ),
            )
            for tc in message.tool_calls
        ]

    if role == "assistant":
        return ChatCompletionMessage(
            role="assistant",
            content=content if content else None,
            name=message.name or None,
            reasoning_content=message.reasoning_content or None,
            thinking_blocks=message.thinking_blocks or None,
            tool_calls=tool_calls,
            refusal=message.refusal or None,
            audio=message.audio or None,
        )
    if role == "function":
        # Legacy function role: OpenAI carries the function name in `name`.
        return ChatCompletionMessage(
            role="function",
            content=content,
            name=message.name or message.tool_call_id or "",
        )
    if role == "tool":
        return ChatCompletionMessage(
            role="tool",
            content=content,
            tool_call_id=message.tool_call_id,
        )
    # user / system / developer
    return ChatCompletionMessage(role=role, content=content, name=message.name)


@synalinks_export(
    [
        "synalinks.backend.to_chat_completion_messages",
        "synalinks.to_chat_completion_messages",
    ]
)
def to_chat_completion_messages(messages):
    """Convert a synalinks `ChatMessages` to a list of `ChatCompletionMessage`.

    Args:
        messages (ChatMessages): The synalinks messages to convert.

    Returns:
        (list[ChatCompletionMessage]): The OpenAI-shaped messages.
    """
    return [to_chat_completion_message(m) for m in messages.messages]


@synalinks_export(
    [
        "synalinks.backend.from_chat_completion_message",
        "synalinks.from_chat_completion_message",
    ]
)
def from_chat_completion_message(message):
    """Convert a `ChatCompletionMessage` to a synalinks `ChatMessage`.

    Each tool call's `function.arguments` conversion is type-driven: a
    string is the wire encoding and is parsed back into a dict, while an
    already-parsed dict is kept as-is (the `{id, type, function}` envelope
    is preserved either way). The legacy `function` role is mapped to
    `tool` (its `name` doubles as the `tool_call_id` fallback so the
    result stays linked to its call). The `reasoning_content` /
    `thinking_blocks` provider extensions are mapped back onto the
    synalinks `reasoning_content` / `thinking_blocks` fields, and `name` /
    `refusal` / `audio` onto their `ChatMessage` equivalents.

    Args:
        message (ChatCompletionMessage): The OpenAI-shaped message.

    Returns:
        (ChatMessage): The synalinks message.

    Raises:
        json.JSONDecodeError: If a tool call's `arguments` string is
            non-empty and not valid JSON.
    """
    role = message.role
    tool_call_id = message.tool_call_id
    if role == "function":
        role = "tool"
        tool_call_id = tool_call_id or message.name
    tool_calls = None
    if message.tool_calls:
        tool_calls = []
        for tc in message.tool_calls:
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    type="function",
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=arguments,
                    ),
                )
            )
    return ChatMessage(
        role=role,
        content=message.content,
        name=message.name,
        reasoning_content=message.reasoning_content,
        thinking_blocks=message.thinking_blocks,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls,
        refusal=message.refusal,
        audio=message.audio,
    )


@synalinks_export(
    [
        "synalinks.backend.from_chat_completion_messages",
        "synalinks.from_chat_completion_messages",
    ]
)
def from_chat_completion_messages(messages):
    """Convert a list of `ChatCompletionMessage` to a synalinks `ChatMessages`.

    Args:
        messages (list[ChatCompletionMessage]): The OpenAI-shaped messages.

    Returns:
        (ChatMessages): The synalinks messages.
    """
    return ChatMessages(messages=[from_chat_completion_message(m) for m in messages])


def _tool_to_chat_completion_tool(tool):
    return ChatCompletionTool(
        type="function",
        function=ChatCompletionToolFunction(
            name=tool.name,
            description=tool.description or None,
            parameters={
                "type": "object",
                "properties": dict(tool._params_schema),
                "required": list(tool._required_params),
            },
        ),
    )


@synalinks_export(
    [
        "synalinks.backend.to_chat_completion_request",
        "synalinks.to_chat_completion_request",
    ]
)
def to_chat_completion_request(messages, model, *, tools=None, **kwargs):
    """Build a full `ChatCompletionRequest` from synalinks inputs.

    Converts `messages` to the OpenAI shape and, when `tools` is given,
    introspects each synalinks `Tool` to emit the `function.parameters`
    JSON Schema OpenAI expects. Any extra keyword arguments are forwarded
    to `ChatCompletionRequest` (e.g. `temperature`, `response_format`,
    `tool_choice`, `stream`, `reasoning_effort`, `parallel_tool_calls`).

    Args:
        messages (ChatMessages): The synalinks conversation.
        model (str): The target model id (e.g. `"gpt-4o-mini"`).
        tools (Iterable | Mapping | None): Optional synalinks tool
            callables. A `{name: Tool}` mapping is also accepted (its
            values are used). Each tool must expose `name`,
            `description`, `_params_schema`, and `_required_params`
            (synalinks `Tool` instances do).
        **kwargs: Forwarded to `ChatCompletionRequest`.

    Returns:
        (ChatCompletionRequest): The OpenAI-shaped request body.
    """
    oa_messages = to_chat_completion_messages(messages)
    oa_tools = None
    if tools:
        if isinstance(tools, dict):
            tools = tools.values()
        oa_tools = [_tool_to_chat_completion_tool(t) for t in tools]
    return ChatCompletionRequest(
        model=model,
        messages=oa_messages,
        tools=oa_tools,
        **kwargs,
    )
