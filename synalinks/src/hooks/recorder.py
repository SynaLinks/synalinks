# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import hashlib
import json
import os
import time
import uuid

import orjson

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.backend.common.name_scope import current_path
from synalinks.src.backend.config import trace_recording_dir
from synalinks.src.backend.config import trace_recording_format
from synalinks.src.hooks.hook import Hook
from synalinks.src.saving import serialization_lib
from synalinks.src.version import __version__

FORMATS = ("openai", "tinker", "customizer")

# One id per process, shared by every Recorder: tells runs apart in metadata.
_SESSION_ID = uuid.uuid4().hex


@synalinks_export("synalinks.hooks.format_conversation")
def format_conversation(messages, tools=None, format="openai"):
    """Format a chat-completion conversation as one fine-tuning example.

    This is how the `Recorder` writes each line of its `train.jsonl` files.
    The formats were checked against the frameworks' own loaders:

    - `"openai"`: `{"messages": [...], "tools": [...]}` with the
        chat-completion messages as is (`reasoning_content`, `tool_calls`
        with JSON-string `arguments`, `role: "tool"` results with
        `tool_call_id`). Loads in NVIDIA NeMo AutoModel (`ChatDataset`)
        and NeMo RL (`openai_format`), which render the tools, reasoning,
        tool calls and tool results.
    - `"tinker"`: `{"messages": [...]}` with `role` and `content` only,
        reasoning prefixed to the content as `<think>...</think>` (Tinker
        drops `reasoning_content`). Loads in Tinker's
        `FromConversationFileBuilder`, which cannot load tool calls from a
        file: conversations with tool calls are not representable.
    - `"customizer"`: `{"messages": [...], "tools": [...]}` restricted to
        the keys NVIDIA NeMo Customizer validates (`role`, `content`,
        `thinking`, `tool_calls`): reasoning becomes a separate `thinking`
        message, tool-call `arguments` a JSON object, and `tool_call_id`
        is dropped.

    `"tinker"` and `"customizer"` only take text content: a conversation
    with image or audio parts is not representable in them.

    Args:
        messages (list): Chat-completion message dicts.
        tools (list): Optional chat-completion tool declarations.
        format (str): One of `"openai"`, `"tinker"`, `"customizer"`.

    Returns:
        (tuple): `(row, reason)`: the example as a dict, or `None` and the
            reason the conversation is not representable in `format`.
    """
    if format not in FORMATS:
        raise ValueError(f"`format` must be one of {FORMATS}, got {format!r}")
    if format == "openai":
        row = {"messages": messages}
        if tools:
            row["tools"] = tools
        return row, None
    formatted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get("type") != "text" for p in content):
                return None, f"the {format} format only takes text content"
            content = "".join(p.get("text", "") for p in content)
        content = content or ""
        reasoning = message.get("reasoning_content")
        if format == "tinker":
            if message.get("tool_calls") or message["role"] == "tool":
                return None, "Tinker cannot load tool calls from a file"
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n{content}"
            formatted.append({"role": message["role"], "content": content})
            continue
        if reasoning:
            formatted.append({"role": message["role"], "thinking": reasoning})
        converted = {"role": message["role"]}
        if content or message["role"] != "assistant":
            converted["content"] = content
        tool_calls = []
        for tool_call in message.get("tool_calls") or []:
            try:
                arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):
                return None, "Customizer needs tool-call arguments as a JSON object"
            tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": arguments,
                    },
                }
            )
        if tool_calls:
            converted["tool_calls"] = tool_calls
        if len(converted) > 1:
            formatted.append(converted)
    row = {"messages": formatted}
    if tools and format == "customizer":
        row["tools"] = tools
    return row, None


@synalinks_export("synalinks.hooks.Recorder")
class Recorder(Hook):
    """Recorder hook that writes `LanguageModel` calls as fine-tuning data.

    Attach it to a `LanguageModel` (or enable it everywhere with
    `synalinks.record_traces()`) and every concrete call (the messages sent
    to the LM, the tools it was given, and the completion it returned) is
    written under `base_dir` (default `./.synalinks` in the working
    directory). Attached to any other module type, it records nothing.
    Symbolic calls (made while building the graph) are not recorded.

    Because a `LanguageModel` is usually shared by several modules
    (`Generator`, `ChainOfThought`, agents ...), each call is attributed to
    the module it originates from (the module whose `call()` invoked the
    LM), with one folder per program and per originating module, nested
    along the module's name scope:

    ```
    ./.synalinks/
    └── program_name/
        └── module_name/            # e.g. function_calling_agent/tool_calls_generator
            ├── train.jsonl         # one fine-tuning example per line
            ├── metadata.jsonl      # one line per train.jsonl line
            └── failed.jsonl        # failed, streamed or not representable calls
    ```

    The program folder is named after the entry module of the call context
    (usually the `Program`). When the LM is called directly, the files land
    in `<base_dir>/<lm_name>/`. Files are appended to across runs.

    `train.jsonl` holds only complete conversations (ending with the
    assistant's completion) and only the keys of the target `format` (see
    `format_conversation`), so it can be given as is to a fine-tuning
    framework: NVIDIA NeMo AutoModel or NeMo RL (`"openai"`, the default),
    Tinker's `FromConversationFileBuilder` (`"tinker"`), or NVIDIA NeMo
    Customizer (`"customizer"`). A structured completion is the JSON string
    of its fields. When a provider emits reasoning only as opaque
    `thinking_blocks` (e.g. Anthropic), their text is used as the reasoning.

    Each `metadata.jsonl` line describes the `train.jsonl` line with the
    same `line_hash` (the SHA-256 of that line):

    - `synalinks_version`: The version of Synalinks that wrote the record.
    - `session_id`: The id of the recording process, shared by its calls.
    - `format`: The format of the `train.jsonl` line.
    - `call_id`: The unique id of the LM call.
    - `parent_call_id`: The id of the parent module's call (`None` if the
        LM is the entry module).
    - `program`: The name of the entry module (usually the `Program`).
    - `module`: The name of the module the LM call originates from.
    - `timestamp`: The time the call began (seconds since the epoch).
    - `duration`: The call duration in seconds.
    - `usage`: The token usage of the call (`prompt_tokens`,
        `completion_tokens`, `total_tokens`, `cached_tokens`,
        `cache_creation_tokens`, `reasoning_tokens`). All zeros on a cache
        hit; `None` if the call failed, was streamed, or was answered by
        the fallback (whose own record carries the usage).
    - `cost`: The cost of the call in USD as reported by the provider
        (`0.0` for local models and cache hits, `None` when `usage` is).
    - `cache_hit`: `True` when the response came from the LM's file cache
        (only present then).
    - `fallback`: `True` when every attempt failed and the LM's fallback
        answered (only present then).
    - `inputs_hash`: The SHA-256 hex digest of the input messages (with
        sorted keys), to deduplicate examples.
    - `line_hash`: The SHA-256 hex digest of the `train.jsonl` line.
    - `outputs`: The completion returned by the LM as JSON.
    - `schema`: The JSON schema the completion was constrained to (`None`
        for a free-form call).
    - `config`: The serialized `LanguageModel` (as given by
        `synalinks.saving.serialize_synalinks_object`).
    - `config_hash`: The SHA-256 hex digest of the `config` JSON (with
        sorted keys), to group examples by model configuration.

    A `failed.jsonl` line has the same keys (without `line_hash`), plus the
    conversation in chat-completion format (`messages`, and `tools` when the
    call had tools) and one of:

    - `exception`: The error, when the call failed (including when every
        attempt failed and no fallback answered) or was streamed.
    - `unsupported`: Why the conversation is not representable in the
        target `format` (e.g. tool calls in the `"tinker"` format).

    Example:

    ```python
    import synalinks

    # Enable recording globally
    synalinks.record_traces()

    # Or attach a Recorder hook directly to a LanguageModel
    lm = synalinks.LanguageModel(
        model="ollama/mistral",
        hooks=[synalinks.hooks.Recorder(format="tinker")],
    )

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="my_program")

    await program(Query(query="What is the capital of France?"))
    # => ./.synalinks/my_program/generator/train.jsonl (+ metadata.jsonl)
    ```

    Args:
        base_dir (str): The root folder where the records are written.
            If None, uses the value from `synalinks.record_traces()`, else
            `./.synalinks` in the working directory.
        format (str): The format of the `train.jsonl` files: `"openai"`,
            `"tinker"` or `"customizer"` (see `format_conversation`). If
            None, uses the value from `synalinks.record_traces()`, else
            `"openai"`.
    """

    def __init__(self, base_dir=None, format=None):
        super().__init__()
        self.base_dir = base_dir or trace_recording_dir()
        self.format = format or trace_recording_format()
        if self.format not in FORMATS:
            raise ValueError(f"`format` must be one of {FORMATS}, got {self.format!r}")
        # call_id -> context captured at call begin, while the call context
        # and the originating module's name scope are still open.
        self._pending = {}
        self._serialized_module = None

    def _is_language_model(self):
        from synalinks.src.modules.language_models.language_model import LanguageModel

        return isinstance(self.module, LanguageModel)

    def _serialize_data(self, data):
        leaves = [
            d for d in tree.flatten(data) if d is not None and hasattr(d, "get_json")
        ]
        if not leaves:
            return None
        jsons = [d.get_json() for d in leaves]
        return jsons[0] if len(jsons) == 1 else jsons

    # `current_call_usage()` key -> record `usage` key.
    _USAGE_KEYS = {
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
        "cache_read_input_tokens": "cached_tokens",
        "cache_creation_input_tokens": "cache_creation_tokens",
        "reasoning_tokens": "reasoning_tokens",
    }

    # Chat-completion message keys kept in the recorded conversations.
    _MESSAGE_KEYS = (
        "role",
        "content",
        "reasoning_content",
        "tool_calls",
        "tool_call_id",
        "name",
    )

    @classmethod
    def _sanitize_message(cls, message):
        sanitized = {
            key: message[key] for key in cls._MESSAGE_KEYS if message.get(key) is not None
        }
        # Reasoning emitted only as thinking blocks: keep their text.
        if "reasoning_content" not in sanitized:
            thinking_texts = [
                block["thinking"]
                for block in message.get("thinking_blocks") or []
                if isinstance(block, dict) and block.get("thinking")
            ]
            if thinking_texts:
                sanitized["reasoning_content"] = "\n".join(thinking_texts)
        return sanitized

    @classmethod
    def _to_assistant_message(cls, serialized_outputs):
        """Convert the LM outputs into an OpenAI-format assistant message.

        A plain LM call already returns a chat-completion message; a
        schema-constrained call returns the parsed structured output,
        which is re-serialized as the assistant content.
        """
        if serialized_outputs is None:
            return None
        if isinstance(serialized_outputs, dict) and "role" in serialized_outputs:
            return cls._sanitize_message(serialized_outputs)
        return {
            "role": "assistant",
            "content": orjson.dumps(serialized_outputs, default=str).decode(),
        }

    @staticmethod
    def _hash_data(data):
        if data is None:
            return None
        return hashlib.sha256(
            orjson.dumps(data, option=orjson.OPT_SORT_KEYS, default=str)
        ).hexdigest()

    def _get_serialized_module(self):
        if self._serialized_module is None:
            try:
                self._serialized_module = serialization_lib.serialize_synalinks_object(
                    self.module
                )
            except Exception as e:
                self._serialized_module = {
                    "class_name": self.module.__class__.__name__,
                    "name": self.module.name,
                    "error": f"Serialization failed: {e}",
                }
        return self._serialized_module

    @staticmethod
    def _append(directory, filename, line):
        with open(os.path.join(directory, filename), "ab") as f:
            f.write(line + b"\n")

    def on_call_begin(
        self,
        call_id,
        parent_call_id=None,
        inputs=None,
        kwargs=None,
    ):
        if not self._is_language_model():
            return
        if any_symbolic_data_models(inputs):
            return
        from synalinks.src.modules.language_models.language_model import _tool_to_wire

        call_context = self.module._get_call_context()
        entry_module = call_context.entry_module if call_context else self.module
        # The originating module's name scope is still open at this point.
        origin_path = current_path()
        serialized_inputs = self._serialize_data(inputs)
        input_messages = []
        if isinstance(serialized_inputs, dict):
            input_messages = [
                self._sanitize_message(m)
                for m in serialized_inputs.get("messages") or []
                if isinstance(m, dict)
            ]
        kwargs = kwargs or {}
        tools = kwargs.get("tools") or []
        if isinstance(tools, dict):
            tools = tools.values()
        tools = [_tool_to_wire(t) for t in tools] + list(kwargs.get("tool_schemas") or [])
        # `parent_call_id` is stashed: the exception path of `Module.__call__`
        # invokes `on_call_end` without it.
        self._pending[call_id] = {
            "program_name": entry_module.name,
            "origin_path": origin_path.split("/") if origin_path else [],
            "parent_call_id": parent_call_id,
            "timestamp": time.time(),
            "input_messages": input_messages,
            "tools": tools,
            "schema": kwargs.get("schema"),
        }

    def on_call_end(
        self,
        call_id,
        parent_call_id=None,
        outputs=None,
        exception=None,
    ):
        pending = self._pending.pop(call_id, None)
        if pending is None:
            return
        from synalinks.src.modules.language_models.language_model import (
            current_call_usage,
        )

        # The per-call usage of this task, not the LM's `last_call_*` counters,
        # which keep the previous call's values on cache hits and fallbacks.
        call_usage = current_call_usage() or {}
        usage = None
        cost = None
        if call_usage.get("cache_hit"):
            usage = dict.fromkeys(self._USAGE_KEYS.values(), 0)
            cost = 0.0
        elif not exception and "error" not in call_usage and call_usage:
            usage = {
                record_key: call_usage.get(usage_key) or 0
                for usage_key, record_key in self._USAGE_KEYS.items()
            }
            cost = call_usage.get("cost") or 0.0
        if not exception and "error" in call_usage and not call_usage.get("fallback"):
            # Every attempt failed and the call returned `None` without raising.
            exception = call_usage["error"]
        serialized_outputs = self._serialize_data(outputs)
        assistant_message = self._to_assistant_message(serialized_outputs)
        if not exception and assistant_message is None:
            exception = "No completion to record (streamed response)."
        messages = list(pending["input_messages"])
        if assistant_message is not None:
            messages.append(assistant_message)
        origin_path = pending["origin_path"] or [self.module.name]
        if pending["program_name"] == origin_path[0]:
            origin_path = origin_path[1:]
        directory = os.path.join(self.base_dir, pending["program_name"], *origin_path)
        os.makedirs(directory, exist_ok=True)
        metadata = {
            "synalinks_version": __version__,
            "session_id": _SESSION_ID,
            "format": self.format,
            "call_id": call_id,
            "parent_call_id": pending["parent_call_id"],
            "program": pending["program_name"],
            "module": origin_path[-1] if origin_path else pending["program_name"],
            "timestamp": pending["timestamp"],
            "duration": time.time() - pending["timestamp"],
            "usage": usage,
            "cost": cost,
        }
        if call_usage.get("cache_hit"):
            metadata["cache_hit"] = True
        if call_usage.get("fallback"):
            metadata["fallback"] = True
        metadata.update(
            {
                "inputs_hash": self._hash_data(pending["input_messages"]),
                "outputs": serialized_outputs,
                "schema": pending["schema"],
                "config": self._get_serialized_module(),
                "config_hash": self._hash_data(self._get_serialized_module()),
            }
        )
        row = None
        unsupported = None
        if not exception:
            row, unsupported = format_conversation(
                messages, tools=pending["tools"], format=self.format
            )
        if row is not None:
            line = orjson.dumps(row, default=str)
            metadata["line_hash"] = hashlib.sha256(line).hexdigest()
            self._append(directory, "train.jsonl", line)
            self._append(directory, "metadata.jsonl", orjson.dumps(metadata, default=str))
            return
        metadata["messages"] = messages
        if pending["tools"]:
            metadata["tools"] = pending["tools"]
        if exception:
            metadata["exception"] = exception
        else:
            metadata["unsupported"] = unsupported
        self._append(directory, "failed.jsonl", orjson.dumps(metadata, default=str))
