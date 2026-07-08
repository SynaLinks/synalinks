# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import hashlib
import os
import time

import orjson

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.backend.common.name_scope import current_path
from synalinks.src.backend.config import trace_recording_dir
from synalinks.src.hooks.hook import Hook
from synalinks.src.saving import serialization_lib
from synalinks.src.version import __version__


@synalinks_export("synalinks.hooks.Recorder")
class Recorder(Hook):
    """Recorder hook that writes `LanguageModel` calls to JSONL files.

    This hook is designed to collect training data: attach it to a
    `LanguageModel` and every concrete call (the chat messages sent to the
    LM and the completion it returned) is appended as one JSON line to a
    file under `base_dir` (default `synalinks_home()`, i.e.
    `$SYNALINKS_HOME` or `~/.synalinks`). Attached to any other module
    type, it records nothing.

    Each record carries the full conversation in a top-level `messages`
    key in OpenAI chat format (completion included as the final assistant
    message), which is the chat dataset schema of OpenAI-compatible
    fine-tuning stacks such as NVIDIA NeMo (NeMo Customizer chat datasets,
    NeMo Framework SFT loaders return the extra metadata keys as-is). For
    a consumer that rejects unknown keys, project the `messages` field
    (e.g. `jq -c '{messages}' *.jsonl`).

    Because a `LanguageModel` is usually shared by several modules
    (`Generator`, `ChainOfThought`, agents ...), each record is attributed
    to the module the call originates from (the module whose `call()`
    invoked the LM), and the files are organized as one folder per program
    and one subfolder per originating module:

    ```
    ~/.synalinks/
    └── program_name/
        └── module_name/
            └── module_name_TIMESTAMP.jsonl
    ```

    The program folder is named after the entry module of the call context
    (the outermost module of the call stack, usually the `Program`). When
    the LM is called directly (no enclosing module), the records land in
    `<base_dir>/<lm_name>/<lm_name>_TIMESTAMP.jsonl`. Symbolic calls (made
    while building the graph) are not recorded.

    Each line is a JSON object with the following keys:

    - `synalinks_version`: The version of Synalinks that wrote the record.
    - `call_id`: The unique id of the LM call.
    - `parent_call_id`: The id of the parent module's call (`None` if the
        LM is the entry module).
    - `program`: The name of the entry module (usually the `Program`).
    - `module`: The name of the module the LM call originates from.
    - `timestamp`: The time the call began (seconds since the epoch).
    - `duration`: The call duration in seconds.
    - `usage`: The token usage of the call (`prompt_tokens`,
        `completion_tokens`, `total_tokens`, `cached_tokens`,
        `cache_creation_tokens`, `reasoning_tokens`; `None` if the call
        failed).
    - `cost`: The cost of the call in USD as reported by the provider
        (`0.0` for local models, `None` if the call failed).
    - `messages`: The conversation in OpenAI chat format: the chat
        messages sent to the LM, with the completion appended as the final
        assistant message (a structured completion is appended as its JSON
        string). Nothing is appended if the call failed or the response
        was streamed. Messages carry the chat-completion keys (`role`,
        `content`, `reasoning_content`, `tool_calls`, `tool_call_id`,
        `name`) — `reasoning_content` is kept because fine-tuning stacks
        with reasoning support (e.g. NeMo) train on it. When a provider
        emits reasoning only as opaque `thinking_blocks` (e.g.
        Anthropic), their text is folded into `reasoning_content`; the
        raw blocks stay in `outputs` only.
    - `inputs_hash`: The SHA-256 hex digest of the input messages (the
        conversation without the appended completion, with sorted keys),
        to deduplicate records.
    - `outputs`: The completion returned by the LM as JSON (`None` if the
        call failed or the response was streamed).
    - `config`: The serialized `LanguageModel` (as given by
        `synalinks.saving.serialize_synalinks_object`).
    - `config_hash`: The SHA-256 hex digest of the `config` JSON (with
        sorted keys), to group records by module configuration.
    - `exception`: The exception message (only present if the call failed).

    You can enable recording for every module by using
    `synalinks.record_traces()` at the beginning of your scripts.

    Example:

    ```python
    import synalinks

    # Enable recording globally
    synalinks.record_traces()

    # Or attach a Recorder hook directly to a LanguageModel
    lm = synalinks.LanguageModel(
        model="ollama/mistral",
        hooks=[synalinks.hooks.Recorder()],
    )

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="my_program")

    await program(Query(query="What is the capital of France?"))
    # => ~/.synalinks/my_program/generator/generator_20260703-101530.jsonl
    ```

    Args:
        base_dir (str): The root folder where the records are written.
            If None, uses the value from
            `synalinks.record_traces()` or `synalinks_home()`.
    """

    def __init__(self, base_dir=None):
        super().__init__()
        self.base_dir = base_dir or trace_recording_dir()
        # call_id -> record context captured at call begin: the
        # `CallContext` (and its `entry_module`) is already reset when the
        # entry module's `on_call_end` fires, and the name scope of the
        # originating module is only open during the call itself, so both
        # must be resolved at `on_call_begin`.
        self._pending = {}
        # (program_name, origin_name) -> jsonl filepath (the timestamp is
        # fixed at the first record so all calls of one session land in
        # the same file).
        self._filepaths = {}
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

    # Chat-completion message keys kept in `messages` for fine-tuning
    # consumers: the base OpenAI keys plus `reasoning_content`, which the
    # NeMo stack supports natively for reasoning models (trained on by
    # default; `mask_reasoning_content` excludes it from the loss).
    # Provider-native `thinking_blocks` are not carried as-is (opaque,
    # signed, non-standard) but their text is folded into
    # `reasoning_content` when the provider emitted no reasoning text —
    # the full-fidelity message stays in `outputs`.
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
        # Anthropic-style providers may emit reasoning only as signed
        # thinking blocks: recover their text as `reasoning_content`
        # instead of dropping it (redacted blocks carry no text).
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

    def _get_filepath(self, program_name, origin_name):
        filepath = self._filepaths.get((program_name, origin_name))
        if filepath is None:
            if program_name == origin_name:
                # The LM was called directly by the entry module (or is
                # the entry module itself): no subfolder needed.
                directory = os.path.join(self.base_dir, program_name)
            else:
                directory = os.path.join(self.base_dir, program_name, origin_name)
            os.makedirs(directory, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = os.path.join(directory, f"{origin_name}_{timestamp}.jsonl")
            self._filepaths[(program_name, origin_name)] = filepath
        return filepath

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
        call_context = self.module._get_call_context()
        entry_module = call_context.entry_module if call_context else self.module
        # The name scope of the module whose `call()` invoked the LM is
        # still open at this point: its innermost component is the
        # originating module.
        origin_path = current_path()
        origin_name = origin_path.rsplit("/", 1)[-1] if origin_path else self.module.name
        serialized_inputs = self._serialize_data(inputs)
        input_messages = []
        if isinstance(serialized_inputs, dict):
            input_messages = [
                self._sanitize_message(m)
                for m in serialized_inputs.get("messages") or []
                if isinstance(m, dict)
            ]
        # `parent_call_id` is stashed too: the exception path of
        # `Module.__call__` invokes `on_call_end` without it.
        self._pending[call_id] = {
            "program_name": entry_module.name,
            "origin_name": origin_name,
            "parent_call_id": parent_call_id,
            "timestamp": time.time(),
            "input_messages": input_messages,
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
        # The `last_call_*` counters are set by the litellm call that just
        # returned. With concurrent LM calls awaited on the same event loop
        # they could in principle be overwritten by an interleaved call,
        # but nothing yields between the counter update and this hook in
        # the common case.
        usage = None
        cost = None
        if not exception:
            usage = {
                "prompt_tokens": self.module.last_call_prompt_tokens,
                "completion_tokens": self.module.last_call_completion_tokens,
                "total_tokens": self.module.last_call_tokens,
                "cached_tokens": self.module.last_call_cached_tokens,
                "cache_creation_tokens": self.module.last_call_cache_creation_tokens,
                "reasoning_tokens": self.module.last_call_reasoning_tokens,
            }
            cost = self.module.last_call_cost
        serialized_outputs = self._serialize_data(outputs)
        assistant_message = self._to_assistant_message(serialized_outputs)
        messages = list(pending["input_messages"])
        if assistant_message is not None:
            messages.append(assistant_message)
        record = {
            "synalinks_version": __version__,
            "call_id": call_id,
            "parent_call_id": pending["parent_call_id"],
            "program": pending["program_name"],
            "module": pending["origin_name"],
            "timestamp": pending["timestamp"],
            "duration": time.time() - pending["timestamp"],
            "usage": usage,
            "cost": cost,
            "messages": messages,
            "inputs_hash": self._hash_data(pending["input_messages"]),
            "outputs": serialized_outputs,
            "config": self._get_serialized_module(),
            "config_hash": self._hash_data(self._get_serialized_module()),
        }
        if exception:
            record["exception"] = exception
        filepath = self._get_filepath(pending["program_name"], pending["origin_name"])
        with open(filepath, "ab") as f:
            f.write(orjson.dumps(record, default=str) + b"\n")
