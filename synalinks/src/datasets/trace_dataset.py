# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Load the recordings of `synalinks.record_traces()` as a dataset."""

import hashlib
import json
import os
from collections import defaultdict
from typing import Optional

import orjson

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.config import trace_recording_dir
from synalinks.src.datasets.dataset import Dataset

_DEFAULT_INPUT_TEMPLATE = '{"messages": {{ messages | tojson }}}'
_DEFAULT_OUTPUT_TEMPLATE = "{{ completion | tojson }}"


def _conversation_key(messages):
    return hashlib.sha256(orjson.dumps(messages, option=orjson.OPT_SORT_KEYS)).hexdigest()


@synalinks_export(["synalinks.TraceDataset", "synalinks.datasets.TraceDataset"])
class TraceDataset(Dataset):
    """Streaming dataset over the recordings of `synalinks.record_traces()`.

    Reads the folders the `synalinks.hooks.Recorder` writes (one per program
    and originating module) and yields one example per recorded LM call:
    by default the input is the `ChatMessages` the LM received and the
    target is the `ChatMessage` it answered. Use it to evaluate a program
    on recorded traffic, to distill a large model into a smaller one, or
    to inspect what was recorded.

    Examples recorded in any `Recorder` format (`"openai"`, `"tinker"`,
    `"customizer"`) are read back as chat-completion messages: a `<think>`
    prefix (`"tinker"`) or a separate `thinking` message (`"customizer"`)
    becomes `reasoning_content`, and object tool-call arguments become JSON
    strings. The tool-call ids the `"customizer"` format drops are
    renumbered, and each tool result is linked back to its call in order.

    ```python
    import synalinks

    dataset = synalinks.datasets.TraceDataset(
        program="my_program",
        module="generator",
        deduplicate=True,
        batch_size=8,
    )
    await program.evaluate(x=dataset())
    ```

    Every example is also given to the templates with its recorded
    metadata, so `input_template` / `output_template` can remap it onto a
    program's own data models. The template variables are `messages` (the
    input messages), `completion` (the assistant's answer), `tools` (the
    tools the LM was given, `[]` if none) and the `metadata.jsonl` keys
    (`program`, `module`, `outputs`, `schema`, `usage`, `cost`,
    `timestamp`, `session_id`, `call_id`, `parent_call_id`, `inputs_hash`,
    `config_hash`, `format`). For example, to train a module on the
    structured answers it gave:

    ```python
    dataset = synalinks.datasets.TraceDataset(
        module="generator",
        output_data_model=Answer,
        output_template="{{ outputs | tojson }}",
    )
    ```

    To give each example the tools its LM call had, pass
    `tools_template="{{ tools | tojson }}"`.

    `deduplicate` and `drop_prefixes` read a whole module folder before
    yielding its examples; otherwise the files are streamed line by line.

    Args:
        root (str): The recording root. Defaults to the one of
            `synalinks.record_traces()` (`./.synalinks` unless set).
        program (str): Optional. Only read this program's recordings.
        module (str): Optional. Only read this module's recordings: its
            name, or its name-scope path below the program
            (e.g. `"function_calling_agent/tool_calls_generator"`).
        include_unsupported (bool): Also read the complete conversations
            the recording format could not express (kept in `failed.jsonl`
            with an `unsupported` reason, e.g. tool calls in the `"tinker"`
            format). Defaults to `False`. Failed calls, which have no
            completion, are never read.
        config_hash (str): Optional. Only read examples produced by this
            `LanguageModel` configuration.
        session_id (str): Optional. Only read examples of this recording
            session.
        since (float): Optional. Only read examples recorded at or after
            this time (seconds since the epoch).
        until (float): Optional. Only read examples recorded before this
            time (seconds since the epoch).
        deduplicate (bool): Keep only the latest example of each set of
            identical inputs (same `inputs_hash`) within a module.
            Defaults to `False`.
        drop_prefixes (bool): Drop the examples whose whole conversation
            is the beginning of another one of the same module: the
            earlier steps of an agent trajectory, each repeated in the
            next. Defaults to `False`.
        input_data_model (DataModel): See `Dataset`. Defaults to
            `ChatMessages`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`. Defaults to the input
            messages.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `ChatMessage`.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`. Defaults to the completion.
        tools_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. Caps how many examples are read. With a
            limit set, `__len__` is also available.
        repeat (int): See `Dataset`.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        *,
        program: Optional[str] = None,
        module: Optional[str] = None,
        include_unsupported: bool = False,
        config_hash: Optional[str] = None,
        session_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        deduplicate: bool = False,
        drop_prefixes: bool = False,
        input_data_model=None,
        input_schema=None,
        input_template: Optional[str] = None,
        output_data_model=None,
        output_schema=None,
        output_template: Optional[str] = None,
        tools_template: Optional[str] = None,
        batch_size: int = 1,
        limit: Optional[int] = None,
        repeat: int = 1,
    ):
        super().__init__(
            input_data_model=input_data_model,
            input_schema=input_schema,
            input_template=input_template or _DEFAULT_INPUT_TEMPLATE,
            output_data_model=output_data_model,
            output_schema=output_schema,
            output_template=output_template or _DEFAULT_OUTPUT_TEMPLATE,
            tools_template=tools_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )
        self.root = root or trace_recording_dir()
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Recording root not found: {self.root}")
        self.program = program
        self.module = module
        self.include_unsupported = include_unsupported
        self.config_hash = config_hash
        self.session_id = session_id
        self.since = since
        self.until = until
        self.deduplicate = deduplicate
        self.drop_prefixes = drop_prefixes

    @staticmethod
    def _row(metadata, messages, tools):
        return {
            **{key: value for key, value in metadata.items() if key.isidentifier()},
            "messages": messages[:-1],
            "completion": messages[-1],
            "tools": tools,
        }

    def _iter_rows(self):
        buffered = self.deduplicate or self.drop_prefixes
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()
            if "train.jsonl" not in filenames and not (
                self.include_unsupported and "failed.jsonl" in filenames
            ):
                continue
            parts = os.path.relpath(dirpath, self.root).split(os.sep)
            if self.program is not None and parts[0] != self.program:
                continue
            if self.module is not None and self.module not in (
                parts[-1],
                "/".join(parts[1:]),
            ):
                continue
            # Metadata, keyed by the hash of the train.jsonl line it describes.
            pending = defaultdict(list)
            if "metadata.jsonl" in filenames:
                with open(os.path.join(dirpath, "metadata.jsonl"), "rb") as f:
                    for line in f:
                        if line.strip():
                            metadata = orjson.loads(line)
                            pending[metadata.get("line_hash")].append(metadata)
            sources = [("train.jsonl", True)]
            if self.include_unsupported:
                sources.append(("failed.jsonl", False))
            examples = []
            for filename, is_train in sources:
                if filename not in filenames:
                    continue
                with open(os.path.join(dirpath, filename), "rb") as f:
                    for line in f:
                        line = line.rstrip(b"\n")
                        if not line.strip():
                            continue
                        if is_train:
                            queue = pending.get(hashlib.sha256(line).hexdigest())
                            metadata = queue.pop(0) if queue else {}
                            row = orjson.loads(line)
                            messages = row["messages"]
                            tools = row.get("tools") or []
                        else:
                            metadata = orjson.loads(line)
                            if "unsupported" not in metadata:
                                continue
                            messages = metadata.pop("messages")
                            tools = metadata.pop("tools", [])
                        format = metadata.get("format") if is_train else "openai"
                        if format == "tinker":
                            restored = []
                            for message in messages:
                                content = message.get("content") or ""
                                if message["role"] == "assistant" and content.startswith(
                                    "<think>\n"
                                ):
                                    reasoning, _, content = content[
                                        len("<think>\n") :
                                    ].partition("\n</think>\n")
                                    message = {
                                        **message,
                                        "content": content,
                                        "reasoning_content": reasoning,
                                    }
                                restored.append(message)
                            messages = restored
                        elif format == "customizer":
                            restored = []
                            reasoning = None
                            call_ids = []
                            for message in messages:
                                if "thinking" in message:
                                    reasoning = message["thinking"]
                                    continue
                                message = {"content": "", **message}
                                if (
                                    reasoning is not None
                                    and message["role"] == "assistant"
                                ):
                                    message["reasoning_content"] = reasoning
                                    reasoning = None
                                tool_calls = []
                                for tool_call in message.get("tool_calls") or []:
                                    call_ids.append(f"call_{len(call_ids)}")
                                    arguments = tool_call["function"]["arguments"]
                                    if not isinstance(arguments, str):
                                        arguments = json.dumps(arguments)
                                    tool_calls.append(
                                        {
                                            "id": call_ids[-1],
                                            "type": "function",
                                            "function": {
                                                "name": tool_call["function"]["name"],
                                                "arguments": arguments,
                                            },
                                        }
                                    )
                                if tool_calls:
                                    message["tool_calls"] = tool_calls
                                if message["role"] == "tool" and call_ids:
                                    message["tool_call_id"] = call_ids.pop(0)
                                restored.append(message)
                            if reasoning is not None:
                                restored.append(
                                    {
                                        "role": "assistant",
                                        "content": "",
                                        "reasoning_content": reasoning,
                                    }
                                )
                            messages = restored
                        timestamp = metadata.get("timestamp")
                        if (
                            not messages
                            or messages[-1].get("role") != "assistant"
                            or (
                                self.config_hash is not None
                                and metadata.get("config_hash") != self.config_hash
                            )
                            or (
                                self.session_id is not None
                                and metadata.get("session_id") != self.session_id
                            )
                            or (
                                self.since is not None
                                and (timestamp is None or timestamp < self.since)
                            )
                            or (
                                self.until is not None
                                and (timestamp is None or timestamp >= self.until)
                            )
                        ):
                            continue
                        if buffered:
                            examples.append((metadata, messages, tools))
                        else:
                            yield self._row(metadata, messages, tools)
            if self.deduplicate:
                latest = {}
                for example in examples:
                    key = example[0].get("inputs_hash") or _conversation_key(
                        example[1][:-1]
                    )
                    latest.pop(key, None)
                    latest[key] = example
                examples = list(latest.values())
            if self.drop_prefixes:
                prefixes = {
                    _conversation_key(example[1][:end])
                    for example in examples
                    for end in range(1, len(example[1]))
                }
                examples = [
                    example
                    for example in examples
                    if _conversation_key(example[1]) not in prefixes
                ]
            for example in examples:
                yield self._row(*example)

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "Recorded traces have unknown length. "
                "Pass `limit=...` if you need __len__."
            )
        return self._total_batches(self.limit)
