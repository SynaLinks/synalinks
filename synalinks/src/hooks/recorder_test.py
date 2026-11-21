# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import hashlib
import json
import os
from unittest.mock import patch

import orjson

from synalinks.src import testing
from synalinks.src import version
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.backend import config
from synalinks.src.hooks.recorder import Recorder
from synalinks.src.hooks.recorder import format_conversation
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.identity import Identity
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs.program import Program


class Query(DataModel):
    query: str


class AnswerWithRationale(DataModel):
    rationale: str
    answer: str


EXPECTED_STRING = """{"rationale": "Toulouse hosts ISAE-SUPAERO", "answer": "Toulouse"}"""


async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The expression to calculate, such as '2 + 2'.
    """
    return {"result": 30.0}


def _response(content, reasoning_content=None, tool_calls=None, usage=None):
    message = {"content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for i, (name, arguments) in enumerate(tool_calls)
        ]
    response = {"choices": [{"message": message}]}
    if usage:
        response["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        }
    return response


def _hello():
    return ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="Hello")])


class RecorderTest(testing.TestCase):
    def _read(self, *path):
        filepath = os.path.join(*path)
        if not os.path.exists(filepath):
            return []
        with open(filepath, "rb") as f:
            return [orjson.loads(line) for line in f]

    async def _call_twice(self, language_model):
        await language_model(_hello())
        await language_model(_hello())

    async def _run_agent(self, mock_completion, base_dir, format):
        mock_completion.side_effect = [
            _response(
                "",
                reasoning_content="I should use the calculator.",
                tool_calls=[("calculate", {"expression": "10 + 20"})],
            ),
            _response("The result is 30.", reasoning_content="The tool returned 30."),
        ]
        language_model = LanguageModel(
            model="openai/gpt-4o-mini",
            hooks=[Recorder(base_dir=base_dir, format=format)],
        )
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=[Tool(calculate)],
            autonomous=True,
            max_iterations=3,
            name="calc_agent",
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs, name="agent_program")
        await program(
            ChatMessages(messages=[ChatMessage(role="user", content="10 + 20?")])
        )
        return os.path.join(
            base_dir,
            "agent_program",
            "calc_agent",
            "tool_calls_generator_calc_agent",
        )

    @patch("litellm.acompletion")
    async def test_records_lm_calls_per_originating_module(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": EXPECTED_STRING}}],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35,
                "prompt_tokens_details": {
                    "cached_tokens": 5,
                    "cache_creation_tokens": 3,
                },
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        }
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        language_model = LanguageModel(
            model="ollama/mistral",
            hooks=[Recorder(base_dir=base_dir)],
        )
        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            name="first_generator",
        )(x0)
        x2 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            name="second_generator",
        )(x1)
        program = Program(inputs=x0, outputs=x2, name="my_program")

        await program(Query(query="What is the french city of aerospace?"))

        # One folder per originating module sharing the same LM.
        for origin in ("first_generator", "second_generator"):
            directory = os.path.join(base_dir, "my_program", origin)
            (row,) = self._read(directory, "train.jsonl")
            (metadata,) = self._read(directory, "metadata.jsonl")
            self.assertEqual(self._read(directory, "failed.jsonl"), [])
            # The training line holds the conversation only.
            self.assertEqual(list(row), ["messages"])
            self.assertEqual(row["messages"][-1]["role"], "assistant")
            self.assertEqual(
                json.loads(row["messages"][-1]["content"]), json.loads(EXPECTED_STRING)
            )
            with open(os.path.join(directory, "train.jsonl"), "rb") as f:
                line = f.readline().rstrip(b"\n")
            self.assertEqual(metadata["line_hash"], hashlib.sha256(line).hexdigest())
            self.assertEqual(next(iter(metadata)), "synalinks_version")
            self.assertEqual(metadata["synalinks_version"], version.__version__)
            self.assertTrue(metadata["session_id"])
            self.assertEqual(metadata["program"], "my_program")
            self.assertEqual(metadata["module"], origin)
            self.assertTrue(metadata["parent_call_id"])
            self.assertNotEqual(metadata["call_id"], metadata["parent_call_id"])
            self.assertGreater(metadata["timestamp"], 0)
            self.assertGreaterEqual(metadata["duration"], 0)
            self.assertEqual(
                metadata["usage"],
                {
                    "prompt_tokens": 25,
                    "completion_tokens": 10,
                    "total_tokens": 35,
                    "cached_tokens": 5,
                    "cache_creation_tokens": 3,
                    "reasoning_tokens": 7,
                },
            )
            self.assertEqual(metadata["cost"], 0.0)
            self.assertEqual(
                metadata["inputs_hash"],
                hashlib.sha256(
                    orjson.dumps(row["messages"][:-1], option=orjson.OPT_SORT_KEYS)
                ).hexdigest(),
            )
            self.assertEqual(metadata["outputs"], json.loads(EXPECTED_STRING))
            self.assertIn("answer", metadata["schema"]["properties"])
            self.assertEqual(metadata["config"]["class_name"], "LanguageModel")
            self.assertEqual(
                metadata["config_hash"],
                hashlib.sha256(
                    orjson.dumps(metadata["config"], option=orjson.OPT_SORT_KEYS)
                ).hexdigest(),
            )

    @patch("litellm.acompletion")
    async def test_direct_lm_call_records_in_lm_folder(self, mock_completion):
        mock_completion.return_value = _response(
            "Hi there", reasoning_content="The user greets me."
        )
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        language_model = LanguageModel(
            model="ollama/mistral",
            name="my_lm",
            hooks=[Recorder(base_dir=base_dir)],
        )

        await language_model(_hello())

        (row,) = self._read(base_dir, "my_lm", "train.jsonl")
        (metadata,) = self._read(base_dir, "my_lm", "metadata.jsonl")
        # The LM is the entry module: no parent call, no subfolder.
        self.assertIsNone(metadata["parent_call_id"])
        self.assertEqual(metadata["program"], "my_lm")
        self.assertEqual(metadata["module"], "my_lm")
        self.assertIsNone(metadata["schema"])
        self.assertEqual(
            row,
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {
                        "role": "assistant",
                        "content": "Hi there",
                        "reasoning_content": "The user greets me.",
                    },
                ]
            },
        )

    @patch("litellm.acompletion")
    async def test_thinking_blocks_folded_into_reasoning_content(self, mock_completion):
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hi there",
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The user greets me.",
                                "signature": "sig",
                            },
                            {"type": "redacted_thinking", "data": "opaque"},
                        ],
                    }
                }
            ]
        }
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        language_model = LanguageModel(
            model="anthropic/claude-sonnet-5",
            name="my_lm",
            hooks=[Recorder(base_dir=base_dir)],
        )

        await language_model(_hello())

        (row,) = self._read(base_dir, "my_lm", "train.jsonl")
        (metadata,) = self._read(base_dir, "my_lm", "metadata.jsonl")
        # Redacted blocks carry no text; the raw blocks stay in `outputs`.
        self.assertEqual(
            row["messages"][-1],
            {
                "role": "assistant",
                "content": "Hi there",
                "reasoning_content": "The user greets me.",
            },
        )
        self.assertEqual(len(metadata["outputs"]["thinking_blocks"]), 2)

    @patch("litellm.acompletion")
    async def test_record_traces_adds_default_hook(self, mock_completion):
        mock_completion.return_value = _response(
            "Hi there", reasoning_content="Greeting."
        )
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        try:
            config.record_traces(base_dir=base_dir, format="tinker")
            # No explicit hook: the Recorder is added by default.
            language_model = LanguageModel(model="ollama/mistral", name="my_lm")
            await language_model(_hello())
        finally:
            config._ENABLE_TRACE_RECORDING = False
            config._TRACE_RECORDING_DIR = None
            config._TRACE_RECORDING_FORMAT = "openai"

        (row,) = self._read(base_dir, "my_lm", "train.jsonl")
        self.assertEqual(
            row["messages"][-1],
            {"role": "assistant", "content": "<think>\nGreeting.\n</think>\nHi there"},
        )

    def test_default_base_dir_is_project_local(self):
        self.assertEqual(
            Recorder().base_dir, os.path.join(os.path.abspath("."), ".synalinks")
        )
        self.assertEqual(Recorder().format, "openai")

    def test_unknown_format_raises(self):
        with self.assertRaisesRegex(ValueError, "`format` must be one of"):
            Recorder(format="jsonl")

    async def test_ignores_non_language_model_modules(self):
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        module = Identity(name="passthrough", hooks=[Recorder(base_dir=base_dir)])

        await module(Query(query="a"))

        self.assertFalse(os.path.exists(base_dir))

    @patch("litellm.acompletion")
    async def test_agent_records_tools_in_openai_format(self, mock_completion):
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        directory = await self._run_agent(mock_completion, base_dir, "openai")

        first, second = self._read(directory, "train.jsonl")
        # Every LM call of the trajectory is an example, with its tools.
        for row in (first, second):
            self.assertEqual(set(row), {"messages", "tools"})
            self.assertEqual(row["tools"][0]["function"]["name"], "calculate")
        tool_call = first["messages"][-1]["tool_calls"][0]
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]), {"expression": "10 + 20"}
        )
        tool_message = second["messages"][-2]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], tool_call["id"])
        self.assertEqual(json.loads(tool_message["content"]), {"result": 30.0})
        self.assertEqual(
            second["messages"][-1]["reasoning_content"], "The tool returned 30."
        )
        self.assertEqual(len(self._read(directory, "metadata.jsonl")), 2)

    @patch("litellm.acompletion")
    async def test_agent_in_tinker_format_is_unsupported(self, mock_completion):
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        directory = await self._run_agent(mock_completion, base_dir, "tinker")

        self.assertEqual(self._read(directory, "train.jsonl"), [])
        failed = self._read(directory, "failed.jsonl")
        self.assertEqual(len(failed), 2)
        for record in failed:
            self.assertEqual(
                record["unsupported"], "Tinker cannot load tool calls from a file"
            )
            self.assertNotIn("exception", record)
            self.assertEqual(record["tools"][0]["function"]["name"], "calculate")
            self.assertEqual(record["messages"][-1]["role"], "assistant")

    @patch("litellm.acompletion")
    async def test_agent_in_customizer_format(self, mock_completion):
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        directory = await self._run_agent(mock_completion, base_dir, "customizer")

        _, row = self._read(directory, "train.jsonl")
        allowed = {"role", "content", "thinking", "tool_calls"}
        for message in row["messages"]:
            self.assertLessEqual(set(message), allowed)
            self.assertFalse("content" in message and "thinking" in message)
        # Reasoning in its own `thinking` message, object arguments, and no
        # `tool_call_id` on the tool result.
        self.assertEqual(
            row["messages"][-5:],
            [
                {"role": "assistant", "thinking": "I should use the calculator."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": {"expression": "10 + 20"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"result": 30.0}'},
                {"role": "assistant", "thinking": "The tool returned 30."},
                {"role": "assistant", "content": "The result is 30."},
            ],
        )
        self.assertEqual(row["tools"][0]["function"]["name"], "calculate")

    @patch("litellm.acompletion")
    async def test_fallback_call_does_not_report_stale_usage(self, mock_completion):
        mock_completion.side_effect = [
            _response("first", usage=(42, 17)),
            Exception("boom"),
            Exception("boom"),
            _response("from fallback", usage=(5, 3)),
        ]
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        language_model = LanguageModel(
            model="ollama/mistral",
            name="primary",
            retry=1,
            retry_max_wait=0,
            hooks=[Recorder(base_dir=base_dir)],
            fallback=LanguageModel(model="ollama/qwen3", name="backup"),
        )
        await self._call_twice(language_model)

        first, second = self._read(base_dir, "primary", "metadata.jsonl")
        self.assertEqual(first["usage"]["total_tokens"], 59)
        self.assertNotIn("fallback", first)
        # Served by the fallback: the primary reports no usage of its own.
        self.assertIsNone(second["usage"])
        self.assertIsNone(second["cost"])
        self.assertTrue(second["fallback"])
        self.assertEqual(second["outputs"]["content"], "from fallback")
        self.assertEqual(self._read(base_dir, "primary", "failed.jsonl"), [])

    @patch("litellm.acompletion")
    async def test_cache_hit_reports_zero_usage(self, mock_completion):
        mock_completion.return_value = _response("cached", usage=(42, 17))
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        language_model = LanguageModel(
            model="ollama/mistral",
            name="cached_lm",
            cache_dir=os.path.join(self.get_temp_dir(), "cache"),
            hooks=[Recorder(base_dir=base_dir)],
        )
        await self._call_twice(language_model)

        self.assertEqual(mock_completion.call_count, 1)
        first, second = self._read(base_dir, "cached_lm", "metadata.jsonl")
        self.assertEqual(first["usage"]["total_tokens"], 59)
        self.assertNotIn("cache_hit", first)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(set(second["usage"].values()), {0})
        self.assertEqual(second["cost"], 0.0)

    @patch("litellm.acompletion")
    async def test_failed_call_without_fallback_records_exception(self, mock_completion):
        mock_completion.side_effect = [
            _response("first", usage=(42, 17)),
            Exception("provider down"),
            Exception("provider down"),
        ]
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        language_model = LanguageModel(
            model="ollama/mistral",
            name="lonely_lm",
            retry=1,
            retry_max_wait=0,
            hooks=[Recorder(base_dir=base_dir)],
        )
        await self._call_twice(language_model)

        self.assertEqual(len(self._read(base_dir, "lonely_lm", "train.jsonl")), 1)
        (failed,) = self._read(base_dir, "lonely_lm", "failed.jsonl")
        self.assertIsNone(failed["usage"])
        self.assertIsNone(failed["cost"])
        self.assertIn("provider down", failed["exception"])
        self.assertIsNone(failed["outputs"])
        self.assertEqual(failed["messages"], [{"role": "user", "content": "Hello"}])
        self.assertNotIn("line_hash", failed)


class FormatConversationTest(testing.TestCase):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "reasoning_content": "Greet."},
    ]

    def test_openai_keeps_the_conversation(self):
        tools = [{"type": "function", "function": {"name": "f"}}]
        row, reason = format_conversation(self.messages, tools=tools)
        self.assertIsNone(reason)
        self.assertEqual(row, {"messages": self.messages, "tools": tools})

    def test_tinker_prefixes_reasoning(self):
        row, _ = format_conversation(self.messages, format="tinker")
        self.assertEqual(
            row["messages"][-1],
            {"role": "assistant", "content": "<think>\nGreet.\n</think>\nHello"},
        )

    def test_customizer_splits_reasoning(self):
        row, _ = format_conversation(self.messages, format="customizer")
        self.assertEqual(
            row["messages"][1:],
            [
                {"role": "assistant", "thinking": "Greet."},
                {"role": "assistant", "content": "Hello"},
            ],
        )

    def test_text_parts_are_joined_and_media_is_unsupported(self):
        text = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        row, _ = format_conversation(text, format="tinker")
        self.assertEqual(row["messages"], [{"role": "user", "content": "Hi"}])
        image = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        ]
        for format in ("tinker", "customizer"):
            row, reason = format_conversation(image, format=format)
            self.assertIsNone(row)
            self.assertIn("only takes text content", reason)
        row, _ = format_conversation(image, format="openai")
        self.assertEqual(row["messages"], image)

    def test_customizer_rejects_non_object_arguments(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "[1"},
                    }
                ],
            }
        ]
        row, reason = format_conversation(messages, format="customizer")
        self.assertIsNone(row)
        self.assertIn("JSON object", reason)

    def test_unknown_format_raises(self):
        with self.assertRaisesRegex(ValueError, "`format` must be one of"):
            format_conversation(self.messages, format="jsonl")
