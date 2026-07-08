# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import glob
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
from synalinks.src.hooks.recorder import Recorder
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.identity import Identity
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs.program import Program


class Query(DataModel):
    query: str


class AnswerWithRationale(DataModel):
    rationale: str
    answer: str


EXPECTED_STRING = """{"rationale": "Toulouse hosts ISAE-SUPAERO", "answer": "Toulouse"}"""


class RecorderTest(testing.TestCase):
    def _read_records(self, filepath):
        with open(filepath, "rb") as f:
            return [orjson.loads(line) for line in f if line.strip()]

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
        program = Program(
            inputs=x0,
            outputs=x2,
            name="my_program",
        )

        await program(Query(query="What is the french city of aerospace?"))

        # One subfolder per originating module sharing the same LM.
        for origin in ("first_generator", "second_generator"):
            files = glob.glob(
                os.path.join(base_dir, "my_program", origin, f"{origin}_*.jsonl")
            )
            self.assertEqual(len(files), 1)
            records = self._read_records(files[0])
            self.assertEqual(len(records), 1)
            record = records[0]
            # The Synalinks version is the first key of every record.
            self.assertEqual(next(iter(record)), "synalinks_version")
            self.assertEqual(record["synalinks_version"], version.__version__)
            self.assertEqual(record["program"], "my_program")
            self.assertEqual(record["module"], origin)
            self.assertTrue(record["call_id"])
            self.assertTrue(record["parent_call_id"])
            self.assertNotEqual(record["call_id"], record["parent_call_id"])
            self.assertGreater(record["timestamp"], 0)
            self.assertGreaterEqual(record["duration"], 0)
            self.assertEqual(
                record["usage"],
                {
                    "prompt_tokens": 25,
                    "completion_tokens": 10,
                    "total_tokens": 35,
                    "cached_tokens": 5,
                    "cache_creation_tokens": 3,
                    "reasoning_tokens": 7,
                },
            )
            self.assertEqual(record["cost"], 0.0)
            # `messages` is the full conversation in OpenAI chat format
            # (NeMo-compatible): inputs plus the completion as the final
            # assistant message.
            self.assertGreaterEqual(len(record["messages"]), 2)
            for message in record["messages"]:
                self.assertIn("role", message)
                self.assertIn("content", message)
            last_message = record["messages"][-1]
            self.assertEqual(last_message["role"], "assistant")
            # The structured completion is serialized as the assistant content.
            self.assertEqual(
                json.loads(last_message["content"]), json.loads(EXPECTED_STRING)
            )
            # The hash is reproducible from the recorded input messages.
            self.assertEqual(
                record["inputs_hash"],
                hashlib.sha256(
                    orjson.dumps(record["messages"][:-1], option=orjson.OPT_SORT_KEYS)
                ).hexdigest(),
            )
            # Outputs are the (structured) completion returned by the LM.
            self.assertEqual(record["outputs"], json.loads(EXPECTED_STRING))
            self.assertEqual(record["config"]["class_name"], "LanguageModel")
            self.assertEqual(
                record["config_hash"],
                hashlib.sha256(
                    orjson.dumps(record["config"], option=orjson.OPT_SORT_KEYS)
                ).hexdigest(),
            )

    @patch("litellm.acompletion")
    async def test_direct_lm_call_records_in_program_folder(self, mock_completion):
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Hi there",
                        "reasoning_content": "The user greets me.",
                    }
                }
            ]
        }
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        language_model = LanguageModel(
            model="ollama/mistral",
            name="my_lm",
            hooks=[Recorder(base_dir=base_dir)],
        )

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        await language_model(messages)

        files = glob.glob(os.path.join(base_dir, "my_lm", "my_lm_*.jsonl"))
        self.assertEqual(len(files), 1)
        records = self._read_records(files[0])
        self.assertEqual(len(records), 1)
        record = records[0]
        # The LM is the entry module: no parent call, no subfolder.
        self.assertIsNone(record["parent_call_id"])
        self.assertEqual(record["program"], "my_lm")
        self.assertEqual(record["module"], "my_lm")
        # A plain LM call yields a ready-to-train OpenAI chat conversation;
        # `reasoning_content` is kept in the messages (NeMo trains on it).
        self.assertEqual(
            record["messages"],
            [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi there",
                    "reasoning_content": "The user greets me.",
                },
            ],
        )
        self.assertEqual(record["outputs"]["content"], "Hi there")

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

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        await language_model(messages)

        files = glob.glob(os.path.join(base_dir, "my_lm", "my_lm_*.jsonl"))
        record = self._read_records(files[0])[0]
        # No `reasoning_content` from the provider: the thinking blocks'
        # text is recovered instead of being dropped (redacted blocks
        # carry no text); the raw blocks stay in `outputs`.
        self.assertEqual(
            record["messages"][-1],
            {
                "role": "assistant",
                "content": "Hi there",
                "reasoning_content": "The user greets me.",
            },
        )
        self.assertEqual(len(record["outputs"]["thinking_blocks"]), 2)

    @patch("litellm.acompletion")
    async def test_record_traces_adds_default_hook(self, mock_completion):
        from synalinks.src.backend import config

        mock_completion.return_value = {"choices": [{"message": {"content": "Hi there"}}]}
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        try:
            config.record_traces(base_dir=base_dir)
            # No explicit hook: the Recorder is added by default.
            language_model = LanguageModel(model="ollama/mistral", name="my_lm")

            messages = ChatMessages(
                messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
            )
            await language_model(messages)
        finally:
            config._ENABLE_TRACE_RECORDING = False
            config._TRACE_RECORDING_DIR = None

        files = glob.glob(os.path.join(base_dir, "my_lm", "my_lm_*.jsonl"))
        self.assertEqual(len(files), 1)
        records = self._read_records(files[0])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["outputs"]["content"], "Hi there")

    async def test_ignores_non_language_model_modules(self):
        base_dir = os.path.join(self.get_temp_dir(), ".synalinks")
        module = Identity(name="passthrough", hooks=[Recorder(base_dir=base_dir)])

        await module(Query(query="a"))

        self.assertFalse(os.path.exists(base_dir))
