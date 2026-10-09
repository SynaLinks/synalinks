# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import json
import os
import shutil
import warnings
from typing import Dict
from typing import List
from typing import Optional
from unittest.mock import patch

import synalinks
from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.modules.language_models import OAuthLanguageModel
from synalinks.src.modules.language_models import oauth_language_model
from synalinks.src.modules.language_models.oauth_language_model import (
    _build_codex_command,
)
from synalinks.src.modules.language_models.oauth_language_model import _parse_codex_events
from synalinks.src.modules.language_models.oauth_language_model import _split_messages
from synalinks.src.modules.language_models.oauth_language_model import _strict_schema

_RUN = (
    "synalinks.src.modules.language_models.oauth_language_model.OAuthLanguageModel._run"
)


def _events(text, usage=None, extra=()):
    """The JSONL stream `codex exec --json` prints for a one-message turn."""
    events = [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "turn.started"},
        *extra,
        {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": text},
        },
        {
            "type": "turn.completed",
            "usage": usage
            or {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 20,
                "reasoning_output_tokens": 5,
            },
        },
    ]
    return "\n".join(json.dumps(e) for e in events)


class Address(DataModel):
    city: str = Field(description="The city")
    zip_code: Optional[str] = None


class Person(DataModel):
    name: str = Field(description="The name")
    age: int = 0
    tags: List[str] = []
    address: Address = Address(city="")


class FreeForm(DataModel):
    payload: Dict[str, str] = {}


class Reasoned(DataModel):
    thinking: str = Field(description="Step by step thinking")
    answer: int = Field(description="The answer")


class Recorder:
    """Stands in for `OAuthLanguageModel._run`, recording what was sent."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.result = (stdout, stderr, returncode)
        self.calls = []

    async def __call__(self, command, prompt, cwd=None):
        files = {}
        for flag, key in (("--output-schema", "schema"),):
            if flag in command:
                with open(command[command.index(flag) + 1], encoding="utf-8") as f:
                    files[key] = json.load(f)
        for arg in command:
            if arg.startswith("model_instructions_file="):
                path = json.loads(arg.split("=", 1)[1])
                with open(path, encoding="utf-8") as f:
                    files["instructions"] = f.read()
        self.calls.append(
            {
                "command": command,
                "prompt": prompt,
                "cwd": cwd,
                "cwd_content": os.listdir(cwd),
                **files,
            }
        )
        return self.result


def _messages():
    return ChatMessages(
        messages=[
            ChatMessage(role=ChatRole.SYSTEM, content="You extract persons."),
            ChatMessage(role=ChatRole.USER, content="Ada, 36, from London."),
        ]
    )


def _user(content):
    return ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content=content)])


class StrictSchemaTest(testing.TestCase):
    def test_every_object_is_closed_and_fully_required(self):
        schema = Person.get_schema()
        strict = _strict_schema(schema)
        self.assertEqual(strict["required"], ["name", "age", "tags", "address"])
        self.assertIs(strict["additionalProperties"], False)
        address = strict["$defs"]["Address"]
        self.assertEqual(address["required"], ["city", "zip_code"])
        self.assertIs(address["additionalProperties"], False)

    def test_input_is_not_mutated(self):
        schema = Person.get_schema()
        before = json.dumps(schema, sort_keys=True)
        _strict_schema(schema)
        self.assertEqual(json.dumps(schema, sort_keys=True), before)

    def test_property_named_like_a_keyword_is_not_mistaken_for_a_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "properties": {"type": "string"},
                "type": {"type": "string"},
            },
        }
        strict = _strict_schema(schema)
        self.assertEqual(strict["required"], ["properties", "type"])
        self.assertNotIn("required", strict["properties"])

    def test_free_form_object_raises(self):
        with self.assertRaisesRegex(ValueError, "free-form object"):
            _strict_schema(FreeForm.get_schema())
        with self.assertRaisesRegex(ValueError, "free-form object"):
            _strict_schema({"type": "object"})


class CodexCommandTest(testing.TestCase):
    def test_command_is_locked_down(self):
        command = _build_codex_command(
            "gpt-5.5",
            "/tmp/work",
            instructions_path="/tmp/i.md",
            schema_path="/tmp/s.json",
            reasoning_effort="low",
        )
        self.assertEqual(command[:2], ["codex", "exec"])
        for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--json"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("-s") + 1], "read-only")
        self.assertEqual(command[command.index("-C") + 1], "/tmp/work")
        self.assertEqual(command[command.index("-m") + 1], "gpt-5.5")
        self.assertEqual(command[command.index("--output-schema") + 1], "/tmp/s.json")
        self.assertIn('web_search="disabled"', command)
        self.assertIn("features.shell_tool=false", command)
        self.assertIn('model_instructions_file="/tmp/i.md"', command)
        self.assertIn('model_reasoning_effort="low"', command)
        self.assertEqual(command[-1], "-")
        self.assertFalse(any("dangerously" in arg for arg in command))

    def test_optional_parts_are_omitted(self):
        command = _build_codex_command(None, "/tmp/work")
        self.assertNotIn("-m", command)
        self.assertNotIn("--output-schema", command)
        self.assertFalse(any(a.startswith("model_instructions_file") for a in command))
        self.assertFalse(any(a.startswith("model_reasoning_effort") for a in command))


class CodexEventsTest(testing.TestCase):
    def test_last_agent_message_and_usage(self):
        stdout = _events(
            "final",
            extra=[
                {"type": "item.completed", "item": {"type": "reasoning", "text": "…"}},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "preamble"},
                },
            ],
        )
        content, usage, errors, forbidden = _parse_codex_events("noise\n" + stdout)
        self.assertEqual(content, "final")
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(errors, [])
        self.assertEqual(forbidden, [])

    def test_tool_items_are_reported(self):
        stdout = _events(
            "done",
            extra=[
                {"type": "item.started", "item": {"type": "web_search"}},
                {"type": "item.completed", "item": {"type": "command_execution"}},
            ],
        )
        _, _, _, forbidden = _parse_codex_events(stdout)
        self.assertEqual(forbidden, ["command_execution", "web_search"])

    def test_notice_items_are_not_failures(self):
        stdout = _events(
            "ok",
            extra=[{"type": "item.completed", "item": {"type": "error", "message": "w"}}],
        )
        content, _, errors, forbidden = _parse_codex_events(stdout)
        self.assertEqual((content, errors, forbidden), ("ok", [], []))

    def test_turn_failure_is_collected(self):
        stdout = "\n".join(
            [
                json.dumps({"type": "error", "message": "boom"}),
                json.dumps({"type": "turn.failed", "error": {"message": "boom"}}),
            ]
        )
        content, _, errors, _ = _parse_codex_events(stdout)
        self.assertIsNone(content)
        self.assertEqual(errors, ["boom", "boom"])


class SplitMessagesTest(testing.TestCase):
    def test_system_becomes_instructions_and_lone_user_is_verbatim(self):
        instructions, prompt = _split_messages(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ]
        )
        self.assertEqual(instructions, "Be terse.")
        self.assertEqual(prompt, "Hello")

    def test_conversation_is_rendered_as_transcript(self):
        _, prompt = _split_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ]
        )
        self.assertEqual(prompt, "[USER]\nHi\n\n[ASSISTANT]\nHello\n\n[USER]\nBye")


class OAuthLanguageModelTest(testing.TestCase):
    def test_model_must_be_a_codex_model(self):
        for model in ("gpt-5.5", "openai/gpt-5.5", "codex/", "claude/sonnet"):
            with self.assertRaisesRegex(ValueError, "expects `model`"):
                OAuthLanguageModel(model=model)
        with self.assertRaisesRegex(ValueError, "need to set the `model`"):
            OAuthLanguageModel()

    def test_get_routes_codex_identifiers(self):
        lm = synalinks.language_models.get("codex/gpt-5.5")
        self.assertIsInstance(lm, OAuthLanguageModel)
        self.assertNotIsInstance(
            synalinks.language_models.get("openai/gpt-4o-mini"), OAuthLanguageModel
        )
        self.assertIs(synalinks.OAuthLanguageModel, OAuthLanguageModel)
        self.assertEqual(OAuthLanguageModel.supported_providers(), ["codex"])
        self.assertNotIn("codex", LanguageModel.supported_providers())

    def test_serialization_roundtrip(self):
        lm = OAuthLanguageModel(
            model="codex/gpt-5.5",
            timeout=30,
            retry=2,
            reasoning_effort="low",
            fallback=OAuthLanguageModel(model="codex/default"),
        )
        config = synalinks.language_models.serialize(lm)
        restored = synalinks.language_models.deserialize(config)
        self.assertIsInstance(restored, OAuthLanguageModel)
        self.assertEqual(restored.get_config(), lm.get_config())
        self.assertIsInstance(restored.fallback, OAuthLanguageModel)
        self.assertEqual(restored.default_kwargs, {"reasoning_effort": "low"})

    async def test_structured_output(self):
        answer = {
            "name": "Ada",
            "age": 36,
            "tags": [],
            "address": {"city": "London", "zip_code": None},
        }
        recorder = Recorder(stdout=_events(json.dumps(answer)))
        lm = OAuthLanguageModel(model="codex/gpt-5.5", reasoning_effort="medium")
        with patch(_RUN, recorder):
            result = await lm(_messages(), schema=Person.get_schema())

        self.assertEqual(result.get_json(), answer)
        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        # The system message is the model instructions, the user message the
        # prompt, and the model works in an empty directory.
        self.assertEqual(call["instructions"], "You extract persons.")
        self.assertEqual(call["prompt"], "Ada, 36, from London.")
        self.assertEqual(call["cwd_content"], [])
        self.assertEqual(call["command"][call["command"].index("-m") + 1], "gpt-5.5")
        self.assertIn('model_reasoning_effort="medium"', call["command"])
        # The schema handed to the CLI is the strict one.
        self.assertEqual(call["schema"]["required"], ["name", "age", "tags", "address"])
        self.assertIs(call["schema"]["additionalProperties"], False)
        # Usage flows into the inherited counters; the cost is not tracked.
        self.assertEqual(lm.last_call_prompt_tokens, 100)
        self.assertEqual(lm.last_call_completion_tokens, 20)
        self.assertEqual(lm.last_call_cached_tokens, 40)
        self.assertEqual(lm.last_call_reasoning_tokens, 5)
        self.assertEqual(lm.cumulated_calls, 1)
        self.assertEqual(lm.cumulated_cost, 0.0)

    async def test_chat_output_and_default_model(self):
        recorder = Recorder(stdout=_events("Hello!"))
        lm = OAuthLanguageModel(model="codex/default")
        messages = ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="Hi")])
        with patch(_RUN, recorder):
            result = await lm(messages)
        self.assertEqual(result.get("role"), ChatRole.ASSISTANT)
        self.assertEqual(result.get("content"), "Hello!")
        command = recorder.calls[0]["command"]
        self.assertNotIn("-m", command)
        self.assertNotIn("--output-schema", command)
        self.assertFalse(any(a.startswith("model_reasoning_effort") for a in command))

    async def test_per_call_reasoning_effort_overrides_default(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5", reasoning_effort="high")
        with patch(_RUN, recorder):
            await lm(_messages(), reasoning_effort="disable")
            await lm(_messages(), reasoning_effort="none")
        # "disable" turns reasoning off, "none" keeps the model default.
        self.assertIn('model_reasoning_effort="none"', recorder.calls[0]["command"])
        self.assertFalse(
            any(
                a.startswith("model_reasoning_effort")
                for a in recorder.calls[1]["command"]
            )
        )

    async def test_thinking_field_is_written_by_the_model(self):
        answer = {"thinking": "s-t-r-a-w-b-e-r-r-y: three r.", "answer": 3}
        stdout = _events(
            json.dumps(answer),
            extra=[
                {
                    "type": "item.completed",
                    "item": {"type": "reasoning", "text": "**Counting**"},
                }
            ],
        )
        recorder = Recorder(stdout=stdout)
        lm = OAuthLanguageModel(model="codex/gpt-5.5", reasoning_effort="low")
        with patch(_RUN, recorder):
            result = await lm(_messages(), schema=Reasoned.get_schema())
        # The effort is sent, the field stays in the schema and the model's
        # text is kept (the one-line summary does not overwrite it).
        self.assertIn('model_reasoning_effort="low"', recorder.calls[0]["command"])
        self.assertEqual(recorder.calls[0]["schema"]["required"], ["thinking", "answer"])
        self.assertEqual(result.get_json(), answer)

    async def test_minimal_effort_is_sent_as_low(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5")
        with patch(_RUN, recorder):
            await lm(_messages(), reasoning_effort="minimal")
        self.assertIn('model_reasoning_effort="low"', recorder.calls[0]["command"])

    async def test_unset_effort_keeps_the_model_default(self):
        # A `Generator` built without an effort passes `reasoning_effort=None`.
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5", reasoning_effort="high")
        with patch(_RUN, recorder):
            await lm(_messages(), reasoning_effort=None)
        self.assertIn('model_reasoning_effort="high"', recorder.calls[0]["command"])

    async def test_unsupported_parameters_are_ignored_with_a_warning(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5", temperature=0.0, max_tokens=10)
        with patch(_RUN, recorder):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await lm(_messages())
        self.assertEqual(result.get("content"), "ok")
        self.assertTrue(
            any("['max_tokens', 'temperature']" in str(w.message) for w in caught)
        )
        self.assertFalse(any("temperature" in a for a in recorder.calls[0]["command"]))

    async def test_tools_and_streaming_are_rejected(self):
        lm = OAuthLanguageModel(model="codex/gpt-5.5")
        with self.assertRaisesRegex(ValueError, "tool calling"):
            await lm(_messages(), tool_schemas=[{"type": "function"}])
        with self.assertRaisesRegex(ValueError, "streaming"):
            await lm(_messages(), streaming=True)

    async def test_free_form_schema_is_rejected_before_any_call(self):
        recorder = Recorder(stdout=_events("{}"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5")
        with patch(_RUN, recorder):
            with self.assertRaisesRegex(ValueError, "free-form object"):
                await lm(_messages(), schema=FreeForm.get_schema())
        self.assertEqual(recorder.calls, [])

    async def test_tool_use_fails_the_call_without_retry(self):
        stdout = _events(
            "I searched the web.",
            extra=[{"type": "item.completed", "item": {"type": "web_search"}}],
        )
        recorder = Recorder(stdout=stdout)
        lm = OAuthLanguageModel(model="codex/gpt-5.5", retry=3, retry_max_wait=0)
        with patch(_RUN, recorder):
            result = await lm(_messages())
        self.assertIsNone(result)
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(lm.cumulated_failed_calls, 1)

    async def test_rejected_request_is_not_retried(self):
        error = '{"error":{"type":"invalid_request_error","message":"bad model"}}'
        stdout = json.dumps({"type": "turn.failed", "error": {"message": error}})
        recorder = Recorder(stdout=stdout, returncode=1)
        lm = OAuthLanguageModel(model="codex/nope", retry=3, retry_max_wait=0)
        with patch(_RUN, recorder):
            result = await lm(_messages())
        self.assertIsNone(result)
        self.assertEqual(len(recorder.calls), 1)

    async def test_transient_failure_is_retried_then_falls_back(self):
        recorder = Recorder(stderr="connection reset", returncode=1)
        fallback = OAuthLanguageModel(model="codex/default")
        lm = OAuthLanguageModel(
            model="codex/gpt-5.5", retry=2, retry_max_wait=0, fallback=fallback
        )
        calls = []

        async def run(self, command, prompt, cwd=None):
            calls.append(self.model)
            if self is fallback:
                return _events("from fallback"), "", 0
            return await recorder(command, prompt, cwd=cwd)

        with patch(_RUN, run):
            result = await lm(_messages())
        self.assertEqual(result.get("content"), "from fallback")
        self.assertEqual(calls, ["codex/gpt-5.5", "codex/gpt-5.5", "codex/default"])
        self.assertEqual(lm.cumulated_fallback_activations, 1)

    async def test_fallback_receives_the_caller_parameters_untouched(self):
        recorder = Recorder(stderr="connection reset", returncode=1)
        lm = OAuthLanguageModel(
            model="codex/gpt-5.5",
            retry=1,
            reasoning_effort="high",
            fallback=LanguageModel(model="openai/gpt-4o-mini"),
        )
        with patch(_RUN, recorder):
            with patch("litellm.acompletion") as mock_completion:
                mock_completion.return_value = {
                    "choices": [{"message": {"content": "from the API"}}]
                }
                result = await lm(_messages(), reasoning_effort="low", temperature=0.2)
        self.assertEqual(result.get("content"), "from the API")
        self.assertIn('model_reasoning_effort="low"', recorder.calls[0]["command"])
        sent = mock_completion.call_args.kwargs
        self.assertEqual(sent["model"], "openai/gpt-4o-mini")
        self.assertEqual(sent["temperature"], 0.2)
        self.assertFalse([key for key in sent if key.startswith("oauth")])

    async def test_concurrent_calls_keep_their_own_reasoning_effort(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5")
        efforts = ["low", "medium", "high", "none"]
        with patch(_RUN, recorder):
            await asyncio.gather(
                *(lm(_user(effort), reasoning_effort=effort) for effort in efforts)
            )
        seen = {}
        for call in recorder.calls:
            flags = [a for a in call["command"] if a.startswith("model_reasoning_effort")]
            seen[call["prompt"]] = flags
        self.assertEqual(
            seen,
            {
                "low": ['model_reasoning_effort="low"'],
                "medium": ['model_reasoning_effort="medium"'],
                "high": ['model_reasoning_effort="high"'],
                "none": [],
            },
        )

    async def test_file_cache_hit_skips_the_cli(self):
        recorder = Recorder(stdout=_events("cached answer"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5", cache_dir=self.get_temp_dir())
        with patch(_RUN, recorder):
            first = await lm(_messages())
            second = await lm(_messages())
        self.assertEqual(first.get_json(), second.get_json())
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(lm.cumulated_cache_hits, 1)

    async def test_file_cache_is_keyed_by_reasoning_effort(self):
        recorder = Recorder(stdout=_events("answer"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5", cache_dir=self.get_temp_dir())
        with patch(_RUN, recorder):
            await lm(_messages(), reasoning_effort="low")
            await lm(_messages(), reasoning_effort="high")
            await lm(_messages(), reasoning_effort="low")
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(lm.cumulated_cache_hits, 1)

    async def test_ignored_parameters_are_reported_once(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-5.5")
        with patch(_RUN, recorder):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                await lm(_messages(), temperature=0.0)
                await lm(_messages(), temperature=0.7)
                await lm(_messages(), temperature=0.0, max_tokens=5)
        reports = [str(w.message) for w in caught if "ignores" in str(w.message)]
        self.assertEqual(len(reports), 2)
        self.assertIn("['temperature']", reports[0])
        self.assertIn("['max_tokens']", reports[1])

    async def test_missing_cli_fails_fast(self):
        lm = OAuthLanguageModel(model="codex/gpt-5.5", retry=3, retry_max_wait=0)
        with patch.object(
            oauth_language_model.asyncio,
            "create_subprocess_exec",
            side_effect=FileNotFoundError("codex"),
        ) as spawn:
            result = await lm(_messages())
        self.assertIsNone(result)
        self.assertEqual(spawn.call_count, 1)

    async def test_subprocess_environment_is_an_allowlist(self):
        lm = OAuthLanguageModel(model="codex/gpt-5.5", retry=1)
        seen = {}

        async def spawn(*command, env=None, **kwargs):
            seen.update(env)
            raise FileNotFoundError("codex")

        environ = {
            "HOME": "/home/me",
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "sk-secret",
            "SOME_SERVICE_TOKEN": "secret",
        }
        with patch.dict(os.environ, environ, clear=True):
            with patch.object(
                oauth_language_model.asyncio, "create_subprocess_exec", spawn
            ):
                await lm(_messages())
        self.assertEqual(seen, {"HOME": "/home/me", "PATH": "/usr/bin"})

    async def test_timeout_kills_the_subprocess(self):
        if not shutil.which("sleep"):
            self.skipTest("`sleep` is not available")
        lm = OAuthLanguageModel(model="codex/gpt-5.5", timeout=0.2)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            await lm._run(["sleep", "30"], "")
