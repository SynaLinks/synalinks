# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import json
import os
import shutil
import sys
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
    _build_app_server_command,
)
from synalinks.src.modules.language_models.oauth_language_model import (
    _build_codex_command,
)
from synalinks.src.modules.language_models.oauth_language_model import _default_effort
from synalinks.src.modules.language_models.oauth_language_model import _dynamic_tools
from synalinks.src.modules.language_models.oauth_language_model import _history_items
from synalinks.src.modules.language_models.oauth_language_model import _parse_codex_events
from synalinks.src.modules.language_models.oauth_language_model import _split_messages
from synalinks.src.modules.language_models.oauth_language_model import _strict_schema
from synalinks.src.modules.language_models.oauth_language_model import _without_tools

_RUN = (
    "synalinks.src.modules.language_models.oauth_language_model.OAuthLanguageModel._run"
)


# What `codex debug models` prints: a gpt-6 entry, whose tools go through code
# mode, and an older one.
_CATALOG = {
    "models": [
        {
            "slug": "gpt-6.1-sol",
            "default_reasoning_level": "low",
            "tool_mode": "code_mode_only",
            "multi_agent_version": "v2",
            "apply_patch_tool_type": "freeform",
            "experimental_supported_tools": ["clock"],
        },
        {"slug": "gpt-5.5", "tool_mode": None, "experimental_supported_tools": []},
    ]
}


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


class Contact(DataModel):
    name: str = Field(description="The name")
    home: Address = Field(description="Where they live")


class FreeForm(DataModel):
    payload: Dict[str, str] = {}


class Reasoned(DataModel):
    thinking: str = Field(description="Step by step thinking")
    answer: int = Field(description="The answer")


class Recorder:
    """Stands in for `OAuthLanguageModel._run`, recording what was sent.

    `codex debug models` is answered with `catalog` and counted apart: `calls`
    only holds the completions.
    """

    def __init__(self, stdout="", stderr="", returncode=0, catalog=None):
        self.result = (stdout, stderr, returncode)
        self.catalog = json.dumps(_CATALOG) if catalog is None else catalog
        self.catalog_reads = 0
        self.calls = []

    async def __call__(self, command, prompt, cwd=None):
        if command[1] == "debug":
            self.catalog_reads += 1
            return self.catalog, "", 0
        files = {}
        for flag, key in (("--output-schema", "schema"),):
            if flag in command:
                with open(command[command.index(flag) + 1], encoding="utf-8") as f:
                    files[key] = json.load(f)
        for arg in command:
            for prefix, key in (
                ("model_instructions_file=", "instructions"),
                ("model_catalog_json=", "catalog"),
            ):
                if arg.startswith(prefix):
                    with open(json.loads(arg[len(prefix) :]), encoding="utf-8") as f:
                        files[key] = f.read()
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

    def test_ref_keeps_no_sibling_keyword(self):
        # `address` has a default: Pydantic writes it next to the `$ref`.
        strict = _strict_schema(Person.get_schema())
        self.assertEqual(strict["properties"]["address"], {"$ref": "#/$defs/Address"})

    def test_described_ref_is_inlined(self):
        strict = _strict_schema(Contact.get_schema())
        home = strict["properties"]["home"]
        self.assertNotIn("$ref", home)
        self.assertEqual(home["description"], "Where they live")
        self.assertEqual(home["required"], ["city", "zip_code"])
        self.assertIs(home["additionalProperties"], False)

    def test_recursive_ref_stays_a_bare_ref(self):
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/Node", "description": "Next"}
                    },
                }
            },
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node", "description": "Top"}},
        }
        strict = _strict_schema(schema)
        child = strict["properties"]["root"]["properties"]["child"]
        self.assertEqual(child, {"$ref": "#/$defs/Node"})

    def test_free_form_object_raises(self):
        with self.assertRaisesRegex(ValueError, "free-form object"):
            _strict_schema(FreeForm.get_schema())
        with self.assertRaisesRegex(ValueError, "free-form object"):
            _strict_schema({"type": "object"})


class CodexCommandTest(testing.TestCase):
    def test_command_is_locked_down(self):
        command = _build_codex_command(
            "gpt-6.1-sol",
            "/tmp/work",
            instructions_path="/tmp/i.md",
            schema_path="/tmp/s.json",
            catalog_path="/tmp/m.json",
            reasoning_effort="low",
        )
        self.assertEqual(command[:2], ["codex", "exec"])
        for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--json"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("-s") + 1], "read-only")
        self.assertEqual(command[command.index("-C") + 1], "/tmp/work")
        self.assertEqual(command[command.index("-m") + 1], "gpt-6.1-sol")
        self.assertEqual(command[command.index("--output-schema") + 1], "/tmp/s.json")
        self.assertIn('web_search="disabled"', command)
        self.assertIn("features.shell_tool=false", command)
        # Nothing but the messages reaches the model: no injected text, and
        # the catalog without the tools it gives a model.
        self.assertIn("include_permissions_instructions=false", command)
        self.assertIn("include_environment_context=false", command)
        self.assertIn("skills.include_instructions=false", command)
        # No analytics upload holding the process after the turn.
        self.assertIn("analytics.enabled=false", command)
        self.assertIn('model_catalog_json="/tmp/m.json"', command)
        self.assertIn('model_instructions_file="/tmp/i.md"', command)
        self.assertIn('model_reasoning_effort="low"', command)
        self.assertEqual(command[-1], "-")
        self.assertFalse(any("dangerously" in arg for arg in command))

    def test_optional_parts_are_omitted(self):
        command = _build_codex_command(None, "/tmp/work")
        self.assertNotIn("-m", command)
        self.assertNotIn("--output-schema", command)
        self.assertFalse(any(a.startswith("model_instructions_file") for a in command))
        self.assertFalse(any(a.startswith("model_catalog_json") for a in command))
        self.assertFalse(any(a.startswith("model_reasoning_effort") for a in command))


class ModelCatalogTest(testing.TestCase):
    def test_no_model_keeps_its_tools(self):
        sol, older = _without_tools(_CATALOG)["models"]
        self.assertEqual(
            sol,
            {
                "slug": "gpt-6.1-sol",
                "default_reasoning_level": "low",
                "tool_mode": None,
                "multi_agent_version": None,
                "apply_patch_tool_type": None,
                "experimental_supported_tools": [],
            },
        )
        # A field an entry does not have is not added to it.
        self.assertEqual(
            older,
            {"slug": "gpt-5.5", "tool_mode": None, "experimental_supported_tools": []},
        )
        self.assertEqual(_CATALOG["models"][0]["tool_mode"], "code_mode_only")


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


class CodexExecutableTest(testing.TestCase):
    def test_extra_binary_wins_over_the_path(self):
        class Bundled:
            @staticmethod
            def bundled_codex_path():
                return "/site-packages/codex_cli_bin/bin/codex"

        with patch.dict(sys.modules, {"codex_cli_bin": Bundled}):
            self.assertEqual(
                oauth_language_model._codex_executable(),
                "/site-packages/codex_cli_bin/bin/codex",
            )

    def test_path_binary_without_the_extra(self):
        with patch.dict(sys.modules, {"codex_cli_bin": None}):
            with patch.object(shutil, "which", return_value="/usr/bin/codex"):
                self.assertEqual(
                    oauth_language_model._codex_executable(), "/usr/bin/codex"
                )

    async def test_missing_cli_raises_on_the_first_call(self):
        with patch.dict(sys.modules, {"codex_cli_bin": None}):
            with patch.object(shutil, "which", return_value=None):
                lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
                with self.assertRaisesRegex(ImportError, r"synalinks\[codex\]"):
                    await lm(_messages())
                with self.assertRaisesRegex(ImportError, r"synalinks\[codex\]"):
                    await OAuthLanguageModel.supported_models("codex")


class OAuthLanguageModelTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(
            oauth_language_model, "_codex_executable", return_value="codex"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_provider_info(self):
        self.assertEqual(
            OAuthLanguageModel.provider_info("codex"),
            {
                "provider": "codex",
                "env": [{"name": "CODEX_HOME", "required": False, "secret": False}],
                "default_api_base": None,
                "lists_models_from": "server",
            },
        )
        with self.assertRaisesRegex(ValueError, "Unsupported language model"):
            OAuthLanguageModel.provider_info("openai")

    async def test_supported_models_are_those_of_the_model_picker(self):
        catalog = {
            "models": [*_CATALOG["models"], {"slug": "internal", "visibility": "hide"}]
        }
        with patch(_RUN, Recorder(catalog=json.dumps(catalog))):
            models = await OAuthLanguageModel.supported_models("codex")
        self.assertEqual(models, ["codex/gpt-5.5", "codex/gpt-6.1-sol"])
        with patch(_RUN, Recorder(catalog="")):
            with self.assertRaisesRegex(ConnectionError, "no model catalog"):
                await OAuthLanguageModel.supported_models("codex")

    def test_model_must_be_a_codex_model(self):
        for model in ("gpt-6.1-sol", "openai/gpt-6.1-sol", "codex/", "claude/sonnet"):
            with self.assertRaisesRegex(ValueError, "expects `model`"):
                OAuthLanguageModel(model=model)
        with self.assertRaisesRegex(ValueError, "need to set the `model`"):
            OAuthLanguageModel()

    def test_get_routes_codex_identifiers(self):
        lm = synalinks.language_models.get("codex/gpt-6.1-sol")
        self.assertIsInstance(lm, OAuthLanguageModel)
        self.assertNotIsInstance(
            synalinks.language_models.get("openai/gpt-4o-mini"), OAuthLanguageModel
        )
        self.assertIs(synalinks.OAuthLanguageModel, OAuthLanguageModel)
        self.assertEqual(OAuthLanguageModel.supported_providers(), ["codex"])
        self.assertNotIn("codex", LanguageModel.supported_providers())

    def test_serialization_roundtrip(self):
        lm = OAuthLanguageModel(
            model="codex/gpt-6.1-sol",
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
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", reasoning_effort="medium")
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
        self.assertEqual(call["command"][call["command"].index("-m") + 1], "gpt-6.1-sol")
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
        # Without a system message, a short default one still replaces the
        # coding-agent prompt of the CLI.
        self.assertEqual(recorder.calls[0]["instructions"], "Answer the user's message.")
        command = recorder.calls[0]["command"]
        self.assertNotIn("-m", command)
        self.assertNotIn("--output-schema", command)
        self.assertFalse(any(a.startswith("model_reasoning_effort") for a in command))

    async def test_per_call_reasoning_effort_overrides_default(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", reasoning_effort="high")
        with patch(_RUN, recorder):
            await lm(_messages(), reasoning_effort="disable")
            await lm(_messages(), reasoning_effort="none")
        # "disable" sends the lowest effort, "none" keeps the model default.
        self.assertIn('model_reasoning_effort="low"', recorder.calls[0]["command"])
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
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", reasoning_effort="low")
        with patch(_RUN, recorder):
            result = await lm(_messages(), schema=Reasoned.get_schema())
        # The effort is sent, the field stays in the schema and the model's
        # text is kept (the one-line summary does not overwrite it).
        self.assertIn('model_reasoning_effort="low"', recorder.calls[0]["command"])
        self.assertEqual(recorder.calls[0]["schema"]["required"], ["thinking", "answer"])
        self.assertEqual(result.get_json(), answer)

    async def test_efforts_the_backend_rejects_are_sent_as_low(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
        with patch(_RUN, recorder):
            await lm(_messages(), reasoning_effort="minimal")
            await lm(_messages(), reasoning_effort="disable")
        for call in recorder.calls:
            self.assertIn('model_reasoning_effort="low"', call["command"])

    async def test_reasoning_effort_follows_the_base_class(self):
        # Same contract as `LanguageModel`: the effort of the call wins over
        # the one of the constructor, `None` (what a module passes when it
        # sets no effort) and "none" both keep the default of the model.
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", reasoning_effort="high")
        with patch(_RUN, recorder):
            await lm(_messages())
            await lm(_messages(), reasoning_effort="low")
            await lm(_messages(), reasoning_effort=None)
            await lm(_messages(), reasoning_effort="none")
        efforts = [
            [a for a in call["command"] if a.startswith("model_reasoning_effort")]
            for call in recorder.calls
        ]
        self.assertEqual(
            efforts,
            [
                ['model_reasoning_effort="high"'],
                ['model_reasoning_effort="low"'],
                [],
                [],
            ],
        )

    async def test_unsupported_parameters_are_ignored_with_a_warning(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", temperature=0.0, max_tokens=10)
        with patch(_RUN, recorder):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await lm(_messages())
        self.assertEqual(result.get("content"), "ok")
        self.assertTrue(
            any("['max_tokens', 'temperature']" in str(w.message) for w in caught)
        )
        self.assertFalse(any("temperature" in a for a in recorder.calls[0]["command"]))

    async def test_streaming_is_rejected(self):
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "streaming"):
            await lm(_messages(), streaming=True)

    async def test_free_form_schema_is_rejected_before_any_call(self):
        recorder = Recorder(stdout=_events("{}"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
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
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", retry=3, retry_max_wait=0)
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
            model="codex/gpt-6.1-sol", retry=2, retry_max_wait=0, fallback=fallback
        )
        calls = []

        async def run(self, command, prompt, cwd=None):
            if "exec" not in command:
                return await recorder(command, prompt, cwd=cwd)
            calls.append(self.model)
            if self is fallback:
                return _events("from fallback"), "", 0
            return await recorder(command, prompt, cwd=cwd)

        with patch(_RUN, run):
            result = await lm(_messages())
        self.assertEqual(result.get("content"), "from fallback")
        self.assertEqual(
            calls, ["codex/gpt-6.1-sol", "codex/gpt-6.1-sol", "codex/default"]
        )
        self.assertEqual(lm.cumulated_fallback_activations, 1)

    async def test_fallback_receives_the_caller_parameters_untouched(self):
        recorder = Recorder(stderr="connection reset", returncode=1)
        lm = OAuthLanguageModel(
            model="codex/gpt-6.1-sol",
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
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
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

    async def test_model_catalog_is_read_once_and_handed_without_tools(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
        with patch(_RUN, recorder):
            await lm(_user("one"))
            await lm(_user("two"))
        self.assertEqual(recorder.catalog_reads, 1)
        for call in recorder.calls:
            self.assertEqual(json.loads(call["catalog"]), _without_tools(_CATALOG))

    async def test_missing_model_catalog_is_reported_once(self):
        # An older CLI has no `codex debug models`: the completion still runs,
        # with the tools the CLI gives the model.
        recorder = Recorder(stdout=_events("ok"), catalog="")
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
        with patch(_RUN, recorder):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await lm(_user("one"))
                await lm(_user("two"))
        self.assertEqual(result.get("content"), "ok")
        self.assertEqual(recorder.catalog_reads, 1)
        reports = [w for w in caught if "no model catalog" in str(w.message)]
        self.assertEqual(len(reports), 1)
        for call in recorder.calls:
            self.assertNotIn("catalog", call)

    async def test_file_cache_hit_skips_the_cli(self):
        recorder = Recorder(stdout=_events("cached answer"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", cache_dir=self.get_temp_dir())
        with patch(_RUN, recorder):
            first = await lm(_messages())
            second = await lm(_messages())
        self.assertEqual(first.get_json(), second.get_json())
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(lm.cumulated_cache_hits, 1)

    async def test_file_cache_is_keyed_by_reasoning_effort(self):
        recorder = Recorder(stdout=_events("answer"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", cache_dir=self.get_temp_dir())
        with patch(_RUN, recorder):
            await lm(_messages(), reasoning_effort="low")
            await lm(_messages(), reasoning_effort="high")
            await lm(_messages(), reasoning_effort="low")
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(lm.cumulated_cache_hits, 1)

    async def test_ignored_parameters_are_reported_once(self):
        recorder = Recorder(stdout=_events("ok"))
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
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
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", retry=3, retry_max_wait=0)
        with patch.object(
            oauth_language_model.asyncio,
            "create_subprocess_exec",
            side_effect=FileNotFoundError("codex"),
        ) as spawn:
            result = await lm(_messages())
        self.assertIsNone(result)
        self.assertEqual(spawn.call_count, 1)

    async def test_subprocess_environment_is_an_allowlist(self):
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", retry=1)
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
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", timeout=0.2)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            await lm._run(["sleep", "30"], "")


# Tool calling through `codex app-server`.


async def add(a: int, b: int):
    """Adds two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return {"result": a + b}


async def multiply(a: int, b: int):
    """Multiplies two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return {"result": a * b}


class Question(DataModel):
    question: str = Field(description="The question")


class Answer(DataModel):
    result: int = Field(description="The result")


_USAGE = {
    "last": {
        "inputTokens": 300,
        "cachedInputTokens": 100,
        "outputTokens": 30,
        "reasoningOutputTokens": 10,
    }
}


def _notification(method, **params):
    return {"method": method, "params": params}


class FakeAppServer:
    """Stands in for `OAuthLanguageModel._app_server`: a scripted app-server.

    `turns` lists what each `turn/start` produces, in order: `{"calls":
    [(tool, arguments), ...]}` (the model calls tools: one `item/tool/call`
    request each, the turn only completes once interrupted), `{"message":
    text}` (the model answers), or `{"item": type}` (the model acted through
    the CLI). Every request is recorded in `requests`, one process per
    completion is counted in `processes`, and `config` is what `config/read`
    returns.
    """

    def __init__(self, turns, config=None):
        self.turns = list(turns)
        self.config = config or {}
        self.requests = []
        self.notifications = []
        self.commands = []
        self.processes = 0
        self._events = []
        self._request_id = 0
        self._call_id = 0

    def __call__(self, command, cwd):
        self.commands.append((command, cwd))
        return self

    async def __aenter__(self):
        self.processes += 1
        self._events = []
        return self

    async def __aexit__(self, *exc_info):
        return False

    def _completion(self, status):
        return [
            _notification("thread/tokenUsage/updated", threadId="t", tokenUsage=_USAGE),
            _notification(
                "turn/completed", threadId="t", turn={"id": "u", "status": status}
            ),
        ]

    async def request(self, method, params=None):
        self.requests.append((method, params))
        if method == "initialize":
            return {"userAgent": "synalinks/0.164.0"}
        if method == "config/read":
            return {"config": self.config}
        if method == "thread/start":
            return {"thread": {"id": "t"}}
        if method == "thread/inject_items":
            return {}
        if method == "turn/start":
            turn = self.turns.pop(0)
            self._events = [
                _notification("item/started", item={"type": "userMessage", "id": "m"})
            ]
            if "calls" in turn:
                for tool, arguments in turn["calls"]:
                    call = {
                        "callId": f"call_{self._call_id}",
                        "tool": tool,
                        "arguments": arguments,
                    }
                    self._call_id += 1
                    self._events.append(
                        _notification(
                            "item/started", item={"type": "dynamicToolCall", **call}
                        )
                    )
                    self._request_id += 1
                    self._events.append(
                        {
                            "id": self._request_id,
                            "method": "item/tool/call",
                            "params": {"threadId": "t", "turnId": "u", **call},
                        }
                    )
            elif "item" in turn:
                self._events.append(
                    _notification("item/started", item={"type": turn["item"], "id": "x"})
                )
                self._events += self._completion("completed")
            else:
                self._events.append(
                    _notification(
                        "item/completed",
                        item={"type": "agentMessage", "id": "a", "text": turn["message"]},
                    )
                )
                self._events += self._completion("completed")
            return {"turn": {"id": "u"}}
        if method == "turn/interrupt":
            self._events += self._completion("interrupted")
            return {}
        raise AssertionError(f"Unexpected request: {method}")

    async def notify(self, method, params=None):
        self.notifications.append(method)

    async def next_event(self):
        return self._events.pop(0) if self._events else None

    def params(self, method):
        """The params of the requests of `method`, one per process."""
        return [params for name, params in self.requests if name == method]


def _trajectory():
    """A conversation where the model already called `add`."""
    return ChatMessages(
        messages=[
            ChatMessage(role=ChatRole.SYSTEM, content="You compute."),
            ChatMessage(role=ChatRole.USER, content="What is (17 + 25) * 3?"),
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'},
                    }
                ],
            ),
            ChatMessage(
                role=ChatRole.TOOL, tool_call_id="call_0", content='{"result": 42}'
            ),
        ]
    )


_TOOLS = [synalinks.Tool(add), synalinks.Tool(multiply)]


class HistoryItemsTest(testing.TestCase):
    def test_messages_become_response_items(self):
        messages = [
            {"role": "system", "content": "You compute."},
            {"role": "user", "content": "What is (17 + 25) * 3?"},
            {
                "role": "assistant",
                "content": "Adding first.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "add", "arguments": {"a": 17, "b": 25}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": '{"result": 42}'},
        ]
        instructions, items = _history_items(messages)
        self.assertEqual(instructions, "You compute.")
        self.assertEqual(
            items,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is (17 + 25) * 3?"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Adding first."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_0",
                    "name": "add",
                    "arguments": '{"a": 17, "b": 25}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_0",
                    "output": '{"result": 42}',
                },
            ],
        )

    def test_dynamic_tools_from_wire_declarations(self):
        self.assertEqual(
            _dynamic_tools(
                [{"name": "ping"}, {"name": "add", "parameters": {"type": "object"}}]
            ),
            [
                {
                    "type": "function",
                    "name": "ping",
                    "description": "",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "add",
                    "description": "",
                    "inputSchema": {"type": "object"},
                },
            ],
        )

    def test_default_effort_comes_from_the_catalog(self):
        catalog = json.dumps(_CATALOG)
        self.assertEqual(_default_effort(catalog, "gpt-6.1-sol"), "low")
        self.assertIsNone(_default_effort(catalog, "gpt-5.5"))
        self.assertIsNone(_default_effort(catalog, None))
        self.assertIsNone(_default_effort("", "gpt-6.1-sol"))


class AppServerCommandTest(testing.TestCase):
    def test_command_is_locked_down(self):
        command = _build_app_server_command(catalog_path="/tmp/m.json", tools=True)
        self.assertEqual(command[0], "codex")
        self.assertEqual(command[-3:], ["app-server", "--listen", "stdio://"])
        # The overrides of `codex exec`, plus what the app-server cannot
        # ignore from the user's configuration, and no harness function.
        for override in (
            'web_search="disabled"',
            "features.shell_tool=false",
            'model_provider="openai"',
            "notify=[]",
            "tools.experimental_request_user_input.enabled=false",
            'model_catalog_json="/tmp/m.json"',
        ):
            self.assertIn(override, command)
        self.assertNotIn("exec", command)

    def test_harness_function_stays_when_no_tool_is_declared(self):
        # Without a declared function, the CLI would send the instructions
        # twice: the key is only worth it with tools.
        command = _build_app_server_command()
        self.assertNotIn("tools.experimental_request_user_input.enabled=false", command)


class AppServerToolCallingTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(
            oauth_language_model, "_codex_executable", return_value="codex"
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.recorder = Recorder()
        run = patch(_RUN, self.recorder)
        run.start()
        self.addCleanup(run.stop)
        # The fake streams nothing: no need to let a response end.
        drain = patch.object(oauth_language_model, "_STREAM_DRAIN_S", 0)
        drain.start()
        self.addCleanup(drain.stop)

    def _lm(self, fake, **kwargs):
        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol", retry_max_wait=0, **kwargs)
        patcher = patch.object(OAuthLanguageModel, "_app_server", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return lm

    async def test_tool_call_turn_is_native(self):
        fake = FakeAppServer([{"calls": [("add", {"a": 17, "b": 25})]}])
        lm = self._lm(fake, reasoning_effort="medium")
        result = await lm(_user("What is (17 + 25) * 3?"), tools=_TOOLS)

        self.assertEqual(
            result.get_json(),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'},
                    }
                ],
            },
        )
        self.assertEqual(lm.last_call_finish_reason, "tool_calls")
        self.assertEqual(fake.processes, 1)
        self.assertEqual(
            [method for method, _ in fake.requests],
            ["initialize", "config/read", "thread/start", "turn/start", "turn/interrupt"],
        )
        self.assertEqual(fake.notifications, ["initialized"])
        initialize = fake.params("initialize")[0]
        self.assertEqual(initialize["capabilities"], {"experimentalApi": True})
        thread = fake.params("thread/start")[0]
        self.assertEqual(thread["model"], "gpt-6.1-sol")
        self.assertEqual(thread["baseInstructions"], "Answer the user's message.")
        self.assertEqual(thread["developerInstructions"], "")
        self.assertTrue(thread["ephemeral"])
        self.assertEqual(thread["sandbox"], "read-only")
        self.assertEqual(thread["approvalPolicy"], "never")
        self.assertEqual(thread["cwd"], fake.commands[0][1])
        self.assertEqual(
            [(t["type"], t["name"]) for t in thread["dynamicTools"]],
            [("function", "add"), ("function", "multiply")],
        )
        self.assertEqual(thread["dynamicTools"][0]["description"], "Adds two numbers.")
        self.assertEqual(thread["dynamicTools"][0]["inputSchema"]["required"], ["a", "b"])
        turn = fake.params("turn/start")[0]
        self.assertEqual(
            turn["input"], [{"type": "text", "text": "What is (17 + 25) * 3?"}]
        )
        self.assertEqual(turn["effort"], "medium")
        self.assertNotIn("outputSchema", turn)
        # The catalog is read once with the CLI, and handed over without tools.
        self.assertEqual(self.recorder.catalog_reads, 1)
        self.assertIn('model_catalog_json="', " ".join(fake.commands[0][0]))

    async def test_history_is_injected_as_response_items(self):
        fake = FakeAppServer([{"calls": [("multiply", {"a": 42, "b": 3})]}])
        lm = self._lm(fake)
        result = await lm(_trajectory(), tools=_TOOLS)

        self.assertEqual(
            result.get_json()["tool_calls"][0]["function"]["name"], "multiply"
        )
        thread = fake.params("thread/start")[0]
        self.assertEqual(thread["baseInstructions"], "You compute.")
        self.assertEqual(
            fake.params("thread/inject_items")[0]["items"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is (17 + 25) * 3?"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_0",
                    "name": "add",
                    "arguments": '{"a": 17, "b": 25}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_0",
                    "output": '{"result": 42}',
                },
            ],
        )
        # The turn starts on the history alone: no user message follows it.
        self.assertEqual(fake.params("turn/start")[0]["input"], [])

    async def test_default_effort_is_the_one_of_the_catalog(self):
        fake = FakeAppServer([{"message": "126"}, {"message": "126"}])
        lm = self._lm(fake)
        await lm(_user("hi"), tools=_TOOLS, tool_choice="none")
        self.assertEqual(fake.params("turn/start")[0]["effort"], "low")
        await lm(
            _user("hi"), tools=_TOOLS, tool_choice="none", reasoning_effort="disable"
        )
        self.assertEqual(fake.params("turn/start")[1]["effort"], "low")

    async def test_tool_choice_none_declares_no_tool(self):
        fake = FakeAppServer([{"message": "The result is 126."}])
        lm = self._lm(fake)
        result = await lm(_trajectory(), tools=_TOOLS, tool_choice="none")

        self.assertEqual(result.get_json()["content"], "The result is 126.")
        self.assertNotIn("tool_calls", result.get_json())
        self.assertEqual(lm.last_call_finish_reason, "stop")
        self.assertNotIn("dynamicTools", fake.params("thread/start")[0])
        self.assertEqual(len(fake.params("thread/inject_items")[0]["items"]), 3)

    async def test_usage_is_the_one_of_the_last_request(self):
        fake = FakeAppServer([{"message": "126"}])
        lm = self._lm(fake)
        await lm(_user("hi"), tools=_TOOLS, tool_choice="none")
        self.assertEqual(lm.last_call_prompt_tokens, 300)
        self.assertEqual(lm.last_call_completion_tokens, 30)
        self.assertEqual(lm.last_call_cached_tokens, 100)
        self.assertEqual(lm.last_call_reasoning_tokens, 10)

    async def test_structured_answer_over_a_tool_history(self):
        fake = FakeAppServer([{"message": '{"result": 126}'}])
        lm = self._lm(fake)
        result = await lm(_trajectory(), schema=Answer.get_schema())

        self.assertEqual(result.get_json(), {"result": 126})
        turn = fake.params("turn/start")[0]
        self.assertEqual(turn["outputSchema"]["required"], ["result"])
        self.assertFalse(turn["outputSchema"]["additionalProperties"])
        self.assertNotIn("dynamicTools", fake.params("thread/start")[0])

    async def test_required_tool_choice_without_a_call_is_retried(self):
        fake = FakeAppServer([{"message": "126"}, {"calls": [("add", {"a": 1, "b": 2})]}])
        lm = self._lm(fake, retry=2)
        result = await lm(_user("hi"), tools=_TOOLS, tool_choice="required")
        self.assertEqual(result.get_json()["tool_calls"][0]["function"]["name"], "add")
        self.assertEqual(fake.processes, 2)
        self.assertIn(
            "Call at least one of the tools",
            fake.params("thread/start")[0]["baseInstructions"],
        )

    async def test_named_tool_choice_offers_that_tool_only(self):
        fake = FakeAppServer([{"calls": [("multiply", {"a": 2, "b": 3})]}])
        lm = self._lm(fake)
        choice = {"type": "function", "function": {"name": "multiply"}}
        result = await lm(_user("hi"), tools=_TOOLS, tool_choice=choice)
        self.assertEqual(
            result.get_json()["tool_calls"][0]["function"]["name"], "multiply"
        )
        thread = fake.params("thread/start")[0]
        self.assertEqual([t["name"] for t in thread["dynamicTools"]], ["multiply"])
        self.assertIn("Call the tool `multiply` now.", thread["baseInstructions"])

    async def test_undeclared_named_tool_fails_before_any_process(self):
        fake = FakeAppServer([])
        lm = self._lm(fake, retry=3)
        choice = {"type": "function", "function": {"name": "divide"}}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = await lm(_user("hi"), tools=_TOOLS, tool_choice=choice)
        self.assertIsNone(result)
        self.assertEqual(fake.processes, 0)

    async def test_tool_use_through_the_cli_fails_without_retry(self):
        fake = FakeAppServer([{"item": "commandExecution"}])
        lm = self._lm(fake, retry=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await lm(_user("hi"), tools=_TOOLS)
        self.assertIsNone(result)
        self.assertEqual(fake.processes, 1)
        self.assertTrue(any("commandExecution" in str(w.message) for w in caught))

    async def test_mcp_servers_of_the_user_config_are_disabled(self):
        fake = FakeAppServer(
            [{"message": "hi"}],
            config={"mcp_servers": {"docs": {"command": "docs-mcp"}}},
        )
        lm = self._lm(fake)
        await lm(_user("hi"), tools=_TOOLS, tool_choice="none")
        self.assertEqual(
            fake.params("thread/start")[0]["config"],
            {"mcp_servers": {"docs": {"enabled": False}}},
        )

    async def test_native_shape_equals_the_litellm_path(self):
        call = {
            "id": "call_0",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'},
        }
        fake = FakeAppServer([{"calls": [("add", {"a": 17, "b": 25})]}])
        lm = self._lm(fake)
        ours = await lm(_user("hi"), tools=_TOOLS)
        api = LanguageModel(model="openai/gpt-4o-mini")
        with patch("litellm.acompletion") as mock_completion:
            mock_completion.return_value = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [call],
                        },
                    }
                ],
                "usage": {},
            }
            theirs = await api(_user("hi"), tools=_TOOLS)
        self.assertEqual(ours.get_json(), theirs.get_json())

    async def test_function_calling_agent_runs_unchanged(self):
        fake = FakeAppServer(
            [
                {"calls": [("add", {"a": 17, "b": 25})]},
                {"calls": [("multiply", {"a": 42, "b": 3})]},
                {"message": "(17 + 25) * 3 = 126."},
                {"message": '{"result": 126}'},
            ]
        )
        lm = self._lm(fake)
        inputs = synalinks.Input(data_model=Question)
        outputs = await synalinks.FunctionCallingAgent(
            data_model=Answer,
            tools=_TOOLS,
            language_model=lm,
            autonomous=True,
            max_iterations=4,
            return_inputs_with_trajectory=True,
        )(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)
        result = await program(Question(question="What is (17 + 25) * 3?"))

        self.assertEqual(result.get_json()["result"], 126)
        self.assertEqual(fake.processes, 4)
        messages = result.get_json()["messages"]
        self.assertEqual(
            [(m["role"], m.get("tool_call_id")) for m in messages],
            [
                ("assistant", None),
                ("tool", "call_0"),
                ("assistant", None),
                ("tool", "call_1"),
            ],
        )
        self.assertEqual(messages[1]["content"], {"result": 42})
        self.assertEqual(messages[3]["content"], {"result": 126})
        # Each step sends the trajectory so far as native items, and the
        # final answer is constrained by the schema over that same history.
        injected = fake.params("thread/inject_items")
        self.assertEqual([len(p["items"]) for p in injected], [3, 5, 5])
        self.assertEqual(injected[-1]["items"][-1]["type"], "function_call_output")
        self.assertIn("outputSchema", fake.params("turn/start")[-1])
        self.assertNotIn("dynamicTools", fake.params("thread/start")[-1])
