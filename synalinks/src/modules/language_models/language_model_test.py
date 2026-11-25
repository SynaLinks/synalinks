# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import base64
import copy
import json
import os
import threading
import warnings
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from unittest.mock import patch

from litellm.types.utils import Choices
from litellm.types.utils import CompletionTokensDetailsWrapper
from litellm.types.utils import Message
from litellm.types.utils import ModelResponse
from litellm.types.utils import PromptTokensDetailsWrapper
from litellm.types.utils import ServerToolUse
from litellm.types.utils import Usage

import synalinks
from synalinks.src import testing
from synalinks.src.backend import Audio
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.backend import Image
from synalinks.src.backend.common.op_scope import _OP_SCOPE
from synalinks.src.backend.pydantic.media import allow_media_root
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.modules.language_models.language_model import _tool_to_wire

_SAMPLE_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "guides",
    "traced_qa.png",
)
allow_media_root(os.path.dirname(_SAMPLE_IMAGE))


class LanguageModelTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_call_api_without_structured_output(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": "Hello, how can I help you?"}}]
        }

        expected = ChatMessage(
            role=ChatRole.ASSISTANT, content="Hello, how can I help you?"
        )
        result = await language_model(messages)
        self.assertEqual(result.get_json(), ChatMessage(**result.get_json()).get_json())
        self.assertEqual(result.get_json(), expected.get_json())

    @patch("litellm.acompletion")
    async def test_call_api_with_structured_output(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")

        messages = ChatMessages(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    content="What is the french city of aerospace and robotics?",
                )
            ]
        )

        expected_string = (
            """{"rationale": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        expected = AnswerWithRationale(
            rationale=(
                "Toulouse hosts numerous research institutions and universities "
                "that specialize in aerospace engineering and robotics, such as "
                "the Institut Supérieur de l'Aéronautique et de l'Espace "
                "(ISAE-SUPAERO) and the "
                "French National Centre for Scientific Research (CNRS)"
            ),
            answer="Toulouse",
        )
        result = await language_model(messages, schema=AnswerWithRationale.get_schema())
        self.assertEqual(
            result.get_json(), AnswerWithRationale(**result.get_json()).get_json()
        )
        self.assertEqual(result.get_json(), expected.get_json())

    @patch("litellm.acompletion")
    async def test_structured_output_via_tool_call_reads_the_arguments(
        self, mock_completion
    ):
        # A reply can carry the structured output as a tool call (e.g. with a
        # caller's own tool payload): the JSON is the call's arguments and
        # `content` is empty. It used to be read from `content`, fail to parse,
        # and score every row as None.
        language_model = LanguageModel(model="bedrock/meta.llama3-70b-instruct-v1:0")

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "structured_output",
                                    "arguments": '{"answer": "B"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Pick a letter.")]
        )
        result = await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})

    @patch("litellm.acompletion")
    async def test_openrouter_requires_endpoints_with_structured_outputs(
        self, mock_completion
    ):
        # OpenRouter constrains the reply natively, but by default may route to
        # an endpoint that ignores `response_format` and answers in prose.
        language_model = LanguageModel(model="openrouter/anthropic/claude-fable-5.1")

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Pick a letter.")]
        )
        result = await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})

        kwargs = mock_completion.call_args.kwargs
        self.assertNotIn("tools", kwargs)
        response_format = kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertIn("answer", response_format["json_schema"]["schema"]["properties"])
        self.assertEqual(
            kwargs["provider"],
            {"require_parameters": True, "ignore": ["coreweave", "nebius"]},
        )

        # The caller's own routing preferences are kept.
        language_model = LanguageModel(
            model="openrouter/anthropic/claude-fable-5.1",
            provider={"order": ["anthropic"]},
        )
        await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(
            mock_completion.call_args.kwargs["provider"],
            {
                "order": ["anthropic"],
                "require_parameters": True,
                "ignore": ["coreweave", "nebius"],
            },
        )

    @patch("litellm.acompletion")
    async def test_openrouter_blacklist_adds_to_the_callers_and_spares_a_chosen_one(
        self, mock_completion
    ):
        """The blacklist is added to the caller's own `ignore`, without
        repeating a name; a provider the caller routes to on purpose (`order`,
        `only`) is not ignored."""

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Pick a letter.")]
        )
        language_model = LanguageModel(
            model="openrouter/moonshotai/kimi-k2.7-code",
            provider={"ignore": ["Venice", "CoreWeave"], "only": ["Nebius"]},
        )
        await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(
            mock_completion.call_args.kwargs["provider"],
            {
                "ignore": ["Venice", "CoreWeave"],
                "only": ["Nebius"],
                "require_parameters": True,
            },
        )

    @patch("litellm.acompletion")
    async def test_groq_asks_for_a_strict_json_schema(self, mock_completion):
        # litellm sends it natively to strict-mode models, and falls back to a
        # forced tool call itself for the others.
        language_model = LanguageModel(model="groq/openai/gpt-oss-120b")

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Pick a letter.")]
        )
        result = await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})
        kwargs = mock_completion.call_args.kwargs
        self.assertNotIn("tools", kwargs)
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])

    @patch("litellm.acompletion")
    async def test_cohere_sends_its_schema_in_the_body(self, mock_completion):
        # Cohere's v2 chat constrains the reply to `response_format.schema`,
        # which litellm refuses as a parameter, so it travels in `extra_body`.
        language_model = LanguageModel(
            model="cohere/command-a-03-2025", extra_body={"safety_mode": "OFF"}
        )

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Pick a letter.")]
        )
        result = await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})
        kwargs = mock_completion.call_args.kwargs
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("response_format", kwargs)
        self.assertEqual(kwargs["extra_body"]["safety_mode"], "OFF")
        response_format = kwargs["extra_body"]["response_format"]
        self.assertEqual(response_format["type"], "json_object")
        self.assertIn("answer", response_format["schema"]["properties"])

    @patch("litellm.acompletion")
    async def test_bedrock_asks_for_a_strict_json_schema(self, mock_completion):
        # litellm maps it to bedrock's native `outputConfig.textFormat` where
        # the model supports it, and falls back to a forced tool call itself.
        language_model = LanguageModel(
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0"
        )

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Pick a letter.")]
        )
        result = await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})
        kwargs = mock_completion.call_args.kwargs
        self.assertNotIn("tools", kwargs)
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])

    @patch("litellm.acompletion")
    async def test_a_caller_can_override_the_structured_output_payload(
        self, mock_completion
    ):
        # The branch picks a payload per provider, but a model it guesses wrong
        # about was impossible to drive: the branch overwrote whatever the
        # caller passed. Now it only fills in what is unset.
        response_format = {"type": "json_object"}
        language_model = LanguageModel(
            model="bedrock/meta.llama3-70b-instruct-v1:0",
            response_format=response_format,
        )

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Pick a letter.")]
        )
        await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(
            mock_completion.call_args.kwargs["response_format"], response_format
        )

    @patch("litellm.acompletion")
    async def test_structured_output_tool_call_wins_over_a_content_preamble(
        self, mock_completion
    ):
        # Some backends narrate before calling the tool.
        language_model = LanguageModel(model="bedrock/meta.llama3-70b-instruct-v1:0")

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I'll answer with the tool.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "structured_output",
                                    "arguments": '{"answer": "true"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Yes or no?")]
        )
        result = await language_model(messages, schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "true"})

    @patch("litellm.acompletion")
    async def test_call_api_streaming_mode(self, mock_completion):
        language_model = LanguageModel(model="ollama/deepseek-r1")

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        mock_response_iterator = iter(
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo,"}}]},
                {"choices": [{"delta": {"content": " how"}}]},
                {"choices": [{"delta": {"content": " can"}}]},
                {"choices": [{"delta": {"content": " I"}}]},
                {"choices": [{"delta": {"content": " help"}}]},
                {"choices": [{"delta": {"content": " you?"}}]},
            ]
        )

        mock_completion.return_value = mock_response_iterator

        expected = "Hello, how can I help you?"

        response = await language_model(messages, streaming=True)

        result = ""
        async for msg in response:
            result += msg.get("content")

        self.assertEqual(result, expected)
        # Consuming the stream to exhaustion records one streamed call and its
        # time-to-first / time-to-last token latencies (all-time counters; no
        # op_scope is active here).
        self.assertEqual(language_model.cumulated_streaming_calls, 1)
        self.assertGreaterEqual(language_model.cumulated_streaming_ttft_s, 0.0)
        self.assertGreaterEqual(
            language_model.cumulated_streaming_ttlt_s,
            language_model.cumulated_streaming_ttft_s,
        )
        # No trajectory_scope active -> whole-trajectory TTFT is not recorded.
        self.assertEqual(language_model.cumulated_trajectory_calls, 0)
        self.assertEqual(language_model.cumulated_trajectory_ttft_s, 0.0)

    @patch("litellm.acompletion")
    async def test_streaming_records_trajectory_ttft(self, mock_completion):
        """Inside a `trajectory_scope` (set by an agent), a streamed call also
        records whole-trajectory time-to-first-token, anchored at the scope's
        start -- so it is >= the per-call streaming TTFT (same first token, an
        earlier anchor)."""
        from synalinks.src.backend.common.op_scope import trajectory_scope

        language_model = LanguageModel(model="ollama/deepseek-r1")
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        mock_completion.return_value = iter(
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
            ]
        )

        with trajectory_scope():
            response = await language_model(messages, streaming=True)
            async for _ in response:
                pass

        self.assertEqual(language_model.cumulated_trajectory_calls, 1)
        self.assertGreaterEqual(language_model.cumulated_trajectory_ttft_s, 0.0)
        self.assertGreaterEqual(
            language_model.cumulated_trajectory_ttft_s,
            language_model.cumulated_streaming_ttft_s,
        )

    @patch("litellm.acompletion")
    async def test_streaming_abandoned_before_exhaustion_records_nothing(
        self, mock_completion
    ):
        """A caller that stops iterating before the stream is exhausted leaves
        no terminal timing, so no streamed call is recorded (a partial
        consumption can't yield a meaningful time-to-last-token)."""
        language_model = LanguageModel(model="ollama/deepseek-r1")
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        mock_completion.return_value = iter(
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo,"}}]},
                {"choices": [{"delta": {"content": " how"}}]},
            ]
        )

        response = await language_model(messages, streaming=True)

        # Consume only the first chunk, then abandon the iterator.
        first = await response.__anext__()
        self.assertEqual(first.get("content"), "Hel")

        # StopAsyncIteration never fires, so the latencies are never recorded.
        self.assertEqual(language_model.cumulated_streaming_calls, 0)
        self.assertEqual(language_model.cumulated_streaming_ttft_s, 0.0)
        self.assertEqual(language_model.cumulated_streaming_ttlt_s, 0.0)

    @patch("litellm.acompletion")
    async def test_call_api_ignores_none_response_cost(self, mock_completion):
        language_model = LanguageModel(model="hosted_vllm/Qwen/Qwen3-4B")

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        class MockResponse(dict):
            def __init__(self):
                super().__init__(
                    {"choices": [{"message": {"content": "Hello, how can I help you?"}}]}
                )
                self._hidden_params = {"response_cost": None}

        mock_completion.return_value = MockResponse()

        result = await language_model(messages)

        expected = ChatMessage(
            role=ChatRole.ASSISTANT, content="Hello, how can I help you?"
        )
        self.assertEqual(result.get_json(), expected.get_json())
        self.assertEqual(language_model.last_call_cost, 0.0)
        self.assertEqual(language_model.cumulated_cost, 0.0)

    @patch("litellm.acompletion")
    async def test_hosted_vllm_structured_output_sets_json_schema_name(
        self, mock_completion
    ):
        language_model = LanguageModel(
            model="hosted_vllm/Qwen/Qwen3-4B",
            api_base="http://localhost:8000/v1",
        )

        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="What is 2+2?")]
        )

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer":"4"}'}}]
        }

        result = await language_model(messages, schema=_FinishReasonAnswer.get_schema())

        self.assertEqual(result.get_json(), {"answer": "4"})
        response_format = mock_completion.call_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "structured_output")

    @patch("litellm.acompletion")
    async def test_reasoning_effort_disable_sends_think_false_for_ollama(
        self, mock_completion
    ):
        """`reasoning_effort="none"` turns native thinking off on ollama
        (which reasons by default) by sending `think=False`; `None` leaves the
        model at its default and sends nothing.
        """
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        ollama_lm = LanguageModel(model="ollama_chat/qwen3:8b")
        await ollama_lm(messages, reasoning_effort="none")
        self.assertIs(mock_completion.call_args.kwargs.get("think"), False)

        await ollama_lm(messages, reasoning_effort=None)
        self.assertNotIn("think", mock_completion.call_args.kwargs)

    @patch("litellm.acompletion")
    async def test_reasoning_effort_none_values_differ(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        lm = LanguageModel(
            model="vllm/Qwen/Qwen3-8B", api_base="http://h/v1", supports_reasoning=True
        )
        await lm(messages, reasoning_effort=None)
        self.assertNotIn("reasoning_effort", mock_completion.call_args.kwargs)
        self.assertNotIn("extra_body", mock_completion.call_args.kwargs)

        await lm(messages, reasoning_effort="none")
        kwargs = mock_completion.call_args.kwargs
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertIs(
            kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"], False
        )

        # "disable" is the same as "none", without a warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await lm(messages, reasoning_effort="disable")
        self.assertEqual(mock_completion.call_args.kwargs, kwargs)

    @patch("litellm.acompletion")
    async def test_reasoning_effort_none_noop_for_non_ollama(self, mock_completion):
        """Opt-in providers reason only when enabled, so "none" sends no
        `think` flag (it is ollama-specific and would be rejected elsewhere).
        """
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        lm = LanguageModel(model="openai/gpt-4o-mini")
        await lm(messages, reasoning_effort="none")
        self.assertNotIn("think", mock_completion.call_args.kwargs)

    @patch("litellm.acompletion")
    async def test_level_litellm_would_drop_goes_to_the_provider(self, mock_completion):
        # drop_params drops gpt-5's "xhigh" (not in litellm's table): sent anyway.
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        lm = LanguageModel(model="openai/gpt-5", supports_reasoning=True)
        for level in ("minimal", "high", "max"):
            await lm(messages, reasoning_effort=level)
            self.assertEqual(mock_completion.call_args.kwargs["reasoning_effort"], level)
        await lm(messages, reasoning_effort="xhigh")
        kwargs = mock_completion.call_args.kwargs
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["extra_body"], {"reasoning_effort": "xhigh"})

    @patch("litellm.acompletion")
    async def test_level_litellm_refuses_goes_to_the_provider(self, mock_completion):
        # litellm validates Gemini's levels and raises on "xhigh": it is sent anyway.
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        lm = LanguageModel(model="gemini/gemini-2.5-flash", supports_reasoning=True)
        await lm(messages, reasoning_effort="xhigh")
        kwargs = mock_completion.call_args.kwargs
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["extra_body"], {"reasoning_effort": "xhigh"})
        await lm(messages, reasoning_effort="high")
        self.assertEqual(mock_completion.call_args.kwargs["reasoning_effort"], "high")

    @patch("litellm.acompletion")
    async def test_reasoning_effort_disable_sends_each_providers_off_switch(
        self, mock_completion
    ):
        """A model reasoning by default gets its provider's off switch, or its
        lowest effort when it cannot stop reasoning; a non-reasoner, nothing.
        """
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        expected = {
            "anthropic/claude-sonnet-5": {"thinking": {"type": "disabled"}},
            "anthropic/claude-opus-5-5": {"reasoning_effort": "low"},
            "openai/gpt-6-sol": {"reasoning_effort": "none"},
            "openai/gpt-5-mini": {"reasoning_effort": "minimal"},
            "openai/gpt-6-astra": {"reasoning_effort": "low"},
            "gemini/gemini-2.5-flash": {"reasoning_effort": "disable"},
            "gemini/gemini-2.5-pro": {"reasoning_effort": "minimal"},
            "deepseek/deepseek-flash": {"reasoning_effort": "none"},
            "together_ai/zai-org/GLM-5.2": {"reasoning_effort": "none"},
            "mistral/mistral-small-2603": {"reasoning_effort": "none"},
            "xai/grok-4.7": {"reasoning_effort": "low"},
            "anthropic/claude-sonnet-5-5": {"thinking": {"type": "between_tools"}},
            "bedrock/us.anthropic.claude-sonnet-5": {"thinking": {"type": "disabled"}},
            "bedrock/us.anthropic.claude-opus-5-5": {"reasoning_effort": "low"},
            "bedrock/us.openai.gpt-6-sol": {"reasoning_effort": "none"},
            "groq/qwen/qwen3.8-27b": {"reasoning_effort": "none"},
            "groq/openai/gpt-oss-120b": {"reasoning_effort": "low"},
            "mistral/mistral-large-4-0": {"reasoning_effort": "low"},
            "cohere/command-a-reasoning-08-2025": {
                "extra_body": {"thinking": {"type": "disabled"}}
            },
            "cohere/command-a-03-2025": {},
            "openai/gpt-4o-mini": {},
        }
        for model, sent in expected.items():
            with self.subTest(model=model):
                # Every model expecting an off switch reasons natively.
                lm = LanguageModel(model=model, supports_reasoning=bool(sent))
                await lm(messages, reasoning_effort="none")
                kwargs = mock_completion.call_args.kwargs
                keys = ("thinking", "reasoning_effort", "extra_body")
                got = {k: kwargs[k] for k in keys if k in kwargs}
                self.assertEqual(got, sent)

    @patch("litellm.acompletion")
    async def test_native_reasoning_fills_thinking_before_the_answer(
        self, mock_completion
    ):
        """With a reasoning effort, `thinking` is stripped from the schema sent
        and filled from the response's `reasoning_content` after the call. It
        comes back first, where the schema holds it, not appended after the
        answer's fields, and replaces a `thinking` the model wrote anyway.
        """

        class Thought(DataModel):
            thinking: str
            answer: str

        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer": "Toulouse", "thinking": "placeholder"}',
                        "reasoning_content": "Aerospace and robotics: Toulouse.",
                    }
                }
            ]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Which city?")]
        )
        lm = LanguageModel(model="openai/o3-mini")
        with patch.object(LanguageModel, "supports_reasoning", return_value=True):
            result = await lm(
                messages, schema=Thought.get_schema(), reasoning_effort="low"
            )

        sent = mock_completion.call_args.kwargs["response_format"]["json_schema"]
        self.assertNotIn("thinking", sent["schema"]["properties"])
        self.assertEqual(
            list(result.get_json()), ["thinking", "answer"], result.get_json()
        )
        self.assertEqual(
            result.get_json()["thinking"], "Aerospace and robotics: Toulouse."
        )


class MultimodalWireTest(testing.TestCase):
    """End-to-end per-modality check at the provider boundary.

    Drives a multimodal `ChatMessages` through the full `LanguageModel` call
    and inspects the exact `messages` payload handed to `litellm.acompletion`,
    proving each modality's deferred reference is resolved to an inline payload
    before it leaves synalinks. Covers both how content is authored:
    constructed `Image`/`Audio` objects, and dataset-style raw JSON refs.
    """

    def _sent_content(self, mock_completion):
        return mock_completion.call_args.kwargs["messages"][0]["content"]

    @patch("litellm.acompletion")
    async def test_image_constructed_object_reaches_provider_inlined(
        self, mock_completion
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        lm = LanguageModel(model="openai/gpt-4o-mini")
        # Constructed Image(data=...) -> already inline; must arrive as a data: URI.
        messages = ChatMessages(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    content=[
                        "what is this?",
                        Image(data="QUJD", mime_type="image/png"),
                    ],
                )
            ]
        )
        await lm(messages)
        part = self._sent_content(mock_completion)[1]
        self.assertEqual(part["type"], "image_url")
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,QUJD")

    @patch("litellm.acompletion")
    async def test_image_dataset_ref_is_resolved_before_send(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        lm = LanguageModel(model="openai/gpt-4o-mini")
        # Dataset-style: raw image_url ref built via model_validate_json (no
        # Image() constructor) -> must be fetched/inlined per batch at send.
        rendered = (
            '{"messages":[{"role":"user","content":['
            '{"type":"text","text":"caption"},'
            f'{{"type":"image_url","image_url":{{"url":"file://{_SAMPLE_IMAGE}"}}}}]}}]}}'
        )
        messages = ChatMessages.model_validate_json(rendered)
        await lm(messages)
        url = self._sent_content(mock_completion)[1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        with open(_SAMPLE_IMAGE, "rb") as f:
            expected = base64.b64encode(f.read()).decode("ascii")
        self.assertEqual(url.split(",", 1)[1], expected)

    @patch("litellm.acompletion")
    async def test_audio_constructed_object_reaches_provider_inlined(
        self, mock_completion
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        lm = LanguageModel(model="gemini/gemini-2.0-flash")
        messages = ChatMessages(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    content=["transcribe", Audio(data="QUJD", format="wav")],
                )
            ]
        )
        await lm(messages)
        part = self._sent_content(mock_completion)[1]
        self.assertEqual(part["type"], "input_audio")
        self.assertEqual(part["input_audio"], {"data": "QUJD", "format": "wav"})

    @patch("litellm.acompletion")
    async def test_audio_dataset_ref_is_resolved_and_stripped_before_send(
        self, mock_completion
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        lm = LanguageModel(model="gemini/gemini-2.0-flash")
        # Dataset-style audio ref: input_audio carrying a file:// `url`.
        rendered = (
            '{"messages":[{"role":"user","content":['
            '{"type":"text","text":"transcribe"},'
            f'{{"type":"input_audio","input_audio":'
            f'{{"format":"wav","url":"file://{_SAMPLE_IMAGE}"}}}}]}}]}}'
        )
        messages = ChatMessages.model_validate_json(rendered)
        await lm(messages)
        audio = self._sent_content(mock_completion)[1]["input_audio"]
        # Source ref consumed; only inline data + format leave synalinks.
        self.assertEqual(sorted(audio.keys()), ["data", "format"])
        with open(_SAMPLE_IMAGE, "rb") as f:
            expected = base64.b64encode(f.read()).decode("ascii")
        self.assertEqual(audio["data"], expected)
        self.assertEqual(audio["format"], "wav")


def _lm_response(prompt_tokens, completion_tokens, total_tokens=None, cost=None):
    """Build a realistic LiteLLM ModelResponse (mirrors what acompletion returns)."""
    resp = ModelResponse(
        id="test-id",
        model="gpt-4o-mini",
        choices=[
            Choices(
                message=Message(content="hello", role="assistant"),
                index=0,
                finish_reason="stop",
            )
        ],
    )
    resp.usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=(
            total_tokens
            if total_tokens is not None
            else prompt_tokens + completion_tokens
        ),
    )
    if cost is not None:
        resp._hidden_params = {"response_cost": cost}
    return resp


def _chat_messages():
    return ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="Hello")])


def _set_scope(value):
    # Phase scope is contextvars-backed; set it directly on the current
    # context for these counter-routing tests (wall-clock is unaffected).
    _OP_SCOPE.set(value)


class LMCounterPopulationTest(testing.TestCase):
    """End-to-end checks that LiteLLM-shaped responses populate operational
    counters correctly. Guards against drift in LiteLLM's response schema,
    in particular the contract we depend on: `response["usage"]["prompt_tokens"]`,
    `response["usage"]["completion_tokens"]`, `response["usage"]["total_tokens"]`,
    and `response._hidden_params["response_cost"]`.
    """

    @patch("litellm.acompletion")
    async def test_lm_populates_inference_counters(self, mock_completion):
        mock_completion.return_value = _lm_response(
            prompt_tokens=42, completion_tokens=17, cost=0.00123
        )
        lm = LanguageModel(model="ollama/mistral")
        _set_scope("inference")
        try:
            await lm(_chat_messages())
        finally:
            _set_scope(None)
        # All-time counters
        self.assertEqual(lm.cumulated_calls, 1)
        self.assertEqual(lm.cumulated_prompt_tokens, 42)
        self.assertEqual(lm.cumulated_completion_tokens, 17)
        self.assertEqual(lm.cumulated_tokens, 59)
        self.assertAlmostEqual(lm.cumulated_cost, 0.00123)
        # Scoped counters
        self.assertEqual(lm.inference_cumulated_calls, 1)
        self.assertEqual(lm.inference_cumulated_prompt_tokens, 42)
        self.assertEqual(lm.inference_cumulated_completion_tokens, 17)
        self.assertEqual(lm.inference_cumulated_tokens, 59)
        self.assertAlmostEqual(lm.inference_cumulated_cost, 0.00123)
        # Other phases untouched
        self.assertEqual(lm.reward_cumulated_calls, 0)
        self.assertEqual(lm.optimizer_cumulated_calls, 0)
        # Last-call mirrors
        self.assertEqual(lm.last_call_prompt_tokens, 42)
        self.assertEqual(lm.last_call_completion_tokens, 17)
        self.assertEqual(lm.last_call_tokens, 59)
        self.assertGreater(lm.last_call_elapsed_s, 0.0)

    @patch("litellm.acompletion")
    async def test_lm_routes_each_scope(self, mock_completion):
        mock_completion.return_value = _lm_response(
            prompt_tokens=10, completion_tokens=5, cost=0.001
        )
        lm = LanguageModel(model="ollama/mistral")
        for scope, expected_phase in (
            ("inference", "inference"),
            ("reward", "reward"),
            ("optimizer", "optimizer"),
        ):
            _set_scope(scope)
            try:
                await lm(_chat_messages())
            finally:
                _set_scope(None)
            self.assertEqual(
                getattr(lm, f"{expected_phase}_cumulated_calls"),
                1,
                f"scope={scope} did not bump {expected_phase}_cumulated_calls",
            )
        # 3 calls total
        self.assertEqual(lm.cumulated_calls, 3)
        self.assertEqual(lm.inference_cumulated_calls, 1)
        self.assertEqual(lm.reward_cumulated_calls, 1)
        self.assertEqual(lm.optimizer_cumulated_calls, 1)

    @patch("litellm.acompletion")
    async def test_lm_no_scope_only_updates_alltime(self, mock_completion):
        mock_completion.return_value = _lm_response(prompt_tokens=8, completion_tokens=2)
        lm = LanguageModel(model="ollama/mistral")
        _set_scope(None)
        await lm(_chat_messages())
        self.assertEqual(lm.cumulated_calls, 1)
        self.assertEqual(lm.cumulated_prompt_tokens, 8)
        # No scoped counters should have moved.
        self.assertEqual(lm.inference_cumulated_calls, 0)
        self.assertEqual(lm.reward_cumulated_calls, 0)
        self.assertEqual(lm.optimizer_cumulated_calls, 0)

    @patch("litellm.acompletion")
    async def test_lm_missing_usage_field(self, mock_completion):
        """Some providers (e.g. Ollama, local stubs) may not return `usage`.
        Counters should still bump the call, with zero tokens.
        """
        resp = ModelResponse(
            id="x",
            model="ollama/mistral",
            choices=[
                Choices(
                    message=Message(content="hi", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
        )
        # Deliberately do NOT set resp.usage or resp._hidden_params.
        mock_completion.return_value = resp
        lm = LanguageModel(model="ollama/mistral")
        _set_scope("inference")
        try:
            await lm(_chat_messages())
        finally:
            _set_scope(None)
        self.assertEqual(lm.cumulated_calls, 1)
        self.assertEqual(lm.cumulated_prompt_tokens, 0)
        self.assertEqual(lm.cumulated_completion_tokens, 0)
        self.assertEqual(lm.cumulated_cost, 0.0)
        self.assertEqual(lm.inference_cumulated_calls, 1)
        self.assertEqual(lm.inference_cumulated_tokens, 0)

    @patch("litellm.acompletion")
    async def test_lm_missing_response_cost(self, mock_completion):
        """If _hidden_params lacks response_cost, tokens still populate but
        cost stays at 0.
        """
        resp = _lm_response(prompt_tokens=100, completion_tokens=50)
        resp._hidden_params = {}  # present but empty
        mock_completion.return_value = resp
        lm = LanguageModel(model="openai/gpt-4o-mini")
        _set_scope("inference")
        try:
            await lm(_chat_messages())
        finally:
            _set_scope(None)
        self.assertEqual(lm.cumulated_prompt_tokens, 100)
        self.assertEqual(lm.cumulated_cost, 0.0)
        self.assertEqual(lm.inference_cumulated_cost, 0.0)


class LMTier1CounterTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_cached_and_cache_creation_tokens(self, mock_completion):
        """Anthropic-style: prompt_tokens_details carries cached + creation."""
        resp = _lm_response(prompt_tokens=1000, completion_tokens=50, cost=0.001)
        resp.usage.prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=900,
            cache_creation_tokens=100,
        )
        mock_completion.return_value = resp
        lm = LanguageModel(model="anthropic/claude-3-5-sonnet")
        _set_scope("inference")
        try:
            await lm(_chat_messages())
        finally:
            _set_scope(None)
        self.assertEqual(lm.cumulated_cached_tokens, 900)
        self.assertEqual(lm.cumulated_cache_creation_tokens, 100)
        self.assertEqual(lm.inference_cumulated_cached_tokens, 900)
        self.assertEqual(lm.inference_cumulated_cache_creation_tokens, 100)
        # Other phases must not move.
        self.assertEqual(lm.reward_cumulated_cached_tokens, 0)
        self.assertEqual(lm.optimizer_cumulated_cached_tokens, 0)

    @patch("litellm.acompletion")
    async def test_reasoning_tokens(self, mock_completion):
        """OpenAI o-series / Claude thinking: completion_tokens_details carries
        reasoning_tokens.
        """
        resp = _lm_response(prompt_tokens=100, completion_tokens=2000)
        resp.usage.completion_tokens_details = CompletionTokensDetailsWrapper(
            reasoning_tokens=1800,
        )
        mock_completion.return_value = resp
        lm = LanguageModel(model="openai/o1-mini")
        _set_scope("inference")
        try:
            await lm(_chat_messages())
        finally:
            _set_scope(None)
        self.assertEqual(lm.cumulated_reasoning_tokens, 1800)
        self.assertEqual(lm.inference_cumulated_reasoning_tokens, 1800)

    @patch("litellm.acompletion")
    async def test_long_tail_details_dict(self, mock_completion):
        """Long-tail fields (multimodal split, tool use, overhead) land in
        the per-phase `details` dict rather than as flat counters.
        """
        resp = _lm_response(prompt_tokens=100, completion_tokens=50)
        resp.usage.prompt_tokens_details = PromptTokensDetailsWrapper(
            audio_tokens=20,
            image_tokens=30,
        )
        resp.usage.completion_tokens_details = CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=10,
        )
        resp.usage.server_tool_use = ServerToolUse(
            web_search_requests=2,
        )
        resp._hidden_params = {
            "response_cost": 0.0,
            "litellm_overhead_time_ms": 7.5,
        }
        mock_completion.return_value = resp
        lm = LanguageModel(model="openai/gpt-4o")
        _set_scope("inference")
        try:
            await lm(_chat_messages())
        finally:
            _set_scope(None)
        details = lm.inference_cumulated_details
        self.assertEqual(details["prompt_audio_tokens"], 20)
        self.assertEqual(details["prompt_image_tokens"], 30)
        self.assertEqual(details["completion_accepted_prediction_tokens"], 10)
        self.assertEqual(details["server_web_search_requests"], 2)
        self.assertEqual(details["litellm_overhead_time_ms"], 7.5)
        # All-time mirror.
        self.assertEqual(lm.cumulated_details["prompt_audio_tokens"], 20)

    @patch("litellm.acompletion")
    async def test_details_accumulates_across_calls(self, mock_completion):
        resp1 = _lm_response(prompt_tokens=100, completion_tokens=10)
        resp1.usage.prompt_tokens_details = PromptTokensDetailsWrapper(audio_tokens=5)
        resp2 = _lm_response(prompt_tokens=100, completion_tokens=10)
        resp2.usage.prompt_tokens_details = PromptTokensDetailsWrapper(audio_tokens=7)
        mock_completion.side_effect = [resp1, resp2]
        lm = LanguageModel(model="openai/gpt-4o")
        _set_scope("inference")
        try:
            await lm(_chat_messages())
            await lm(_chat_messages())
        finally:
            _set_scope(None)
        self.assertEqual(lm.inference_cumulated_details["prompt_audio_tokens"], 12)


class LMFileCacheTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_identical_call_served_from_disk(self, mock_completion):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")
        mock_completion.return_value = _lm_response(prompt_tokens=10, completion_tokens=5)
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)

        first = await lm(_chat_messages())
        second = await lm(_chat_messages())

        self.assertEqual(mock_completion.call_count, 1)
        self.assertEqual(first.get_json(), second.get_json())
        self.assertEqual(lm.cumulated_calls, 1)
        self.assertEqual(lm.cumulated_cache_hits, 1)

    @patch("litellm.acompletion")
    async def test_cache_persists_across_instances(self, mock_completion):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")
        mock_completion.return_value = _lm_response(prompt_tokens=10, completion_tokens=5)
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)
        first = await lm(_chat_messages())

        # A fresh instance (as in a new process) reuses the same files.
        lm2 = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)
        second = await lm2(_chat_messages())

        self.assertEqual(mock_completion.call_count, 1)
        self.assertEqual(first.get_json(), second.get_json())
        self.assertEqual(lm2.cumulated_cache_hits, 1)

    @patch("litellm.acompletion")
    async def test_different_prompt_is_a_miss(self, mock_completion):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")
        mock_completion.return_value = _lm_response(prompt_tokens=10, completion_tokens=5)
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)

        await lm(_chat_messages())
        await lm(ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="Bye")]))

        self.assertEqual(mock_completion.call_count, 2)
        self.assertEqual(lm.cumulated_cache_hits, 0)

    @patch("litellm.acompletion")
    async def test_structured_output_is_cached(self, mock_completion):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "Toulouse"}'}}]
        }
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)

        first = await lm(_chat_messages(), schema=_FinishReasonAnswer.get_schema())
        second = await lm(_chat_messages(), schema=_FinishReasonAnswer.get_schema())

        self.assertEqual(mock_completion.call_count, 1)
        self.assertEqual(first.get_json(), {"answer": "Toulouse"})
        self.assertEqual(second.get_json(), {"answer": "Toulouse"})
        self.assertEqual(second.get_schema(), first.get_schema())

    @patch("litellm.acompletion")
    async def test_drained_stream_is_cached_and_replayed(self, mock_completion):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)
        mock_completion.return_value = iter(
            [
                {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
            ]
        )
        stream = await lm(_chat_messages(), streaming=True)
        first = await stream.aconsume()
        self.assertEqual(first.get_json()["content"], "Hello")

        # Second identical call replays the drained stream from disk.
        replay = await lm(_chat_messages(), streaming=True)
        second = await replay.aconsume()

        self.assertEqual(mock_completion.call_count, 1)
        self.assertEqual(second.get_json()["content"], "Hello")
        self.assertEqual(second.get_json()["reasoning_content"], "hmm")
        self.assertEqual(lm.cumulated_cache_hits, 1)

    @patch("litellm.acompletion")
    async def test_abandoned_stream_is_not_cached(self, mock_completion):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)
        for _ in range(2):
            mock_completion.return_value = iter(
                [
                    {"choices": [{"delta": {"content": "Hel"}}]},
                    {"choices": [{"delta": {"content": "lo"}}]},
                ]
            )
            stream = await lm(_chat_messages(), streaming=True)
            # Consume only the first chunk, then abandon the stream.
            await stream.__anext__()
        self.assertEqual(mock_completion.call_count, 2)
        self.assertEqual(lm.cumulated_cache_hits, 0)

    @patch("litellm.acompletion")
    async def test_streamed_and_non_streamed_calls_share_entries(self, mock_completion):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)
        mock_completion.return_value = _lm_response(prompt_tokens=10, completion_tokens=5)
        first = await lm(_chat_messages())

        # The same request with streaming=True replays the cached message.
        replay = await lm(_chat_messages(), streaming=True)
        second = await replay.aconsume()

        self.assertEqual(mock_completion.call_count, 1)
        self.assertEqual(second.get_json()["content"], first.get_json()["content"])
        self.assertEqual(lm.cumulated_cache_hits, 1)

    def test_cache_dir_in_config_roundtrip(self):
        cache_dir = os.path.join(self.get_temp_dir(), "lm_cache")
        lm = LanguageModel(model="openai/gpt-4o", cache_dir=cache_dir)
        config = lm.get_config()
        self.assertEqual(config["cache_dir"], cache_dir)
        restored = LanguageModel.from_config(config)
        self.assertEqual(restored.cache_dir, cache_dir)
        self.assertIsNotNone(restored._file_cache)

    def test_credentials_are_not_serialized(self):
        lm = LanguageModel(
            model="openai/gpt-4o",
            api_key="sk-SECRET",
            aws_session_token="tok",
            extra_headers={"Authorization": "Bearer x"},
            temperature=0.2,
            max_tokens=16,
        )
        config = lm.get_config()
        self.assertNotIn("sk-SECRET", str(config))
        self.assertNotIn("aws_session_token", config)
        self.assertNotIn("extra_headers", config)
        self.assertEqual((config["temperature"], config["max_tokens"]), (0.2, 16))


class SupportedProvidersTest(testing.TestCase):
    def test_lists_every_provider_prefix(self):
        providers = LanguageModel.supported_providers()
        self.assertEqual(providers, sorted(providers))
        for provider in (
            "anthropic",
            "azure",
            "bedrock",
            "cohere",
            "deepseek",
            "doubleword",
            "gemini",
            "groq",
            "huggingface",
            "mirai",
            "mistral",
            "nebius",
            "ollama",
            "openai",
            "openrouter",
            "sglang",
            "together_ai",
            "vllm",
            "xai",
        ):
            self.assertIn(provider, providers)
        # Internal litellm aliases are not what users write in `model`.
        self.assertNotIn("ollama_chat", providers)
        self.assertNotIn("hosted_vllm", providers)

    def test_package_level_function_matches_classmethod(self):
        self.assertEqual(
            synalinks.language_models.supported_providers(),
            LanguageModel.supported_providers(),
        )

    def test_returns_a_fresh_list(self):
        providers = LanguageModel.supported_providers()
        providers.append("nope")
        self.assertNotIn("nope", LanguageModel.supported_providers())

    def test_nebius_is_a_provider_with_its_key_and_endpoint(self):
        info = LanguageModel.provider_info("nebius")
        self.assertEqual(
            [(v["name"], v["required"], v["secret"]) for v in info["env"]],
            [("NEBIUS_API_KEY", True, True), ("NEBIUS_API_BASE", False, False)],
        )
        self.assertEqual(info["lists_models_from"], "catalog")

    @patch("litellm.acompletion")
    async def test_nebius_structured_output_sends_a_strict_json_schema(
        self, mock_completion
    ):
        class Answer(DataModel):
            answer: str

        lm = LanguageModel(model="nebius/meta-llama/Llama-3.3-70B-Instruct")
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        result = await lm(_chat_messages(), schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})
        called = mock_completion.call_args.kwargs
        self.assertEqual(called["response_format"]["type"], "json_schema")
        self.assertTrue(called["response_format"]["json_schema"]["strict"])
        # Not routed through OpenRouter: no provider preferences are added.
        self.assertNotIn("provider", called)

    @patch("litellm.acompletion")
    async def test_mirai_routes_to_local_hosted_vllm(self, mock_completion):
        class Answer(DataModel):
            answer: str

        lm = LanguageModel(model="mirai/Qwen3.8-27B-M")
        self.assertEqual(lm.model, "hosted_vllm/Qwen3.8-27B-M")
        self.assertEqual(lm.api_base, "http://localhost:8000/v1")
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        result = await lm(_chat_messages(), schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})
        called = mock_completion.call_args.kwargs
        self.assertEqual(called["api_base"], "http://localhost:8000/v1")
        self.assertEqual(called["response_format"]["type"], "json_schema")
        self.assertEqual(
            called["response_format"]["json_schema"]["schema"], Answer.get_schema()
        )

    @patch.dict(os.environ, {}, clear=False)
    def test_vllm_defaults_to_the_v1_endpoint(self):
        os.environ.pop("HOSTED_VLLM_API_BASE", None)
        lm = LanguageModel(model="vllm/Qwen/Qwen3-8B")
        self.assertEqual(lm.api_base, "http://localhost:8000/v1")

    def test_mirai_keeps_an_explicit_api_base(self):
        lm = LanguageModel(
            model="mirai/Qwen3.8-27B-M", api_base="http://10.0.0.2:8080/v1"
        )
        self.assertEqual(lm.api_base, "http://10.0.0.2:8080/v1")

    @patch("litellm.acompletion")
    async def test_sglang_routes_to_local_hosted_vllm(self, mock_completion):
        class Answer(DataModel):
            answer: str

        lm = LanguageModel(model="sglang/Qwen/Qwen3-8B")
        self.assertEqual(lm.model, "hosted_vllm/Qwen/Qwen3-8B")
        self.assertEqual(lm.api_base, "http://localhost:30000/v1")
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        result = await lm(_chat_messages(), schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})
        called = mock_completion.call_args.kwargs
        self.assertEqual(called["model"], "hosted_vllm/Qwen/Qwen3-8B")
        self.assertEqual(called["api_base"], "http://localhost:30000/v1")
        self.assertEqual(called["response_format"]["type"], "json_schema")

    @patch.dict(os.environ, {"SGLANG_API_BASE": "http://gpu:30000/v1"})
    def test_sglang_reads_its_base_url_variable(self):
        self.assertEqual(
            LanguageModel(model="sglang/Qwen/Qwen3-8B").api_base, "http://gpu:30000/v1"
        )
        lm = LanguageModel(model="sglang/Qwen/Qwen3-8B", api_base="http://box:9000/v1")
        self.assertEqual(lm.api_base, "http://box:9000/v1")

    @patch("litellm.acompletion")
    async def test_huggingface_uses_native_json_schema(self, mock_completion):
        class Answer(DataModel):
            answer: str

        lm = LanguageModel(model="huggingface/together/Qwen/Qwen3-8B")
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        result = await lm(_chat_messages(), schema=Answer.get_schema())
        self.assertEqual(result.get_json(), {"answer": "B"})
        response_format = mock_completion.call_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "structured_output")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], Answer.get_schema())
        self.assertNotIn("tools", mock_completion.call_args.kwargs)

    async def test_unsupported_provider_error_lists_supported_ones(self):
        class Answer(DataModel):
            answer: str

        lm = LanguageModel(model="unknown_provider/some-model")
        with self.assertRaisesRegex(ValueError, "Supported providers: anthropic"):
            await lm(_chat_messages(), schema=Answer.get_schema())


class ToolWireFormatTest(testing.TestCase):
    """A wire-format tool declaration must be plain, self-contained data.

    A `Tool` builds its schema from its own attributes, so the module's
    `Tracker` has wrapped them as `TrackedDict`/`TrackedList`. Each wrapper
    points back at that tracker, and through it at every variable and submodule
    of the owning program. litellm's Anthropic handler deepcopies the optional
    params it is given, so a leaked wrapper made it copy the whole module graph
    — and the graph's cycles left the copied tracker half-built, failing every
    tool-calling request on that provider with `'Tracker' object has no
    attribute 'config'`.
    """

    @staticmethod
    def _tool():
        @synalinks.saving.register_synalinks_serializable()
        async def search(query: str, limit: int):
            """Search the index.

            Args:
                query (str): What to look for.
                limit (int): How many results to return.
            """
            return {"results": []}

        return Tool(func=search)

    def test_wire_declaration_uses_builtin_containers(self):
        wire = _tool_to_wire(self._tool())

        def assert_plain(value, path):
            if isinstance(value, dict):
                self.assertIs(type(value), dict, f"{path} is {type(value).__name__}")
                for key, item in value.items():
                    assert_plain(item, f"{path}.{key}")
            elif isinstance(value, list):
                self.assertIs(type(value), list, f"{path} is {type(value).__name__}")
                for index, item in enumerate(value):
                    assert_plain(item, f"{path}[{index}]")

        assert_plain(wire, "tool")

    def test_wire_declaration_holds_no_tracker_reference(self):
        """No node on the wire may be a handle on the module graph."""
        wire = _tool_to_wire(self._tool())
        stack = [("tool", wire)]
        while stack:
            path, value = stack.pop()
            self.assertFalse(
                hasattr(value, "tracker"),
                f"{path} carries a tracker reference into the provider payload",
            )
            if isinstance(value, dict):
                stack.extend((f"{path}.{k}", v) for k, v in value.items())
            elif isinstance(value, list):
                stack.extend((f"{path}[{i}]", v) for i, v in enumerate(value))

    def test_wire_declaration_preserves_the_schema(self):
        wire = _tool_to_wire(self._tool())
        parameters = wire["function"]["parameters"]
        self.assertEqual(wire["type"], "function")
        self.assertEqual(wire["function"]["name"], "search")
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(set(parameters["properties"]), {"query", "limit"})
        self.assertEqual(sorted(parameters["required"]), ["limit", "query"])

    async def test_fallback_receives_the_tools(self):
        calls = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            if kwargs["model"] == "openai/primary":
                raise RuntimeError("primary down")
            return {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ]
            }

        fallback = LanguageModel(model="openai/backup", retry=1)
        lm = LanguageModel(model="openai/primary", retry=1, fallback=fallback)
        messages = ChatMessages(messages=[ChatMessage(role="user", content="hi")])
        extra = {"type": "function", "function": {"name": "raw", "parameters": {}}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch("litellm.acompletion", side_effect=fake_acompletion):
                await lm(messages, tools=[self._tool()], tool_schemas=[extra])
        self.assertEqual(
            [[t["function"]["name"] for t in c["tools"]] for c in calls],
            [["search", "raw"], ["search", "raw"]],
        )


class _FinishReasonAnswer(DataModel):
    answer: str


class FinishReasonTest(testing.TestCase):
    """`finish_reason` is a choice field, so it is reported on the model
    (`last_call_finish_reason`) rather than on the returned message."""

    @staticmethod
    def _response(content="", finish_reason="stop"):
        return {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
        }

    @patch("litellm.acompletion")
    async def test_finish_reason_is_reported_on_the_model(self, mock_completion):
        mock_completion.return_value = self._response(content="hi", finish_reason="stop")
        lm = LanguageModel(model="ollama/mistral")
        response = await lm(_chat_messages())
        self.assertEqual(response.get("content"), "hi")
        self.assertEqual(lm.last_call_finish_reason, "stop")
        # The message stays exactly a chat-completion message.
        self.assertNotIn("finish_reason", response.get_json())

    @patch("litellm.acompletion")
    async def test_finish_reason_is_cleared_between_calls(self, mock_completion):
        lm = LanguageModel(model="ollama/mistral")
        mock_completion.return_value = self._response(
            content="hi", finish_reason="length"
        )
        await lm(_chat_messages())
        self.assertEqual(lm.last_call_finish_reason, "length")
        mock_completion.return_value = {"choices": [{"message": {"content": "hi"}}]}
        await lm(_chat_messages())
        self.assertIsNone(lm.last_call_finish_reason)

    @patch("litellm.acompletion")
    async def test_length_cut_is_reported_once_not_retried(self, mock_completion):
        # An identical retry reproduces a spent budget, so it costs
        # `retry` x `timeout` and buys nothing.
        mock_completion.return_value = self._response(finish_reason="length")
        lm = LanguageModel(model="ollama/mistral", retry=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = await lm(_chat_messages())
        self.assertEqual(mock_completion.call_count, 1)
        self.assertIsNone(result)
        self.assertEqual(lm.last_call_finish_reason, "length")

    @patch("litellm.acompletion")
    async def test_blocked_completion_is_reported_once_not_retried(self, mock_completion):
        mock_completion.return_value = self._response(finish_reason="content_filter")
        lm = LanguageModel(model="ollama/mistral", retry=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await lm(_chat_messages())
        self.assertEqual(mock_completion.call_count, 1)
        self.assertEqual(lm.last_call_finish_reason, "content_filter")

    @patch("litellm.acompletion")
    async def test_transient_empty_response_still_retries(self, mock_completion):
        # The case a retry does fix keeps its retries.
        mock_completion.return_value = self._response(finish_reason="stop")
        lm = LanguageModel(model="ollama/mistral", retry=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = await lm(_chat_messages())
        self.assertEqual(mock_completion.call_count, 3)
        self.assertIsNone(result)

    @patch("litellm.acompletion")
    async def test_empty_response_error_names_the_cause(self, mock_completion):
        mock_completion.return_value = self._response(finish_reason="length")
        lm = LanguageModel(model="ollama/mistral", retry=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await lm(_chat_messages(), schema=_FinishReasonAnswer.get_schema())
        messages = " ".join(str(w.message) for w in caught)
        self.assertIn("finish_reason='length'", messages)
        self.assertIn("completion token budget was exhausted", messages)
        self.assertIn("max_tokens", messages)


class CurrentCallUsageTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_usage_of_the_call_in_this_task(self, mock_completion):
        from synalinks.src.modules.language_models.language_model import (
            current_call_usage,
        )

        mock_completion.return_value = _lm_response(
            prompt_tokens=10, completion_tokens=4, cost=0.5
        )
        lm = LanguageModel(model="ollama/mistral")
        await lm(_chat_messages())
        usage = current_call_usage()
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 4)
        self.assertEqual(usage["total_tokens"], 14)
        self.assertEqual(usage["cost"], 0.5)
        self.assertIn("finish_reason", usage)

    @patch("litellm.acompletion")
    async def test_concurrent_calls_see_their_own_usage(self, mock_completion):
        import asyncio

        from synalinks.src.modules.language_models.language_model import (
            current_call_usage,
        )

        responses = [
            _lm_response(prompt_tokens=1, completion_tokens=1),
            _lm_response(prompt_tokens=2, completion_tokens=2),
            _lm_response(prompt_tokens=3, completion_tokens=3),
        ]
        mock_completion.side_effect = responses
        lm = LanguageModel(model="ollama/mistral")

        async def one():
            await lm(_chat_messages())
            return current_call_usage()["input_tokens"]

        seen = await asyncio.gather(one(), one(), one())
        self.assertEqual(sorted(seen), [1, 2, 3])


class CapabilitiesTest(testing.TestCase):
    async def test_flags_decide_the_capabilities(self):
        lm = LanguageModel(
            model="deepseek/deepseek-chat",
            supports_reasoning=True,
            supports_vision=True,
            supports_audio=True,
        )
        self.assertTrue(await lm.supports_reasoning())
        self.assertTrue(await lm.supports_vision())
        self.assertTrue(await lm.supports_audio())

        off = LanguageModel(model="gemini/gemini-2.5-flash", supports_audio=False)
        self.assertFalse(await off.supports_audio())

    async def test_flags_survive_serialization(self):
        lm = LanguageModel(
            model="vllm/Qwen/Qwen3-8B",
            api_base="http://x/v1",
            supports_reasoning=True,
            supports_vision=False,
        )
        restored = LanguageModel.from_config(lm.get_config())
        self.assertTrue(await restored.supports_reasoning())
        self.assertFalse(await restored.supports_vision())
        self.assertFalse(await restored.supports_audio())

    async def test_capabilities_are_off_unless_flagged(self):
        lm = LanguageModel(model="openai/gpt-4o")
        self.assertFalse(await lm.supports_reasoning())
        self.assertFalse(await lm.supports_vision())
        self.assertFalse(await lm.supports_audio())

    def test_supports_softprompt_only_on_vllm(self):
        self.assertTrue(LanguageModel(model="vllm/qwen").supports_softprompt())
        self.assertFalse(LanguageModel(model="openai/gpt-4o-mini").supports_softprompt())
        self.assertFalse(LanguageModel(model="ollama/mistral").supports_softprompt())


class ToolResultImagesTest(testing.TestCase):
    """Where the images a tool returned end up in the provider request."""

    def _messages(self):
        return ChatMessages(
            messages=[
                ChatMessage(role=ChatRole.USER, content="plot it"),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run_python_code", "arguments": {}},
                        }
                    ],
                ),
                ChatMessage(
                    role=ChatRole.TOOL,
                    tool_call_id="c1",
                    content=[
                        '{"images": ["<image 1>"]}',
                        Image(data="QUJD", mime_type="image/png"),
                    ],
                ),
            ]
        )

    @patch("litellm.acompletion")
    async def test_anthropic_keeps_images_in_the_tool_result(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="anthropic/claude-sonnet-4-5", supports_vision=True)(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[2]["content"][1]["type"], "image_url")

    @patch("litellm.acompletion")
    async def test_gemini_3_keeps_images_in_the_tool_result(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="gemini/gemini-3-flash-preview", supports_vision=True)(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[2]["content"][1]["type"], "image_url")

    @patch("litellm.acompletion")
    async def test_gemini_2_moves_images_to_a_user_message(self, mock_completion):
        # Multimodal function responses are Gemini 3 only.
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="gemini/gemini-2.5-flash", supports_vision=True)(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 4)
        self.assertEqual(sent[3]["content"][1]["type"], "image_url")

    @patch.object(LanguageModel, "supports_vision", return_value=True)
    @patch("litellm.acompletion")
    async def test_openrouter_claude_moves_images_to_a_user_message(
        self, mock_completion, _
    ):
        # OpenRouter speaks the OpenAI format, whose tool messages are text
        # only, even when the model behind it is Claude.
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="openrouter/anthropic/claude-sonnet-4.5")(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 4)
        self.assertIsInstance(sent[2]["content"], str)
        self.assertIn("1 image(s) attached in the next message", sent[2]["content"])
        self.assertEqual(sent[3]["role"], "user")
        self.assertEqual(
            sent[3]["content"][1]["image_url"]["url"], "data:image/png;base64,QUJD"
        )

    @patch("litellm.acompletion")
    async def test_openai_moves_images_to_a_user_message(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="openai/gpt-4o-mini", supports_vision=True)(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 4)
        self.assertIsInstance(sent[2]["content"], str)
        self.assertIn("attached in the next message", sent[2]["content"])
        self.assertEqual(sent[3]["role"], "user")
        self.assertEqual(
            sent[3]["content"][1]["image_url"]["url"], "data:image/png;base64,QUJD"
        )

    @patch("litellm.acompletion")
    async def test_text_only_model_gets_a_note_instead(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="deepseek/deepseek-chat")(self._messages())
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 3)
        self.assertIn("omitted", sent[2]["content"])


class ToolResultAudioTest(testing.TestCase):
    """Audio a tool returned always travels in a follow-up user message."""

    def _messages(self):
        return ChatMessages(
            messages=[
                ChatMessage(role=ChatRole.USER, content="listen"),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read_audio", "arguments": {}},
                        }
                    ],
                ),
                ChatMessage(
                    role=ChatRole.TOOL,
                    tool_call_id="c1",
                    content=['{"audio": "<audio 1>"}', Audio(data="QUJD", format="wav")],
                ),
            ]
        )

    @patch("litellm.acompletion")
    async def test_gemini_gets_the_audio_in_a_user_message(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="gemini/gemini-2.5-flash", supports_audio=True)(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 4)
        self.assertIn("1 audio clip(s) attached", sent[2]["content"])
        self.assertEqual(sent[3]["role"], "user")
        self.assertEqual(sent[3]["content"][1]["input_audio"]["data"], "QUJD")

    @patch("litellm.acompletion")
    async def test_images_stay_in_the_tool_result_next_to_dropped_audio(
        self, mock_completion
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = self._messages()
        messages.messages[2].content.append(
            Image(data="QUJD", mime_type="image/png").to_content_part()
        )
        await LanguageModel(model="anthropic/claude-sonnet-4-5", supports_vision=True)(
            messages
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 3)
        text, image = sent[2]["content"]
        self.assertIn("1 audio clip(s) omitted", text["text"])
        self.assertEqual(image["type"], "image_url")

    # litellm's model table, fetched at import, comes and goes on OpenRouter's
    # models: what is tested is the message, not the table.
    @patch.object(LanguageModel, "supports_audio", return_value=True)
    @patch("litellm.acompletion")
    async def test_openrouter_gemini_gets_the_audio_in_a_user_message(
        self, mock_completion, _
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="openrouter/google/gemini-2.5-flash")(self._messages())
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 4)
        self.assertIn("1 audio clip(s) attached", sent[2]["content"])
        self.assertEqual(sent[3]["role"], "user")
        self.assertEqual(
            sent[3]["content"][1],
            {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
        )

    @patch.object(LanguageModel, "supports_audio", return_value=False)
    @patch("litellm.acompletion")
    async def test_openrouter_claude_gets_an_audio_note(self, mock_completion, _):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="openrouter/anthropic/claude-sonnet-4.5")(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 3)
        self.assertIn("1 audio clip(s) omitted", sent[2]["content"])

    @patch("litellm.acompletion")
    async def test_model_without_audio_gets_a_note(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        await LanguageModel(model="anthropic/claude-sonnet-4-5", supports_vision=True)(
            self._messages()
        )
        sent = mock_completion.call_args.kwargs["messages"]
        self.assertEqual(len(sent), 3)
        self.assertIn("1 audio clip(s) omitted", sent[2]["content"])


class _EngineStub(BaseHTTPRequestHandler):
    """A self-hosted engine: metadata routes, streamed and plain chat replies."""

    model_info = None
    server_info = None
    models = None
    capabilities = None
    reasoning = False

    def _reply(self, status, payload=None, sse=None):
        data = (
            "".join(f"data: {json.dumps(c)}\n\n" for c in sse) + "data: [DONE]\n\n"
            if sse is not None
            else json.dumps(payload)
        ).encode()
        self.send_response(status)
        self.send_header(
            "Content-Type", "text/event-stream" if sse else "application/json"
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.server.paths.append(self.path)
        if self.path.startswith("/model_info") and self.model_info is not None:
            return self._reply(200, self.model_info)
        if self.path.startswith("/server_info") and self.server_info is not None:
            return self._reply(200, self.server_info)
        if self.path == "/api/v1/models" and self.models is not None:
            return self._reply(200, {"data": self.models})
        self._reply(404, {"detail": "Not Found"})

    def do_POST(self):
        self.server.paths.append(self.path)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            return self._reply(200, {"capabilities": self.capabilities or []})
        if body.get("stream"):
            self.server.probes.append(body)
            key = "reasoning_content" if self.reasoning else "content"
            delta = {"role": "assistant", key: "Okay"}
            chunk = {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            return self._reply(200, sse=[chunk])
        self.server.requests.append(body)
        schema = (body.get("response_format") or {}).get("json_schema") or {}
        fields = (schema.get("schema") or {}).get("properties") or {}
        answer = {"answer": "Paris"}
        if "thinking" in fields:
            answer = {"thinking": "written by the model", **answer}
        message = {"role": "assistant", "content": json.dumps(answer)}
        if self.reasoning:
            message["reasoning_content"] = "native trace"
        self._reply(
            200,
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def log_message(self, *args):
        pass


class _Thought(DataModel):
    thinking: str
    answer: str


class CapabilityFlagsTest(testing.TestCase):
    """The constructor flags decide reasoning; the server is never asked."""

    def _serve(self, **behavior):
        handler = type("Handler", (_EngineStub,), behavior)
        server = HTTPServer(("127.0.0.1", 0), handler)
        server.paths, server.probes, server.requests = [], [], []
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        self.addCleanup(server.shutdown)
        return server, f"http://127.0.0.1:{server.server_port}"

    async def _ask(self, lm, effort="low"):
        messages = ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="?")])
        result = await lm(messages, schema=_Thought.get_schema(), reasoning_effort=effort)
        return result.get_json()

    def _schema_fields(self, sent):
        return sent["response_format"]["json_schema"]["schema"]["properties"]

    async def test_vllm_flagged_as_reasoner_forwards_the_effort(self):
        server, base = self._serve(reasoning=True)
        lm = LanguageModel(
            model="vllm/Qwen/Qwen3-8B", api_base=f"{base}/v1", supports_reasoning=True
        )

        result = await self._ask(lm)

        self.assertEqual(server.probes, [])
        self.assertEqual(server.paths, ["/v1/chat/completions"])
        sent = server.requests[0]
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertEqual(sent["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("thinking", self._schema_fields(sent))
        self.assertEqual(result, {"thinking": "native trace", "answer": "Paris"})

    async def test_unflagged_self_hosted_model_keeps_thinking(self):
        server, base = self._serve(reasoning=True)
        for model in ("vllm/Qwen/Qwen3-8B", "sglang/Qwen/Qwen3-8B"):
            await self._ask(LanguageModel(model=model, api_base=f"{base}/v1"))
        await self._ask(
            LanguageModel(
                model="doubleword/Qwen/Qwen3-235B-A22B",
                api_base=f"{base}/v1",
                api_key="k",
            )
        )

        self.assertEqual(server.probes, [])
        self.assertEqual(set(server.paths), {"/v1/chat/completions"})
        for sent in server.requests:
            self.assertNotIn("reasoning_effort", sent)
            self.assertNotIn("chat_template_kwargs", sent)
            self.assertIn("thinking", self._schema_fields(sent))

    async def test_disable_turns_the_chat_template_switch_off(self):
        server, base = self._serve(reasoning=True)
        lm = LanguageModel(model="vllm/Qwen/Qwen3-8B", api_base=f"{base}/v1")

        await self._ask(lm, effort="none")

        self.assertEqual(server.probes, [])
        self.assertEqual(
            server.requests[0]["chat_template_kwargs"], {"enable_thinking": False}
        )

    async def test_missing_trace_leaves_thinking_empty(self):
        _, base = self._serve(reasoning=False)
        lm = LanguageModel(
            model="sglang/Qwen/Qwen3-8B", api_base=f"{base}/v1", supports_reasoning=True
        )

        result = await self._ask(lm)

        self.assertEqual(result, {"thinking": "", "answer": "Paris"})

    async def test_openrouter_flagged_reasoner_fills_thinking_from_the_trace(self):
        server, base = self._serve(reasoning=True)
        lm = LanguageModel(
            model="openrouter/qwen/qwen3.8-27b",
            api_base=f"{base}/api/v1",
            supports_reasoning=True,
        )

        result = await self._ask(lm)

        self.assertNotIn("/api/v1/models", server.paths)
        sent = server.requests[0]
        self.assertNotIn("thinking", self._schema_fields(sent))
        self.assertNotIn("chat_template_kwargs", sent)
        self.assertEqual(result, {"thinking": "native trace", "answer": "Paris"})

    async def test_openrouter_disable_turns_reasoning_off_for_a_reasoner(self):
        server, base = self._serve()
        api_base = f"{base}/api/v1"
        thinking = LanguageModel(
            model="openrouter/qwen/qwen3.8-27b",
            api_base=api_base,
            supports_reasoning=True,
        )
        plain = LanguageModel(
            model="openrouter/openai/gpt-4o-mini",
            api_base=api_base,
            supports_reasoning=False,
        )

        await self._ask(thinking, effort="none")
        await self._ask(plain, effort="none")

        self.assertEqual(server.requests[0]["reasoning"], {"enabled": False})
        self.assertNotIn("reasoning_effort", server.requests[0])
        self.assertNotIn("reasoning", server.requests[1])

    async def test_ollama_is_never_asked_its_capabilities(self):
        server, base = self._serve(capabilities=["completion", "thinking", "vision"])
        lm = LanguageModel(model="ollama/qwen3:8b", api_base=base)

        self.assertFalse(await lm.supports_reasoning())
        self.assertFalse(await lm.supports_vision())
        self.assertEqual(server.paths, [])


class ProviderPrefixRewriteTest(testing.TestCase):
    def test_only_the_provider_prefix_is_rewritten(self):
        vllm = LanguageModel(model="vllm/RedHatAI/Qwen3-8B-vllm-FP8", api_base="http://x")
        ollama = LanguageModel(model="ollama/me/ollama-tuned")

        self.assertEqual(vllm.model, "hosted_vllm/RedHatAI/Qwen3-8B-vllm-FP8")
        self.assertEqual(ollama.model, "ollama_chat/me/ollama-tuned")


class _BuiltAnswer(DataModel):
    answer: str


_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a word up.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
}


def _question():
    return ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="Pick one.")])


def _sent(mock_completion):
    """What `call()` handed to litellm, without the transport settings."""
    kwargs = dict(mock_completion.call_args.kwargs)
    kwargs.pop("timeout")
    kwargs.pop("caching")
    return kwargs


class BuildRequestTest(testing.TestCase):
    """`build_request` returns what a call would send, without sending it."""

    @patch("litellm.acompletion")
    async def test_build_request_is_what_call_sends_for_an_openrouter_structured_call(
        self, mock_completion
    ):
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        lm = LanguageModel(model="openrouter/acme/plain-1")
        schema = _BuiltAnswer.get_schema()
        built = await lm.build_request(_question(), schema=schema)
        await lm(_question(), schema=schema)
        self.assertEqual(built, _sent(mock_completion))
        self.assertEqual(
            built["provider"],
            {"require_parameters": True, "ignore": ["coreweave", "nebius"]},
        )
        self.assertTrue(built["response_format"]["json_schema"]["strict"])

    @patch("litellm.acompletion")
    async def test_build_request_is_what_call_sends_for_an_openrouter_tool_call(
        self, mock_completion
    ):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        lm = LanguageModel(model="openrouter/acme/plain-1")
        built = await lm.build_request(_question(), tool_schemas=[_LOOKUP_TOOL])
        await lm(_question(), tool_schemas=[_LOOKUP_TOOL])
        self.assertEqual(built, _sent(mock_completion))
        self.assertIn("tools", built)
        self.assertNotIn("response_format", built)
        self.assertNotIn("ignore", built.get("provider") or {})

    @patch("litellm.acompletion")
    async def test_build_request_forwards_a_level_only_to_a_model_declared_as_reasoning(
        self, mock_completion
    ):
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        schema = _BuiltAnswer.get_schema()
        thinker = LanguageModel(
            model="openrouter/acme/thinker-1", supports_reasoning=True
        )
        built = await thinker.build_request(
            _question(), schema=schema, reasoning_effort="high"
        )
        await thinker(_question(), schema=schema, reasoning_effort="high")
        self.assertEqual(built, _sent(mock_completion))
        self.assertEqual(built["reasoning_effort"], "high")
        plain = LanguageModel(model="openrouter/acme/plain-1")
        built = await plain.build_request(
            _question(), schema=schema, reasoning_effort="high"
        )
        self.assertNotIn("reasoning_effort", built)

    @patch("litellm.acompletion")
    async def test_build_request_keeps_another_providers_own_shape(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer": "B"}'}}]
        }
        lm = LanguageModel(model="openai/gpt-4o-mini")
        schema = _BuiltAnswer.get_schema()
        built = await lm.build_request(_question(), schema=schema)
        await lm(_question(), schema=schema)
        self.assertEqual(built, _sent(mock_completion))
        self.assertNotIn("provider", built)

    async def test_build_request_neither_fetches_media_nor_changes_its_inputs(self):
        lm = LanguageModel(model="openrouter/acme/plain-1", provider={"order": ["acme"]})
        defaults = copy.deepcopy(lm.default_kwargs)
        per_call = {"temperature": 0.0}
        sent_per_call = copy.deepcopy(per_call)
        messages = ChatMessages.model_validate_json(
            '{"messages":[{"role":"user","content":['
            '{"type":"text","text":"caption"},'
            '{"type":"image_url","image_url":{"url":"https://example.invalid/x.png"}}]}]}'
        )
        lm.last_call_finish_reason = "stop"
        with patch(
            "synalinks.src.modules.language_models.language_model.resolve_content_media",
            side_effect=AssertionError("build_request fetched media"),
        ):
            built = await lm.build_request(
                messages, schema=_BuiltAnswer.get_schema(), **per_call
            )
        part = built["messages"][0]["content"][1]
        self.assertEqual(part["image_url"]["url"], "https://example.invalid/x.png")
        self.assertEqual(lm.default_kwargs, defaults)
        self.assertEqual(per_call, sent_per_call)
        self.assertNotIn("stream", built)
        self.assertEqual(lm.last_call_finish_reason, "stop")

    async def test_build_request_refuses_a_schema_with_tools(self):
        lm = LanguageModel(model="openrouter/acme/plain-1")
        with self.assertRaises(ValueError):
            await lm.build_request(
                _question(),
                schema=_BuiltAnswer.get_schema(),
                tool_schemas=[_LOOKUP_TOOL],
            )

    async def test_build_request_works_on_an_oauth_language_model(self):
        from synalinks.src.modules.language_models import OAuthLanguageModel

        lm = OAuthLanguageModel(model="codex/gpt-6.1-sol")
        built = await lm.build_request(_question())
        self.assertEqual(built["model"], "codex/gpt-6.1-sol")
        self.assertEqual(built["messages"][0]["content"], "Pick one.")

    async def test_build_request_returns_a_request_of_its_own(self):
        lm = LanguageModel(model="openrouter/acme/plain-1", provider={"order": ["acme"]})
        tool = copy.deepcopy(_LOOKUP_TOOL)
        built = await lm.build_request(_question(), tool_schemas=[tool])
        built["provider"]["ignore"] = ["nebius"]
        built["tools"][0]["function"]["name"] = "changed"
        self.assertEqual(lm.default_kwargs["provider"], {"order": ["acme"]})
        self.assertEqual(tool, _LOOKUP_TOOL)

    async def test_build_request_keeps_a_setting_that_cannot_be_copied(self):
        lock = threading.Lock()
        lm = LanguageModel(model="openai/gpt-4o-mini", client=lock)
        built = await lm.build_request(_question())
        self.assertIs(built["client"], lock)

    async def test_build_request_has_nothing_to_send_without_messages(self):
        lm = LanguageModel(model="openrouter/acme/plain-1")
        self.assertIsNone(await lm.build_request(None))

    async def test_build_request_of_an_empty_conversation_sends_no_message(self):
        lm = LanguageModel(model="anthropic/claude-x")
        built = await lm.build_request(ChatMessages(messages=[]))
        self.assertEqual(built["messages"], [])

    @patch("litellm.acompletion")
    async def test_build_request_streams_as_a_call_does(self, mock_completion):
        mock_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        lm = LanguageModel(model="openai/gpt-4o-mini")
        built = await lm.build_request(_question(), streaming=True)
        await lm(_question(), streaming=True)
        self.assertEqual(built, _sent(mock_completion))
        self.assertTrue(built["stream"])
        self.assertNotIn("streaming", built)
        structured = await lm.build_request(
            _question(), schema=_BuiltAnswer.get_schema(), streaming=True
        )
        self.assertNotIn("stream", structured)


class _ThoughtAnswer(DataModel):
    thinking: str
    answer: str


class FallbackSchemaTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_fallback_receives_the_callers_schema(self, mock_completion):
        mock_completion.side_effect = [
            Exception("primary down"),
            {"choices": [{"message": {"content": '{"thinking": "t", "answer": "B"}'}}]},
        ]
        lm = LanguageModel(
            model="openrouter/acme/thinker-1",
            supports_reasoning=True,
            retry=1,
            retry_max_wait=0,
            fallback=LanguageModel(model="openai/gpt-4o-mini"),
        )
        schema = _ThoughtAnswer.get_schema()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await lm(_question(), schema=schema, reasoning_effort="high")
        primary, fallback = (c.kwargs for c in mock_completion.call_args_list)
        primary_schema = primary["response_format"]["json_schema"]["schema"]
        fallback_schema = fallback["response_format"]["json_schema"]["schema"]
        self.assertNotIn("thinking", primary_schema["properties"])
        self.assertIn("thinking", fallback_schema["properties"])
        self.assertIn("thinking", fallback_schema["required"])
