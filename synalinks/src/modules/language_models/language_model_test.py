# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from litellm.types.utils import Choices
from litellm.types.utils import CompletionTokensDetailsWrapper
from litellm.types.utils import Message
from litellm.types.utils import ModelResponse
from litellm.types.utils import PromptTokensDetailsWrapper
from litellm.types.utils import ServerToolUse
from litellm.types.utils import Usage

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.backend.common.op_scope import _OP_SCOPE
from synalinks.src.modules.language_models import LanguageModel


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

        result = await language_model(messages, schema=Answer.get_schema())

        self.assertEqual(result.get_json(), {"answer": "4"})
        response_format = mock_completion.call_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "structured_output")

    @patch("litellm.acompletion")
    async def test_reasoning_effort_disable_sends_think_false_for_ollama(
        self, mock_completion
    ):
        """`reasoning_effort="disable"` turns native thinking off on ollama
        (which reasons by default) by sending `think=False`; `"none"` leaves the
        model at its default and sends nothing.
        """
        mock_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        ollama_lm = LanguageModel(model="ollama_chat/qwen3:8b")
        await ollama_lm(messages, reasoning_effort="disable")
        self.assertIs(mock_completion.call_args.kwargs.get("think"), False)

        await ollama_lm(messages, reasoning_effort="none")
        self.assertNotIn("think", mock_completion.call_args.kwargs)

    @patch("litellm.acompletion")
    async def test_reasoning_effort_disable_noop_for_non_ollama(self, mock_completion):
        """Opt-in providers reason only when enabled, so "disable" sends no
        `think` flag (it is ollama-specific and would be rejected elsewhere).
        """
        mock_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        messages = ChatMessages(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )

        lm = LanguageModel(model="openai/gpt-4o-mini")
        await lm(messages, reasoning_effort="disable")
        self.assertNotIn("think", mock_completion.call_args.kwargs)


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
    counters correctly. Guards against drift in LiteLLM's response schema —
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
