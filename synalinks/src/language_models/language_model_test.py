# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel


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
        self.assertEqual(result, ChatMessage(**result).get_json())
        self.assertEqual(result, expected.get_json())

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
        self.assertEqual(result, AnswerWithRationale(**result).get_json())
        self.assertEqual(result, expected.get_json())

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
        for msg in response:
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
        self.assertEqual(result, expected.get_json())
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

        self.assertEqual(result, {"answer": "4"})
        response_format = mock_completion.call_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            response_format["json_schema"]["name"], "structured_output"
        )
