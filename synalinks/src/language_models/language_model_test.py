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
    async def test_retry_succeeds_on_second_attempt(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral", retry=3)

        messages = ChatMessages(
            messages=[
                ChatMessage(role=ChatRole.USER, content="Hello")
            ]
        )

        mock_completion.side_effect = [
            Exception("Temporary failure"),
            {
                "choices": [
                    {"message": {"content": "Hello, how can I help?"}}
                ]
            },
        ]

        result = await language_model(messages)
        self.assertEqual(result["content"], "Hello, how can I help?")
        self.assertEqual(mock_completion.call_count, 2)

    @patch("litellm.acompletion")
    async def test_retry_exhausted_returns_none(self, mock_completion):
        language_model = LanguageModel(
            model="ollama/mistral", retry=2
        )

        messages = ChatMessages(
            messages=[
                ChatMessage(role=ChatRole.USER, content="Hello")
            ]
        )

        mock_completion.side_effect = Exception("Persistent failure")

        result = await language_model(messages)
        self.assertIsNone(result)
        self.assertEqual(mock_completion.call_count, 2)

    @patch("litellm.acompletion")
    async def test_retry_exhausted_uses_fallback(self, mock_completion):
        fallback_model = LanguageModel(model="ollama/llama3")
        language_model = LanguageModel(
            model="ollama/mistral", retry=2, fallback=fallback_model
        )

        messages = ChatMessages(
            messages=[
                ChatMessage(role=ChatRole.USER, content="Hello")
            ]
        )

        # First 2 calls fail (primary), third succeeds (fallback)
        mock_completion.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            {
                "choices": [
                    {"message": {"content": "Fallback response"}}
                ]
            },
        ]

        result = await language_model(messages)
        self.assertEqual(result["content"], "Fallback response")
        self.assertEqual(mock_completion.call_count, 3)

    @patch("litellm.acompletion")
    async def test_retry_with_structured_output(self, mock_completion):
        language_model = LanguageModel(
            model="ollama/mistral", retry=3
        )

        messages = ChatMessages(
            messages=[
                ChatMessage(
                    role=ChatRole.USER,
                    content="What is 1+1?",
                )
            ]
        )

        class Answer(DataModel):
            answer: str

        mock_completion.side_effect = [
            Exception("Temporary failure"),
            {
                "choices": [
                    {"message": {"content": '{"answer": "2"}'}}
                ]
            },
        ]

        result = await language_model(
            messages, schema=Answer.get_schema()
        )
        self.assertEqual(result, {"answer": "2"})
        self.assertEqual(mock_completion.call_count, 2)
