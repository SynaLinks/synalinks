# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.programs.program import Program


class MockMessage:
    """Mock litellm Message that supports both dict and attribute access."""

    def __init__(self, content, reasoning_content=None):
        self._data = {"content": content}
        self.reasoning_content = reasoning_content

    def __getitem__(self, key):
        return self._data.get(key)


class ChainOfThoughtModuleTest(testing.TestCase):
    async def test_default_reasoning_effort_is_low(self):
        """Test that ChainOfThought defaults to 'low' reasoning effort."""

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")

        cot = ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            name="test_cot",
        )

        self.assertEqual(cot.reasoning_effort, "low")

    async def test_explicit_reasoning_effort_none(self):
        """Test that reasoning_effort can be explicitly set to 'none'."""

        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")

        cot = ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            reasoning_effort="none",
            name="test_cot",
        )

        self.assertEqual(cot.reasoning_effort, "none")

    @patch("litellm.supports_reasoning")
    @patch("litellm.acompletion")
    async def test_thinking_from_reasoning_content(
        self, mock_completion, mock_supports_reasoning
    ):
        """Test thinking field is populated from reasoning model's reasoning_content."""
        mock_supports_reasoning.return_value = True

        class Query(DataModel):
            query: str = Field(description="The user query")

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="openai/gpt-5.2")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            reasoning_effort="low",
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="test_reasoning",
            description="Test reasoning content",
        )

        # Mock message with reasoning_content attribute
        mock_message = MockMessage(
            content='{"answer": "Paris"}',
            reasoning_content="Let me think step by step about this question.",
        )

        mock_completion.return_value = {"choices": [{"message": mock_message}]}

        result = await program(Query(query="What is the capital of France?"))

        # Verify the mock was called
        mock_completion.assert_called_once()
        mock_supports_reasoning.assert_called()

        # Verify thinking field is populated from reasoning_content
        result_json = result.get_json()
        self.assertIn("thinking", result_json)
        self.assertIn("answer", result_json)
        self.assertEqual(
            result_json["thinking"], "Let me think step by step about this question."
        )
        self.assertEqual(result_json["answer"], "Paris")

    @patch("litellm.supports_reasoning")
    @patch("litellm.acompletion")
    async def test_thinking_empty_when_no_reasoning_content(
        self, mock_completion, mock_supports_reasoning
    ):
        """Test that thinking field is empty string when reasoning_content is None."""
        mock_supports_reasoning.return_value = True

        class Query(DataModel):
            query: str = Field(description="The user query")

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="openai/gpt-5.2")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            reasoning_effort="low",
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="test_no_reasoning",
            description="Test without reasoning content",
        )

        # Mock message without reasoning_content
        mock_message = MockMessage(
            content='{"answer": "Paris"}',
            reasoning_content=None,
        )

        mock_completion.return_value = {"choices": [{"message": mock_message}]}

        result = await program(Query(query="What is the capital of France?"))

        # Verify the mock was called
        mock_completion.assert_called_once()
        mock_supports_reasoning.assert_called()

        # Verify thinking field is empty string (not missing)
        result_json = result.get_json()
        self.assertEqual(result_json["thinking"], "")
        self.assertEqual(result_json["answer"], "Paris")

    @patch("litellm.supports_reasoning")
    @patch("litellm.acompletion")
    async def test_thinking_generated_when_model_not_supports_reasoning(
        self, mock_completion, mock_supports_reasoning
    ):
        """Test thinking is generated by model when it doesn't support reasoning."""
        mock_supports_reasoning.return_value = False

        class Query(DataModel):
            query: str = Field(description="The user query")

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="openai/gpt-4o")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            reasoning_effort="low",
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="test_no_reasoning_support",
            description="Test model without reasoning support",
        )

        # When model doesn't support reasoning, thinking stays in schema
        # and model generates it as part of the JSON output
        expected_response = {
            "thinking": "The capital of France is a well-known fact.",
            "answer": "Paris",
        }

        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(expected_response)}}]
        }

        result = await program(Query(query="What is the capital of France?"))

        result_json = result.get_json()
        self.assertEqual(
            result_json["thinking"], "The capital of France is a well-known fact."
        )
        self.assertEqual(result_json["answer"], "Paris")

    @patch("litellm.acompletion")
    async def test_chain_of_thought_with_reasoning_effort_none(self, mock_completion):
        """Test ChainOfThought with reasoning_effort='none' (thinking in schema)."""

        class Query(DataModel):
            query: str = Field(
                description="The user query",
            )

        class Answer(DataModel):
            answer: str = Field(
                description="The correct answer",
            )

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            reasoning_effort="none",
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_tought_qa",
            description="Answer the user query step by step",
        )

        expected_string = (
            """{"thinking": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        result = await program(
            Query(
                query="What is the French city of aerospace and robotics?",
            )
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))
