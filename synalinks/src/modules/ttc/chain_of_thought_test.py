# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.programs.program import Program


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

    async def test_cot_schema_always_has_thinking(self):
        """ChainOfThought's output schema always carries `thinking` so
        callers get a consistent shape regardless of reasoning_effort."""

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")

        for effort in ("low", "none"):
            cot = ChainOfThought(
                data_model=Answer,
                language_model=language_model,
                reasoning_effort=effort,
            )
            gen_schema = cot.generator.schema
            self.assertIn("thinking", gen_schema["properties"])
            self.assertIn("answer", gen_schema["properties"])

    @patch("litellm.supports_reasoning")
    @patch("litellm.acompletion")
    async def test_thinking_stripped_from_lm_call_and_reinjected(
        self, mock_completion, mock_supports_reasoning
    ):
        """When reasoning is enabled and supported, the LM call schema has
        `thinking` removed (saving tokens) and the field is filled back in
        from `reasoning_content` after the call."""
        mock_supports_reasoning.return_value = True

        class Query(DataModel):
            query: str = Field(description="The user query")

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="anthropic/claude-3-5-sonnet")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            reasoning_effort="low",
        )(x0)

        program = Program(inputs=x0, outputs=x1, name="cot_strip_reinject")

        # LM only returns the user fields; no `thinking` generated.
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"answer": "Paris"}),
                        "reasoning_content": "France's capital is Paris.",
                    }
                }
            ]
        }

        result = await program(Query(query="What is the capital of France?"))
        result_json = result.get_json()
        self.assertEqual(result_json["thinking"], "France's capital is Paris.")
        self.assertEqual(result_json["answer"], "Paris")

        # The schema actually sent to the LM had `thinking` stripped.
        sent_schema = mock_completion.call_args.kwargs["response_format"][
            "json_schema"
        ]["schema"]
        self.assertIn("answer", sent_schema["properties"])
        self.assertNotIn("thinking", sent_schema["properties"])

    @patch("litellm.supports_reasoning")
    @patch("litellm.acompletion")
    async def test_thinking_kept_in_lm_call_when_model_lacks_reasoning_support(
        self, mock_completion, mock_supports_reasoning
    ):
        """If the model doesn't support reasoning, the LM call schema must
        keep `thinking` so the model can fill it via prompting."""
        mock_supports_reasoning.return_value = False

        class Query(DataModel):
            query: str = Field(description="The user query")

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            reasoning_effort="low",
        )(x0)

        program = Program(inputs=x0, outputs=x1, name="cot_no_reasoning_support")

        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"thinking": "prompted trace", "answer": "Paris"}
                        )
                    }
                }
            ]
        }

        result = await program(Query(query="What is the capital of France?"))
        result_json = result.get_json()
        self.assertEqual(result_json["thinking"], "prompted trace")
        self.assertEqual(result_json["answer"], "Paris")

        sent_schema = mock_completion.call_args.kwargs["response_format"][
            "json_schema"
        ]["schema"]
        self.assertIn("thinking", sent_schema["properties"])

    @patch("litellm.supports_reasoning")
    @patch("litellm.acompletion")
    async def test_user_schema_with_thinking_overridden_by_reasoning_content(
        self, mock_completion, mock_supports_reasoning
    ):
        """If a user explicitly defines a data_model with a `thinking`
        field, the LanguageModel overrides it from `reasoning_content`."""
        mock_supports_reasoning.return_value = True

        class Query(DataModel):
            query: str = Field(description="The user query")

        class AnswerWithThinking(DataModel):
            thinking: str = Field(description="The reasoning trace")
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=AnswerWithThinking,
            language_model=language_model,
            reasoning_effort="low",
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="cot_user_thinking_override",
        )

        structured = {"thinking": "placeholder", "answer": "Paris"}
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(structured),
                        "reasoning_content": "France's capital is Paris.",
                    }
                }
            ]
        }

        result = await program(Query(query="What is the capital of France?"))
        result_json = result.get_json()
        self.assertEqual(result_json["thinking"], "France's capital is Paris.")
        self.assertEqual(result_json["answer"], "Paris")

    @patch("litellm.supports_reasoning")
    @patch("litellm.acompletion")
    async def test_chain_of_thought_without_data_model(
        self, mock_completion, mock_supports_reasoning
    ):
        """ChainOfThought with data_model=None returns a ChatMessage and
        populates the thinking field from the LM's reasoning_content."""
        mock_supports_reasoning.return_value = True

        class Query(DataModel):
            query: str = Field(description="The user query")

        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=None,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="cot_no_schema",
        )

        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Paris",
                        "reasoning_content": "France's capital is Paris.",
                    }
                }
            ]
        }

        result = await program(Query(query="What is the capital of France?"))

        result_json = result.get_json()
        self.assertEqual(result_json["content"], "Paris")
        self.assertEqual(result_json["thinking"], "France's capital is Paris.")
        self.assertEqual(result_json["role"], "assistant")

    @patch("litellm.acompletion")
    async def test_chain_of_thought_streaming_without_data_model(
        self, mock_completion
    ):
        """ChainOfThought(data_model=None, streaming=True) streams the LM
        response as a StreamingIterator. Reasoning chunks (delta has
        `reasoning_content` only, empty content) are exposed as `thinking`
        chunks rather than ending the stream prematurely."""

        class Query(DataModel):
            query: str = Field(description="The user query")

        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await ChainOfThought(
            data_model=None,
            language_model=language_model,
            streaming=True,
        )(x0)

        program = Program(inputs=x0, outputs=x1, name="cot_streaming")

        mock_completion.return_value = iter(
            [
                {"choices": [{"delta": {"reasoning_content": "Capital "}}]},
                {"choices": [{"delta": {"reasoning_content": "of France."}}]},
                {"choices": [{"delta": {"content": "Paris"}}]},
                {"choices": [{"delta": {"content": "."}}]},
            ]
        )

        stream = await program(Query(query="What is the capital of France?"))
        thinking_parts = []
        content_parts = []
        async for chunk in stream:
            if "thinking" in chunk:
                thinking_parts.append(chunk["thinking"])
            if "content" in chunk:
                content_parts.append(chunk["content"])

        self.assertEqual("".join(thinking_parts), "Capital of France.")
        self.assertEqual("".join(content_parts), "Paris.")

    async def test_streaming_disabled_when_schema_provided(self):
        """Setting streaming=True alongside a structured schema is silently
        downgraded — structured output requires the full response."""

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")
        cot = ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            streaming=True,
        )
        self.assertFalse(cot.streaming)
        self.assertFalse(cot.generator.streaming)

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
