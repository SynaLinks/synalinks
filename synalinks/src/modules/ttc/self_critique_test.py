# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import FineScore
from synalinks.src.backend import Rating
from synalinks.src.backend import Rating20
from synalinks.src.backend import Score
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.modules.ttc.self_critique import SelfCritique
from synalinks.src.programs.program import Program


class SelfCritiqueModuleTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_self_critique(self, mock_completion):
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
            return_inputs=True,
        )(x0)
        x2 = await SelfCritique(
            language_model=language_model,
        )(x1)

        program = Program(
            inputs=x0,
            outputs=x2,
            name="answer_with_cot_and_self_critique",
            description="Useful to answer accurately",
        )

        expected_answer = (
            """{"thinking": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        expected_critique = (
            """{"critique": "The response provided by the model is accurate and"""
            """ well-structured. The information about Toulouse's contribution"""
            """ to aerospace and robotics is also relevant. However, consider """
            """adding more conversational tone or humanizing the output slightly"""
            """ for better user experience.", "reward": 0.9}"""
        )

        mock_responses = [
            {"choices": [{"message": {"content": expected_answer}}]},
            {"choices": [{"message": {"content": expected_critique}}]},
        ]

        mock_completion.side_effect = mock_responses

        result = await program(
            Query(
                query="What is the French city of aerospace and robotics?",
            )
        )

        expected_string = (
            """{"query": "What is the French city of aerospace and robotics?","""
            """ "thinking": "Toulouse hosts numerous research institutions and """
            """universities that specialize in aerospace engineering and robotics,"""
            """ such as the Institut Supérieur de l'Aéronautique et de l'Espace """
            """(ISAE-SUPAERO) and the French National Centre for Scientific Research """
            """(CNRS)", "answer": "Toulouse", "critique": "The response provided by """
            """the model is accurate and well-structured. The information about """
            """Toulouse's contribution to aerospace and robotics is also relevant. """
            """However, consider adding more conversational tone or humanizing the """
            """output slightly for better user experience.", "reward": 0.9}"""
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    @patch("litellm.acompletion")
    async def test_self_critique_with_rating_score_type(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")

        critique = SelfCritique(
            language_model=language_model,
            score_type=Rating,
        )
        # The generator asks the LM for the native 1..5 integer scale.
        reward_schema = critique.generator.schema["properties"]["reward"]
        self.assertEqual(reward_schema["type"], "integer")
        self.assertEqual(reward_schema["enum"], [1, 2, 3, 4, 5])
        self.assertNotIn("$defs", critique.generator.schema)

        # Symbolic output spec: `reward` is a normalized 0..1 float.
        x0 = Input(data_model=Answer)
        x1 = await critique(x0)
        symbolic_reward = x1.get_schema()["properties"]["reward"]
        self.assertEqual(symbolic_reward["type"], "number")
        self.assertEqual(symbolic_reward["minimum"], 0.0)
        self.assertEqual(symbolic_reward["maximum"], 1.0)
        self.assertNotIn("$defs", x1.get_schema())

        mock_completion.return_value = {
            "choices": [
                {"message": {"content": '{"critique": "Mostly right.", "reward": 4}'}}
            ]
        }
        result = await critique(Answer(answer="Toulouse"))
        self.assertEqual(result.get("critique"), "Mostly right.")
        # 4 on a 1..5 scale is normalized to (4 - 1) / (5 - 1) = 0.75
        self.assertAlmostEqual(result.get("reward"), 0.75)
        self.assertEqual(result.get_schema()["properties"]["reward"]["type"], "number")

    @patch("litellm.acompletion")
    async def test_self_critique_with_fine_score_type_name(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        # The score type can also be given by name.
        critique = SelfCritique(
            language_model=language_model,
            score_type="FineScore",
            return_inputs=False,
        )
        self.assertIs(critique.score_type, FineScore)
        self.assertEqual(
            critique.generator.schema["properties"]["reward"]["enum"][:3],
            [0.0, 0.05, 0.1],
        )
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "ok", "reward": 0.85}'}}]
        }
        result = await critique(Answer(answer="x"))
        self.assertAlmostEqual(result.get("reward"), 0.85)

    def test_self_critique_score_type_config_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        critique = SelfCritique(language_model=language_model, score_type=Rating20)
        config = critique.get_config()
        self.assertEqual(config["score_type"], "Rating20")
        clone = SelfCritique.from_config(config)
        self.assertIs(clone.score_type, Rating20)
        self.assertEqual(
            clone.generator.schema["properties"]["reward"]["enum"],
            list(range(1, 21)),
        )

    def test_self_critique_default_score_type(self):
        language_model = LanguageModel(model="ollama/mistral")
        critique = SelfCritique(language_model=language_model)
        self.assertIs(critique.score_type, Score)
        self.assertEqual(critique.get_config()["score_type"], "Score")

    def test_self_critique_default_instructions_spell_out_scale(self):
        language_model = LanguageModel(model="ollama/mistral")
        # Default 0.0..1.0 `Score`.
        critique = SelfCritique(language_model=language_model)
        self.assertIn("a float between 0.0 and 1.0", critique.instructions)
        self.assertIn("1.0 very good", critique.instructions)
        self.assertEqual(critique.generator.instructions, critique.instructions)
        # Integer Likert-style scale: the instructions must name its bounds.
        critique = SelfCritique(language_model=language_model, score_type=Rating)
        self.assertIn("an integer between 1 and 5", critique.instructions)
        self.assertIn("5 very good", critique.instructions)
        # Without a reward only the critique is requested.
        critique = SelfCritique(language_model=language_model, return_reward=False)
        self.assertNotIn("reward", critique.instructions)
        # User-provided instructions always win.
        critique = SelfCritique(
            language_model=language_model,
            score_type=Rating,
            instructions="Grade harshly.",
        )
        self.assertEqual(critique.instructions, "Grade harshly.")
        self.assertEqual(critique.generator.instructions, "Grade harshly.")

    def test_self_critique_invalid_score_type(self):
        language_model = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            SelfCritique(language_model=language_model, score_type="NotAScale")
        with self.assertRaises(ValueError):
            SelfCritique(language_model=language_model, score_type=str)

    @patch("litellm.acompletion")
    async def test_self_critique_eager_first_call_traces_symbolically(
        self, mock_completion
    ):
        # Regression: the first eager call of an unbuilt module used to trace
        # `call()` on the concrete inputs, costing one real LM request whose
        # result was discarded. The build must be symbolic: one LM call total.
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        critique = SelfCritique(language_model=language_model, return_inputs=False)
        self.assertFalse(critique.built)
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "ok", "reward": 0.9}'}}]
        }
        result = await critique(Answer(answer="Toulouse"))
        self.assertTrue(critique.built)
        self.assertEqual(mock_completion.call_count, 1)
        self.assertAlmostEqual(result.get("reward"), 0.9)
