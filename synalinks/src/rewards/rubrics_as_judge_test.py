# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Rating
from synalinks.src.backend import Rating20
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.rewards.rubrics_as_judge import Rubric
from synalinks.src.rewards.rubrics_as_judge import RubricsAsJudge
from synalinks.src.rewards.rubrics_as_judge import RubricsAsJudgeProgram


class RubricsAsJudgeTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_rubrics_as_judge_weighted_reward(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[
                {
                    "name": "correct",
                    "description": "The answer states the right value.",
                    "weight": 3,
                },
                {
                    "name": "grounded",
                    "description": "Every claim comes from the result set.",
                },
            ],
            score_type=Rating,
        )

        schema = reward.program.generator.schema
        self.assertEqual(schema["required"], ["critique", "correct", "grounded"])
        self.assertEqual(schema["properties"]["correct"]["type"], "integer")
        self.assertEqual(schema["properties"]["correct"]["enum"], [1, 2, 3, 4, 5])

        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"critique": "Correct but weakly grounded.", '
                            '"correct": 5, "grounded": 3}'
                        )
                    }
                }
            ]
        }
        score = await reward(
            y_true=Answer(answer="Paris"), y_pred=Answer(answer="Paris")
        )
        # correct: (5 - 1) / 4 = 1.0, grounded: (3 - 1) / 4 = 0.5
        self.assertAlmostEqual(score, (3 * 1.0 + 1 * 0.5) / 4)

    @patch("litellm.acompletion")
    async def test_rubrics_as_judge_empty_prediction(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[{"name": "correct", "description": "Correct."}],
        )
        self.assertIs(reward.program.score_type, Rating20)
        score = await reward(y_true=None, y_pred=None)
        self.assertEqual(score, 0.0)
        mock_completion.assert_not_called()

    def test_rubric_validation(self):
        rubric = Rubric.parse("No preamble, no restating the question.")
        self.assertEqual(rubric.name, "no_preamble_no_restating_the_question")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            RubricsAsJudge(
                language_model=LanguageModel(model="ollama/mistral"), rubrics=[]
            )
        with self.assertRaisesRegex(ValueError, "Duplicate rubric names"):
            RubricsAsJudge(
                language_model=LanguageModel(model="ollama/mistral"),
                rubrics=[
                    {"name": "A B", "description": "one"},
                    {"name": "a-b", "description": "two"},
                ],
            )
        with self.assertRaisesRegex(ValueError, "weights must be positive"):
            Rubric("correct", "Correct.", weight=0)

    def test_rubrics_as_judge_preset(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(language_model=language_model, rubrics="faithfulness")
        self.assertEqual(
            [rubric.name for rubric in reward.program.rubrics],
            ["faithfulness"],
        )

    def test_rubrics_as_judge_config_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[{"name": "correct", "description": "Correct.", "weight": 2}],
            score_type=Rating,
        )
        config = reward.program.get_config()
        self.assertEqual(config["score_type"], "Rating")
        program = RubricsAsJudgeProgram.from_config(config)
        self.assertIs(program.score_type, Rating)
        self.assertEqual(program.rubrics[0].weight, 2.0)

    def test_rubric_parse(self):
        # Names are normalized into JSON property names.
        self.assertEqual(Rubric("Is It Correct?", "Correct.").name, "is_it_correct")
        self.assertEqual(Rubric("2nd opinion", "Opinion.").name, "criterion_2nd_opinion")
        rubric = Rubric.parse({"name": "correct", "description": "Correct.", "weight": 2})
        self.assertIs(Rubric.parse(rubric), rubric)
        self.assertEqual(Rubric.from_config(rubric.get_config()).weight, 2.0)
        with self.assertRaisesRegex(ValueError, "missing \\['description'\\]"):
            Rubric.parse({"name": "correct"})
        with self.assertRaisesRegex(TypeError, "Cannot interpret"):
            Rubric.parse(42)

    @patch("litellm.acompletion")
    async def test_rubrics_as_judge_without_gold_reference(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[{"name": "correct", "description": "Correct."}],
            score_type=Rating,
        )

        mock_completion.return_value = {
            "choices": [
                {"message": {"content": '{"critique": "Plausible.", "correct": 5}'}}
            ]
        }
        score = await reward(y_true=None, y_pred=Answer(answer="Paris"))
        self.assertEqual(score, 1.0)
        # Without a reference the judge only sees the prediction: no `gold_` key.
        prompt = json.dumps(mock_completion.call_args.kwargs["messages"])
        self.assertNotIn("gold_", prompt)
        self.assertIn("Paris", prompt)

    @patch("litellm.acompletion")
    async def test_rubrics_as_judge_incomplete_grade_sheet(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[
                {"name": "correct", "description": "Correct."},
                {"name": "grounded", "description": "Grounded."},
            ],
            score_type=Rating,
        )

        # A criterion the judge skipped is not silently averaged away.
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "Half.", "correct": 5}'}}]
        }
        score = await reward(y_true=Answer(answer="Paris"), y_pred=Answer(answer="Paris"))
        self.assertEqual(score, 0.0)

    @patch("litellm.acompletion")
    async def test_rubrics_as_judge_no_grade_sheet(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[{"name": "correct", "description": "Correct."}],
        )

        # The provider returns no content: score 0.0 without raising.
        mock_completion.return_value = {"choices": [{"message": {"content": None}}]}
        score = await reward(y_true=Answer(answer="Paris"), y_pred=Answer(answer="Paris"))
        self.assertEqual(score, 0.0)

    async def test_rubrics_as_judge_malformed_inputs(self):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[{"name": "correct", "description": "Correct."}],
        )

        with self.assertRaisesRegex(ValueError, "list or tuple"):
            await reward.program(Answer(answer="Paris"))
        with self.assertRaisesRegex(ValueError, "length of 2"):
            await reward.program([Answer(answer="Paris")])

    async def test_rubrics_as_judge_symbolic_output_schema(self):
        class Answer(DataModel):
            answer: str = Field(description="The answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[{"name": "correct", "description": "Correct."}],
            score_type=Rating,
        )

        outputs = await reward.program(
            [
                Answer(answer="Paris").to_symbolic_data_model(),
                Answer(answer="Paris").to_symbolic_data_model(),
            ]
        )
        self.assertTrue(is_symbolic_data_model(outputs))
        # The grade sheet plus the combined 0..1 reward.
        properties = outputs.get_schema()["properties"]
        self.assertIn("critique", properties)
        self.assertIn("correct", properties)
        self.assertEqual(properties["reward"]["minimum"], 0.0)
        self.assertEqual(properties["reward"]["maximum"], 1.0)

    def test_rubrics_as_judge_registry_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RubricsAsJudge(
            language_model=language_model,
            rubrics=[{"name": "correct", "description": "Correct.", "weight": 2}],
            score_type=Rating,
            in_mask=["answer"],
            name="my_judge",
        )

        restored = rewards.deserialize(rewards.serialize(reward))
        self.assertIsInstance(restored, RubricsAsJudge)
        self.assertEqual(restored.name, "my_judge")
        self.assertEqual(restored.in_mask, ["answer"])
        self.assertIs(restored.program.score_type, Rating)
        self.assertEqual(restored.program.rubrics[0].name, "correct")
        self.assertEqual(restored.program.rubrics[0].weight, 2.0)
