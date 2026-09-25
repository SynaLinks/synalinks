# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules import Input
from synalinks.src.modules.core.decision import Decision
from synalinks.src.modules.decision_models import DecisionModel
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.testing.test_utils import mock_decision_model


class DecisionTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_decision(self, mock_completion):
        class Query(DataModel):
            query: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        expected_string = (
            """{"thinking": "The question ask for the capital of France, """
            """the answer is straitforward as it is well known that Paris is the"""
            """ capital", "choice": "easy"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await Decision(
            question="What is the difficulty level of the above provided query?",
            labels=["easy", "difficult", "unkown"],
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(Query(query="What is the French capital?"))

        self.assertEqual(result.get_json(), json.loads(expected_string))


class DecisionConfigTest(testing.TestCase):
    def test_config_round_trip(self):
        decision = Decision(
            question="Easy?",
            labels=["easy", "difficult"],
            seed_instructions=["Pick the difficulty."],
            language_model=LanguageModel(model="ollama/mistral"),
        )
        restored = Decision.from_config(decision.get_config())
        self.assertEqual(restored.seed_instructions, ["Pick the difficulty."])
        self.assertEqual(restored.decision.seed_instructions, ["Pick the difficulty."])


@patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"})
class DecisionWithDecisionModelTest(testing.TestCase):
    async def test_decision_model_decision(self):
        class Query(DataModel):
            query: str

        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(
            decision_model,
            {
                "choice": {
                    "type": "choice",
                    "choice": "easy",
                    "probabilities": {"easy": 0.9, "difficult": 0.1},
                    "confidence": 0.8,
                }
            },
        )

        x0 = Input(data_model=Query)
        x1 = await Decision(
            question="What is the difficulty level of the query?",
            labels=["easy", "difficult"],
            decision_model=decision_model,
        )(x0)
        program = Program(inputs=x0, outputs=x1)

        result = await program(Query(query="What is the French capital?"))

        self.assertEqual(result.get_json(), {"choice": "easy"})
        self.assertNotIn("thinking", x1.get_schema()["properties"])
        restored = Decision.from_config(program.get_module(index=1).get_config())
        self.assertIsInstance(restored.decision_model, DecisionModel)
        self.assertEqual(restored.schema, program.get_module(index=1).schema)
        self.assertEqual(
            payloads[0]["questions"],
            {
                "choice": {
                    "type": "choice",
                    "instructions": "What is the difficulty level of the query?",
                    "criteria": {"easy": None, "difficult": None},
                }
            },
        )
