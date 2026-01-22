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


class ChainOfThoughtModuleTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_chain_of_thought(self, mock_completion):
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
