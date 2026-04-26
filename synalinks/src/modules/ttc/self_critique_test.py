# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.core.input_module import Input
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
