# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import modules
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import standardize_schema
from synalinks.src.modules import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Sequential


class SequentialTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_flow_with_input(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama_chat/deepseek-r1",
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

        program = Sequential(
            name="chain_of_thought",
            description="Useful to answer in a step by step manner",
        )
        program.add(Input(data_model=Query))
        program.add(
            modules.Generator(
                data_model=AnswerWithRationale,
                language_model=language_model,
            )
        )

        self.assertEqual(len(program.modules), 2)
        self.assertTrue(program.built)
        self.assertEqual(len(program.variables), 1)

        # Test eager call
        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )
        self.assertEqual(result.get_json(), json.loads(expected_string))

        # Test symbolic call
        x = SymbolicDataModel(data_model=Query)
        y = await program(x)
        self.assertIsInstance(y, SymbolicDataModel)
        self.assertEqual(
            y.get_schema(), standardize_schema(AnswerWithRationale.get_schema())
        )

        # Test `modules` constructor arg
        program = Sequential(
            modules=[
                Input(data_model=Query),
                modules.Generator(
                    data_model=AnswerWithRationale,
                    language_model=language_model,
                ),
            ],
            name="chain_of_thought",
            description="Useful to answer in a step by step manner",
        )
        self.assertEqual(len(program.modules), 2)
        self.assertTrue(program.built)
        self.assertEqual(len(program.variables), 1)

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )
        self.assertEqual(result.get_json(), json.loads(expected_string))

        # Test pop
        program.pop()
        self.assertEqual(len(program.modules), 1)
        self.assertFalse(program.built)
        self.assertEqual(len(program.variables), 0)

    def test_representation(self):
        program = Sequential(
            name="chain_of_thought",
            description="Useful to answer in a step by step manner",
        )
        self.assertEqual(
            str(program),
            "<Sequential name=chain_of_thought, "
            "description='Useful to answer in a step by step manner', "
            "built=False>",
        )
