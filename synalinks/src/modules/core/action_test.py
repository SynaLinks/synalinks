# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

import synalinks
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.core.action import Action
from synalinks.src.modules.core.tool import Tool
from synalinks.src.programs import Program


class ActionModuleTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_action(self, mock_completion):
        class Query(DataModel):
            query: str

        @synalinks.saving.register_synalinks_serializable()
        async def calculate(expression: str):
            """Calculate the result of a mathematical expression.

            Args:
                expression (str): The mathematical expression to calculate, such as
                    '2 + 2'. The expression can contain numbers, operators
                    (+, -, *, /), parentheses and spaces.
            """
            if not all(char in "0123456789+-*/(). " for char in expression):
                return {
                    "result": None,
                    "log": "Error: invalid characters in expression",
                }
            try:
                # Evaluate the mathematical expression safely
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                return {
                    "result": result,
                    "log": "Successfully executed",
                }
            except Exception as e:
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        language_model = LanguageModel("ollama_chat/mistral")

        expected_string = """{"expression": "12 + 15"}"""

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await Action(
            tool=Tool(calculate),
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="calculator",
            description=(
                "This program perform the calculation of a mathematical expression"
            ),
        )

        result = await program(
            Query(
                query=(
                    "You have a basket with 12 apples. "
                    "Your friend gives you 15 more apples. "
                    "How many apples do you have in total now?"
                )
            )
        )

        expected_json = {
            "action": {
                "inputs": {
                    "expression": "12 + 15",
                },
                "outputs": {
                    "result": 27.00,
                    "log": "Successfully executed",
                },
            }
        }

        self.assertEqual(result.get_json(), expected_json)
