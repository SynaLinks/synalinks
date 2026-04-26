# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from enum import Enum
from typing import List
from typing import Literal
from typing import Union

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import is_schema_equal
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_tool_calls
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_tool_choice
from synalinks.src.utils.tool_utils import Tool


async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
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


async def thinking(thinking: str):
    """Think about something.

    Args:
        thinking (str): Your step by step thinking.
    """
    return {
        "thinking": thinking,
    }


class DynamicEnumTest(testing.TestCase):
    def test_basic_dynamic_enum(self):
        class DecisionAnswer(DataModel):
            thinking: str
            choice: str

        class Choice(str, Enum):
            easy = "easy"
            difficult = "difficult"
            unknown = "unknown"

        class Decision(DataModel):
            thinking: str
            choice: Choice

        labels = ["easy", "difficult", "unkown"]

        schema = dynamic_enum(DecisionAnswer.get_schema(), "choice", labels)

        self.assertTrue(is_schema_equal(Decision.get_schema(), schema))


class DynamicToolCallsSchemaTest(testing.TestCase):
    def test_dynamic_tool_call_schema(self):
        class Calculate(DataModel):
            """Calculate the result of a mathematical expression."""

            tool_name: Literal["calculate"]
            expression: str = Field(
                description=(
                    "The mathematical expression to calculate, such as "
                    "'2 + 2'. The expression can contain numbers, operators (+, -, *, /),"
                    " parentheses, and spaces."
                )
            )

        class Thinking(DataModel):
            """Think about something."""

            tool_name: Literal["thinking"]
            thinking: str = Field(description="Your step by step thinking.")

        class ToolCalls(DataModel):
            tool_calls: List[Union[Calculate, Thinking]]

        expected_schema = ToolCalls.get_schema()

        tools = [
            Tool(calculate),
            Tool(thinking),
        ]

        dynamic_schema = dynamic_tool_calls(tools=tools)

        print("Expected:")
        print(ToolCalls.prettify_schema())
        print("Generated:")
        import json

        print(json.dumps(dynamic_schema, indent=2))

        self.assertEqual(expected_schema, dynamic_schema)


class DynamicToolChoiceSchemaTest(testing.TestCase):
    def test_dynamic_tool_call_schema(self):
        class Calculate(DataModel):
            """Calculate the result of a mathematical expression."""

            tool_name: Literal["calculate"]
            expression: str = Field(
                description=(
                    "The mathematical expression to calculate, such as "
                    "'2 + 2'. The expression can contain numbers, operators (+, -, *, /),"
                    " parentheses, and spaces."
                )
            )

        class Thinking(DataModel):
            """Think about something."""

            tool_name: Literal["thinking"]
            thinking: str = Field(description="Your step by step thinking.")

        class ToolChoice(DataModel):
            tool_choice: Union[Calculate, Thinking]

        expected_schema = ToolChoice.get_schema()

        tools = [
            Tool(calculate),
            Tool(thinking),
        ]

        dynamic_schema = dynamic_tool_choice(tools=tools)
        print("Expected:")
        print(ToolChoice.prettify_schema())
        print("Generated:")
        import json

        print(json.dumps(dynamic_schema, indent=2))

        self.assertEqual(expected_schema, dynamic_schema)
