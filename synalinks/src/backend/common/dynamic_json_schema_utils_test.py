# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from enum import Enum
from typing import List
from typing import Literal
from typing import Union

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import is_schema_equal
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum_array
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


class DynamicEnumArrayTest(testing.TestCase):
    def test_basic_dynamic_enum_array(self):
        class DecisionAnswer(DataModel):
            thinking: str
            choices: List[str]

        class Choices(str, Enum):
            easy = "easy"
            difficult = "difficult"
            unknown = "unknown"

        class Decision(DataModel):
            thinking: str
            choices: List[Choices]

        labels = ["easy", "difficult", "unknown"]

        schema = dynamic_enum_array(DecisionAnswer.get_schema(), "choices", labels)

        self.assertTrue(is_schema_equal(Decision.get_schema(), schema))

    def test_dynamic_enum_array_in_defs(self):
        class Group(DataModel):
            name: str
            tags: List[str]

        class Parent(DataModel):
            group: Group

        labels = ["alpha", "beta", "gamma"]

        schema = dynamic_enum_array(Parent.get_schema(), "tags", labels)

        # The ``tags`` property lives under $defs.Group.properties and
        # should have been patched there, not at the top level.
        group_def = schema["$defs"]["Group"]
        self.assertEqual(group_def["properties"]["tags"]["type"], "array")
        self.assertEqual(
            group_def["properties"]["tags"]["items"],
            {"$ref": "#/$defs/Tags"},
        )
        # The enum $def carries the labels verbatim.
        self.assertEqual(schema["$defs"]["Tags"]["enum"], labels)
        self.assertEqual(schema["$defs"]["Tags"]["type"], "string")
        # Top-level properties must not have gained a stray ``tags`` entry.
        self.assertNotIn("tags", schema.get("properties", {}))

    def test_dynamic_enum_array_description(self):
        class DecisionAnswer(DataModel):
            thinking: str
            choices: List[str]

        labels = ["easy", "difficult", "unknown"]
        description = "The allowed difficulty labels."

        schema = dynamic_enum_array(
            DecisionAnswer.get_schema(),
            "choices",
            labels,
            description=description,
        )

        self.assertEqual(schema["$defs"]["Choices"]["description"], description)
        self.assertEqual(schema["$defs"]["Choices"]["enum"], labels)

    def test_dynamic_enum_array_empty_labels(self):
        class DecisionAnswer(DataModel):
            thinking: str
            choices: List[str]

        schema = dynamic_enum_array(DecisionAnswer.get_schema(), "choices", [])

        self.assertEqual(schema["$defs"]["Choices"]["enum"], [])
        self.assertEqual(
            schema["properties"]["choices"]["items"],
            {"$ref": "#/$defs/Choices"},
        )
        self.assertEqual(schema["properties"]["choices"]["type"], "array")


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
