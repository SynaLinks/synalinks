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

        # inline=False matches the Pydantic-generated $defs/$ref layout.
        schema = dynamic_enum(
            DecisionAnswer.get_schema(), "choice", labels, inline=False
        )

        self.assertTrue(is_schema_equal(Decision.get_schema(), schema))

    def test_inline_dynamic_enum(self):
        """With inline=True, the enum is written directly into the
        property rather than placed under $defs with a $ref."""

        class DecisionAnswer(DataModel):
            thinking: str
            choice: str

        labels = ["easy", "difficult", "unknown"]

        schema = dynamic_enum(
            DecisionAnswer.get_schema(),
            "choice",
            labels,
            inline=True,
        )

        choice = schema["properties"]["choice"]
        self.assertEqual(choice["enum"], labels)
        self.assertEqual(choice["type"], "string")
        # No indirection through $defs.
        self.assertNotIn("$ref", choice)
        self.assertNotIn("Choice", schema.get("$defs", {}))


class DynamicToolCallsSchemaTest(testing.TestCase):
    def test_dynamic_tool_call_schema(self):
        """Default (inline=True) embeds per-tool sub-schemas directly in
        `anyOf` and pins `tool_name` with both an items-level enum and a
        per-branch const — strict enough for backends that don't honor
        `const` inside `anyOf` (e.g. Gemini)."""
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]

        dynamic_schema = dynamic_tool_calls(tools=tools)

        self.assertNotIn("$defs", dynamic_schema)
        items = dynamic_schema["properties"]["tool_calls"]["items"]
        self.assertEqual(
            sorted(items["properties"]["tool_name"]["enum"]),
            ["calculate", "thinking"],
        )
        self.assertIn("tool_name", items["required"])
        const_values = []
        for branch in items["anyOf"]:
            self.assertNotIn("$ref", branch)
            self.assertIn("properties", branch)
            const_values.append(branch["properties"]["tool_name"]["const"])
        self.assertEqual(sorted(const_values), ["calculate", "thinking"])

    def test_dynamic_tool_calls_schema_with_refs(self):
        """inline=False keeps the older $defs/$ref layout for callers that
        want to share definitions."""
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]

        dynamic_schema = dynamic_tool_calls(tools=tools, inline=False)

        self.assertEqual(
            dynamic_schema["$defs"]["Calculate"]["properties"]["tool_name"]["const"],
            "calculate",
        )
        items = dynamic_schema["properties"]["tool_calls"]["items"]
        ref_targets = sorted(b["$ref"] for b in items["anyOf"])
        self.assertEqual(
            ref_targets,
            ["#/$defs/Calculate", "#/$defs/Thinking"],
        )


class DynamicToolChoiceSchemaTest(testing.TestCase):
    def test_dynamic_tool_call_schema(self):
        """Default (inline=True) embeds per-tool sub-schemas directly."""
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]

        dynamic_schema = dynamic_tool_choice(tools=tools)

        self.assertNotIn("$defs", dynamic_schema)
        choice = dynamic_schema["properties"]["tool_choice"]
        self.assertEqual(
            sorted(choice["properties"]["tool_name"]["enum"]),
            ["calculate", "thinking"],
        )
        self.assertIn("tool_name", choice["required"])
        for branch in choice["anyOf"]:
            self.assertNotIn("$ref", branch)
            self.assertIn("properties", branch)
            self.assertIn("tool_name", branch["properties"])

    def test_dynamic_tool_choice_schema_with_refs(self):
        """inline=False keeps the older $defs/$ref layout."""
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]

        dynamic_schema = dynamic_tool_choice(tools=tools, inline=False)

        choice = dynamic_schema["properties"]["tool_choice"]
        ref_targets = sorted(b["$ref"] for b in choice["anyOf"])
        self.assertEqual(
            ref_targets,
            ["#/$defs/Calculate", "#/$defs/Thinking"],
        )
