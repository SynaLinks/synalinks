# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import typing

import docstring_parser
import pydantic

from synalinks.src import saving
from synalinks.src import testing
from synalinks.src.utils.tool_utils import Tool
from synalinks.src.utils.tool_utils import get_params_schema


@saving.object_registration.register_synalinks_serializable()
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


class ToolUtilsTest(testing.TestCase):
    def test_basic_tool(self):
        _ = Tool(calculate)

    async def test_tool_serialization(self):
        tool = Tool(calculate)
        tool_config = tool.get_config()
        new_tool = Tool.from_config(tool_config)

        tool_call = await new_tool("2+2")
        result = tool_call.get("result")

        self.assertTrue(result == 4)


class JsonSchemaTypeTest(testing.TestCase):
    def test_description_keeps_a_wrapped_summary(self):
        async def search(regex: str):
            """Find the predicates of a layer and read their
            files: how you discover what the layer knows.

            Args:
                regex (str): A regular expression.
            """
            return {"regex": regex}

        tool = Tool(search)
        expected = (
            "Find the predicates of a layer and read their\n"
            "files: how you discover what the layer knows."
        )
        self.assertEqual(tool.description, expected)
        self.assertEqual(tool.get_tool_schema()["description"], expected)


class GetParamsSchemaTest(testing.TestCase):
    def params_schema(self, func):
        return get_params_schema(func, docstring_parser.parse(func.__doc__))

    def test_optional_types_allow_null(self):
        async def f(a: str | None, b: None | int = None, c: typing.Optional[str] = None):
            """F.

            Args:
                a (str | None): A.
                b (int | None): B.
                c (str | None): C.
            """

        properties, required = self.params_schema(f)
        self.assertEqual(required, ["a"])
        self.assertEqual(properties["a"]["anyOf"], [{"type": "string"}, {"type": "null"}])
        self.assertEqual(
            properties["b"]["anyOf"], [{"type": "integer"}, {"type": "null"}]
        )
        self.assertEqual(properties["b"]["default"], None)
        self.assertEqual(properties["c"]["description"], "C.")

    def test_nested_models_are_inlined(self):
        class Point(pydantic.BaseModel):
            x: int

        async def f(points: list[Point]):
            """F.

            Args:
                points (list): The points.
            """

        properties, _ = self.params_schema(f)
        self.assertNotIn("$ref", str(properties))
        self.assertEqual(
            properties["points"]["items"]["properties"]["x"]["type"], "integer"
        )

    def test_recursive_model_raises(self):
        class Node(pydantic.BaseModel):
            children: list["Node"]

        async def f(node: Node):
            """F.

            Args:
                node (Node): The node.
            """

        with self.assertRaises(ValueError):
            self.params_schema(f)

    def test_missing_type_hint_raises(self):
        async def f(x):
            """F.

            Args:
                x: X.
            """

        with self.assertRaises(ValueError):
            self.params_schema(f)

    def test_missing_description_raises(self):
        async def f(x: int):
            """F."""

        with self.assertRaises(ValueError):
            self.params_schema(f)


class ToolValidationTest(testing.TestCase):
    def test_sync_function_raises(self):
        def sync_fn(x: str):
            """A sync function.

            Args:
                x (str): Input.
            """
            return x

        with self.assertRaises(TypeError):
            Tool(sync_fn)

    def test_no_docstring_raises(self):
        async def no_doc(x: str):
            pass

        with self.assertRaises(ValueError):
            Tool(no_doc)

    def test_tool_name(self):
        tool = Tool(calculate)
        self.assertEqual(tool.name, "calculate")

    def test_tool_description(self):
        tool = Tool(calculate)
        self.assertEqual(
            tool.description, "Calculate the result of a mathematical expression."
        )

    def test_tool_schema(self):
        tool = Tool(calculate)
        schema = tool.get_tool_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("expression", schema["properties"])
        self.assertIn("expression", schema["required"])
        self.assertEqual(schema["properties"]["expression"]["type"], "string")

    def test_tool_with_default_param(self):
        @saving.object_registration.register_synalinks_serializable()
        async def greet(name: str, greeting: str = "Hello"):
            """Greet someone.

            Args:
                name (str): The name of the person.
                greeting (str): The greeting to use.
            """
            return f"{greeting}, {name}!"

        tool = Tool(greet)
        schema = tool.get_tool_schema()
        self.assertIn("name", schema["required"])
        self.assertNotIn("greeting", schema["required"])
        self.assertEqual(schema["properties"]["greeting"]["default"], "Hello")

    def test_missing_type_hint_raises(self):
        async def bad_fn(x):
            """A function.

            Args:
                x (str): Input.
            """
            return x

        with self.assertRaises(ValueError):
            Tool(bad_fn)

    def test_missing_docstring_param_raises(self):
        async def bad_fn(x: str):
            """A function with no param docs."""
            return x

        with self.assertRaises(ValueError):
            Tool(bad_fn)
