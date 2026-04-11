# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import typing

from synalinks.src import saving
from synalinks.src import testing
from synalinks.src.utils.tool_utils import Tool
from synalinks.src.utils.tool_utils import json_schema_type


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
    def test_basic_types(self):
        self.assertEqual(json_schema_type(int), "integer")
        self.assertEqual(json_schema_type(float), "number")
        self.assertEqual(json_schema_type(bool), "boolean")
        self.assertEqual(json_schema_type(str), "string")
        self.assertEqual(json_schema_type(type(None)), "null")

    def test_unparameterized_list(self):
        result = json_schema_type(list)
        self.assertEqual(result, {"type": "array", "items": {}})

    def test_unparameterized_dict(self):
        result = json_schema_type(dict)
        self.assertEqual(result, {"type": "object", "additionalProperties": {}})

    def test_typed_list(self):
        result = json_schema_type(typing.List[str])
        self.assertEqual(result, {"type": "array", "items": {"type": "string"}})

    def test_typed_list_of_dict(self):
        result = json_schema_type(typing.List[typing.Dict[str, int]])
        self.assertEqual(
            result,
            {
                "type": "array",
                "items": {"type": "object", "additionalProperties": {"type": "integer"}},
            },
        )

    def test_typed_dict(self):
        result = json_schema_type(typing.Dict[str, int])
        self.assertEqual(
            result, {"type": "object", "additionalProperties": {"type": "integer"}}
        )

    def test_typed_dict_with_complex_value(self):
        result = json_schema_type(typing.Dict[str, typing.List[int]])
        self.assertEqual(
            result,
            {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        )

    def test_optional_type(self):
        result = json_schema_type(typing.Optional[str])
        self.assertEqual(result, "string")

    def test_union_type(self):
        result = json_schema_type(typing.Union[str, int, float])
        self.assertEqual(result, ["string", "integer", "number"])

    def test_unsupported_type(self):
        class Custom:
            pass

        with self.assertRaises(ValueError):
            json_schema_type(Custom)


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
        self.assertEqual(
            schema["properties"]["greeting"]["default"], "Hello"
        )

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
