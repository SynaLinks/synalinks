# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import synalinks
from synalinks.src import testing
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.core.tool import json_schema_type
from synalinks.src.modules.module import Module


class ToolModuleTest(testing.TestCase):
    """Tests for the Tool module."""

    def test_inheritance(self):
        """Test that Tool inherits from Module."""
        self.assertTrue(issubclass(Tool, Module))

    def test_init_with_valid_function(self):
        """Test initialization with a valid async function."""

        @synalinks.saving.register_synalinks_serializable()
        async def my_tool(arg1: str):
            """A test tool.

            Args:
                arg1 (str): The first argument.
            """
            return {"result": arg1}

        tool = Tool(func=my_tool)
        self.assertEqual(tool.name, "my_tool")
        self.assertEqual(tool.description, "A test tool.")
        self.assertFalse(tool.trainable)

    def test_init_with_custom_name(self):
        """Test initialization with custom name and description."""

        @synalinks.saving.register_synalinks_serializable()
        async def my_tool(arg1: str):
            """A test tool.

            Args:
                arg1 (str): The first argument.
            """
            return {"result": arg1}

        tool = Tool(func=my_tool, name="custom_name", description="Custom description")
        self.assertEqual(tool.name, "custom_name")
        self.assertEqual(tool.description, "Custom description")

    def test_init_with_sync_function_raises(self):
        """Test that sync functions raise TypeError."""

        def sync_func(arg1: str):
            """A sync function.

            Args:
                arg1 (str): The first argument.
            """
            return {"result": arg1}

        with self.assertRaises(TypeError):
            Tool(func=sync_func)

    def test_init_without_docstring_raises(self):
        """Test that functions without docstrings raise ValueError."""

        async def no_docstring(arg1: str):
            return {"result": arg1}

        with self.assertRaises(ValueError):
            Tool(func=no_docstring)

    async def test_call_executes_function(self):
        """Test that call executes the wrapped function."""

        @synalinks.saving.register_synalinks_serializable()
        async def add_numbers(a: int, b: int):
            """Add two numbers.

            Args:
                a (int): First number.
                b (int): Second number.
            """
            return {"sum": a + b}

        tool = Tool(func=add_numbers)
        result = await tool(a=5, b=3)
        self.assertEqual(result.get_json(), {"sum": 8})

    async def test_call_with_training_flag(self):
        """Test that call works with training flag."""

        @synalinks.saving.register_synalinks_serializable()
        async def simple_tool(value: str):
            """A simple tool.

            Args:
                value (str): A string value.
            """
            return {"value": value}

        tool = Tool(func=simple_tool)
        result = await tool(training=True, value="test")
        self.assertEqual(result.get_json(), {"value": "test"})

    def test_get_tool_schema(self):
        """Test get_tool_schema returns correct schema."""

        @synalinks.saving.register_synalinks_serializable()
        async def test_func(required_arg: str, optional_arg: int = 10):
            """Test function with required and optional args.

            Args:
                required_arg (str): A required string argument.
                optional_arg (int): An optional integer argument.
            """
            return {"result": required_arg}

        tool = Tool(func=test_func)
        schema = tool.get_tool_schema()

        self.assertEqual(schema["type"], "object")
        self.assertIn("required_arg", schema["properties"])
        self.assertIn("optional_arg", schema["properties"])
        self.assertIn("required_arg", schema["required"])
        self.assertNotIn("optional_arg", schema["required"])
        self.assertEqual(schema["properties"]["required_arg"]["type"], "string")
        self.assertEqual(schema["properties"]["optional_arg"]["type"], "integer")
        self.assertEqual(schema["properties"]["optional_arg"]["default"], 10)

    def test_get_config(self):
        """Test get_config returns serializable config."""

        @synalinks.saving.register_synalinks_serializable()
        async def my_tool(arg1: str):
            """A test tool.

            Args:
                arg1 (str): The first argument.
            """
            return {"result": arg1}

        tool = Tool(func=my_tool, name="test_tool", description="Test description")
        config = tool.get_config()

        self.assertIn("func", config)
        self.assertEqual(config["name"], "test_tool")
        self.assertEqual(config["description"], "Test description")
        self.assertFalse(config["trainable"])

    async def test_from_config(self):
        """Test from_config reconstructs the tool."""

        @synalinks.saving.register_synalinks_serializable()
        async def my_tool(arg1: str):
            """A test tool.

            Args:
                arg1 (str): The first argument.
            """
            return {"result": arg1}

        original_tool = Tool(func=my_tool)
        config = original_tool.get_config()

        restored_tool = Tool.from_config(config)
        self.assertEqual(restored_tool.name, original_tool.name)
        self.assertEqual(restored_tool.description, original_tool.description)

        # Test that the restored tool works
        result = await restored_tool(arg1="test")
        self.assertEqual(result.get_json(), {"result": "test"})


class JsonSchemaTypeTest(testing.TestCase):
    """Tests for json_schema_type function."""

    def test_basic_types(self):
        """Test conversion of basic Python types."""
        self.assertEqual(json_schema_type(int), "integer")
        self.assertEqual(json_schema_type(float), "number")
        self.assertEqual(json_schema_type(bool), "boolean")
        self.assertEqual(json_schema_type(str), "string")
        self.assertEqual(json_schema_type(type(None)), "null")

    def test_list_type(self):
        """Test conversion of list types."""
        from typing import List

        result = json_schema_type(List[str])
        self.assertEqual(result["type"], "array")
        self.assertEqual(result["items"]["type"], "string")

    def test_dict_type(self):
        """Test conversion of dict types."""
        from typing import Dict

        result = json_schema_type(Dict[str, int])
        self.assertEqual(result["type"], "object")
        self.assertEqual(result["additionalProperties"]["type"], "integer")

    def test_optional_type(self):
        """Test conversion of Optional types."""
        from typing import Optional

        result = json_schema_type(Optional[str])
        self.assertEqual(result, "string")

    def test_unsupported_type_raises(self):
        """Test that unsupported types raise ValueError."""

        class CustomClass:
            pass

        with self.assertRaises(ValueError):
            json_schema_type(CustomClass)


class ToolOutputSchemaTest(testing.TestCase):
    """Tests for Tool output schema generation."""

    def test_output_schema_no_return_type(self):
        """Test output schema when no return type is specified."""

        @synalinks.saving.register_synalinks_serializable()
        async def no_return_type(arg1: str):
            """A tool with no return type.

            Args:
                arg1 (str): The first argument.
            """
            return {"result": arg1}

        tool = Tool(func=no_return_type)
        schema = tool._build_output_schema()

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["title"], "no_return_type_output")
        self.assertTrue(schema["additionalProperties"])

    def test_output_schema_dict_return_type(self):
        """Test output schema with Dict[str, str] return type."""
        from typing import Dict

        @synalinks.saving.register_synalinks_serializable()
        async def dict_return(arg1: str) -> Dict[str, str]:
            """A tool with dict return type.

            Args:
                arg1 (str): The first argument.
            """
            return {"result": arg1}

        tool = Tool(func=dict_return)
        schema = tool._build_output_schema()

        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["title"], "dict_return_output")
        self.assertEqual(schema["additionalProperties"]["type"], "string")

    def test_output_schema_typed_dict_return(self):
        """Test output schema with TypedDict return type."""
        from typing import TypedDict

        class CalculationResult(TypedDict):
            result: int
            message: str

        @synalinks.saving.register_synalinks_serializable()
        async def typed_dict_return(a: int, b: int) -> CalculationResult:
            """A tool with TypedDict return type.

            Args:
                a (int): First number.
                b (int): Second number.
            """
            return {"result": a + b, "message": "Success"}

        tool = Tool(func=typed_dict_return)
        schema = tool._build_output_schema()

        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("result", schema["properties"])
        self.assertIn("message", schema["properties"])
        self.assertEqual(schema["properties"]["result"]["type"], "integer")
        self.assertEqual(schema["properties"]["message"]["type"], "string")
        self.assertFalse(schema["additionalProperties"])

    def test_get_input_schema(self):
        """Test that get_input_schema returns correct input schema."""

        @synalinks.saving.register_synalinks_serializable()
        async def my_tool(name: str, count: int = 5):
            """A tool with multiple parameters.

            Args:
                name (str): The name parameter.
                count (int): The count parameter.
            """
            return {"name": name, "count": count}

        tool = Tool(func=my_tool)
        schema = tool.get_input_schema()

        self.assertEqual(schema["type"], "object")
        self.assertIn("name", schema["properties"])
        self.assertIn("count", schema["properties"])
        self.assertIn("name", schema["required"])
        self.assertNotIn("count", schema["required"])
        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertEqual(schema["properties"]["count"]["type"], "integer")
