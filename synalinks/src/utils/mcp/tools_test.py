# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from mcp.types import CallToolResult
from mcp.types import EmbeddedResource
from mcp.types import ImageContent
from mcp.types import TextContent
from mcp.types import TextResourceContents
from mcp.types import Tool as MCPTool

from synalinks.src import testing
from synalinks.src.modules.core.tool import Tool
from synalinks.src.utils.mcp.tools import ToolException
from synalinks.src.utils.mcp.tools import _convert_call_tool_result
from synalinks.src.utils.mcp.tools import convert_mcp_tool_to_synalinks_tool
from synalinks.src.utils.mcp.tools import load_mcp_tools


class MCPToolsTest(testing.TestCase):
    def test_convert_empty_text_content(self):
        result = CallToolResult(
            content=[],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], "")

    def test_convert_single_text_content(self):
        result = CallToolResult(
            content=[TextContent(type="text", text="test result")],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], "test result")

    def test_convert_multiple_text_contents(self):
        result = CallToolResult(
            content=[
                TextContent(type="text", text="result 1"),
                TextContent(type="text", text="result 2"),
            ],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], ["result 1", "result 2"])

    def test_convert_with_non_text_content(self):
        image_content = ImageContent(
            type="image", mimeType="image/png", data="base64data"
        )
        resource_content = EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="resource://test", mimeType="text/plain", text="hi"
            ),
        )

        result = CallToolResult(
            content=[
                TextContent(type="text", text="text result"),
                image_content,
                resource_content,
            ],
            isError=False,
        )

        tool_message = _convert_call_tool_result(result)

        self.assertEqual(tool_message["response"], "text result")
        self.assertNotIn("artifact", tool_message)

    def test_convert_with_error(self):
        result = CallToolResult(
            content=[TextContent(type="text", text="error message")],
            isError=True,
        )

        with self.assertRaises(ToolException) as exc_info:
            _convert_call_tool_result(result)

        self.assertEqual(str(exc_info.exception), "error message")

    async def test_convert_mcp_tool_to_synalinks_tool(self):
        tool_input_schema = {
            "properties": {
                "param1": {"title": "Param1", "type": "string"},
                "param2": {"title": "Param2", "type": "integer"},
            },
            "required": ["param1", "param2"],
            "title": "ToolSchema",
            "type": "object",
        }

        session = AsyncMock()
        session.call_tool.return_value = CallToolResult(
            content=[TextContent(type="text", text="tool result")],
            isError=False,
        )

        mcp_tool = MCPTool(
            name="test_tool",
            description="Test tool description",
            inputSchema=tool_input_schema,
        )

        synalinks_tool = convert_mcp_tool_to_synalinks_tool(session, mcp_tool)

        self.assertIsInstance(synalinks_tool, Tool)
        self.assertEqual(synalinks_tool.name, "test_tool")
        self.assertStartsWith(synalinks_tool.description, "Test tool description")

        result = await synalinks_tool(param1="test", param2=42)

        session.call_tool.assert_called_once_with(
            "test_tool", {"param1": "test", "param2": 42}
        )

        # New Tool module returns JsonDataModel, use get_json() to access data
        self.assertEqual(result.get_json()["response"], "tool result")

    async def test_load_mcp_tools(self):
        tool_input_schema = {
            "properties": {
                "param1": {"title": "Param1", "type": "string"},
                "param2": {"title": "Param2", "type": "integer"},
            },
            "required": ["param1", "param2"],
            "title": "ToolSchema",
            "type": "object",
        }

        session = AsyncMock()
        mcp_tools = [
            MCPTool(
                name="tool1",
                description="Tool 1 description",
                inputSchema=tool_input_schema,
            ),
            MCPTool(
                name="tool2",
                description="Tool 2 description",
                inputSchema=tool_input_schema,
            ),
        ]
        session.list_tools.return_value = MagicMock(tools=mcp_tools, nextCursor=None)

        async def mock_call_tool(tool_name, arguments):
            if tool_name == "tool1":
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"tool1 result with {arguments}")
                    ],
                    isError=False,
                )
            elif tool_name == "tool2":
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"tool2 result with {arguments}")
                    ],
                    isError=False,
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        session.call_tool.side_effect = mock_call_tool

        toolkit = await load_mcp_tools(session)

        self.assertEqual(len(toolkit), 2)
        self.assertTrue(all(isinstance(tool, Tool) for tool in toolkit))
        self.assertEqual(toolkit[0].name, "tool1")
        self.assertStartsWith(toolkit[0].description, "Tool 1 description")
        self.assertEqual(toolkit[1].name, "tool2")
        self.assertStartsWith(toolkit[1].description, "Tool 2 description")

        result1 = await toolkit[0](param1="test1", param2=1)
        self.assertEqual(
            result1.get_json()["response"],
            "tool1 result with {'param1': 'test1', 'param2': 1}",
        )

        result2 = await toolkit[1](param1="test2", param2=2)
        self.assertEqual(
            result2.get_json()["response"],
            "tool2 result with {'param1': 'test2', 'param2': 2}",
        )

    async def test_load_mcp_tools_with_namespace(self):
        tool_input_schema = {
            "properties": {
                "param1": {"title": "Param1", "type": "string"},
                "param2": {"title": "Param2", "type": "integer"},
            },
            "required": ["param1", "param2"],
            "title": "ToolSchema",
            "type": "object",
        }

        namespace = "testing"

        session = AsyncMock()
        mcp_tools = [
            MCPTool(
                name="tool1",
                description="Tool 1 description",
                inputSchema=tool_input_schema,
            ),
            MCPTool(
                name="tool2",
                description="Tool 2 description",
                inputSchema=tool_input_schema,
            ),
        ]
        session.list_tools.return_value = MagicMock(tools=mcp_tools, nextCursor=None)

        async def mock_call_tool(tool_name, arguments):
            if tool_name == "tool1":
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"tool1 result with {arguments}")
                    ],
                    isError=False,
                )
            elif tool_name == "tool2":
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"tool2 result with {arguments}")
                    ],
                    isError=False,
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        session.call_tool.side_effect = mock_call_tool

        toolkit = await load_mcp_tools(session, namespace=namespace)

        self.assertEqual(len(toolkit), 2)
        self.assertTrue(all(isinstance(tool, Tool) for tool in toolkit))
        self.assertEqual(toolkit[0].name, f"{namespace}_tool1")
        self.assertStartsWith(toolkit[0].description, "Tool 1 description")
        self.assertEqual(toolkit[1].name, f"{namespace}_tool2")
        self.assertStartsWith(toolkit[1].description, "Tool 2 description")

        result1 = await toolkit[0](param1="test1", param2=1)
        self.assertEqual(
            result1.get_json()["response"],
            "tool1 result with {'param1': 'test1', 'param2': 1}",
        )

        result2 = await toolkit[1](param1="test2", param2=2)
        self.assertEqual(
            result2.get_json()["response"],
            "tool2 result with {'param1': 'test2', 'param2': 2}",
        )
