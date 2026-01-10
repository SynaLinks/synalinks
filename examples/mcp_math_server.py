"""
MCP Math Server

A simple MCP server that provides mathematical tools.
This server is used by the MCP agent example.

To run this server standalone:
    uv run python examples/mcp_math_server.py

The server uses stdio transport by default.
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from mcp.types import Tool

# Create the MCP server
server = Server("math-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="add",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number",
                    },
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="subtract",
            description="Subtract two numbers (a - b)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The number to subtract from",
                    },
                    "b": {
                        "type": "number",
                        "description": "The number to subtract",
                    },
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="multiply",
            description="Multiply two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number",
                    },
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="divide",
            description="Divide two numbers (a / b)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The dividend (number to be divided)",
                    },
                    "b": {
                        "type": "number",
                        "description": "The divisor (number to divide by)",
                    },
                },
                "required": ["a", "b"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool with the given arguments."""
    if name == "add":
        result = arguments["a"] + arguments["b"]
        return [TextContent(type="text", text=str(result))]

    elif name == "subtract":
        result = arguments["a"] - arguments["b"]
        return [TextContent(type="text", text=str(result))]

    elif name == "multiply":
        result = arguments["a"] * arguments["b"]
        return [TextContent(type="text", text=str(result))]

    elif name == "divide":
        if arguments["b"] == 0:
            return [TextContent(type="text", text="Error: Division by zero")]
        result = arguments["a"] / arguments["b"]
        return [TextContent(type="text", text=str(result))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
