import inspect
import typing
from typing import cast

from mcp import ClientSession
from mcp.types import CallToolResult
from mcp.types import EmbeddedResource
from mcp.types import ImageContent
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from synalinks.src.modules.core.tool import Tool
from synalinks.src.utils.mcp.sessions import Connection
from synalinks.src.utils.mcp.sessions import create_session

NonTextContent = ImageContent | EmbeddedResource
MAX_ITERATIONS = 1000


class ToolException(Exception):  # noqa: N818
    """Exception thrown when a tool execution error occurs.

    This exception allows tools to signal errors without stopping the agent.
    """


class ToolMessage(typing.TypedDict):
    """The tool response JSON schema"""

    response: str | list[str]


# TODO: support for non-text artifacts in the tool response
def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    text_contents: list[TextContent] = []
    non_text_contents = []

    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content: str | list[str] = [content.text for content in text_contents]
    if not text_contents:
        tool_content = ""
    elif len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise ToolException(tool_content)

    tool_message = {
        "response": tool_content,
        # "artifact": non_text_contents,
    }

    return tool_message


async def _list_all_tools(session: ClientSession) -> list[MCPTool]:
    current_cursor: str | None = None
    all_tools: list[MCPTool] = []

    iterations = 0

    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise RuntimeError("Reached max of 1000 iterations while listing tools.")

        list_tools_page_result = await session.list_tools(cursor=current_cursor)

        if list_tools_page_result.tools:
            all_tools.extend(list_tools_page_result.tools)

        if list_tools_page_result.nextCursor is None:
            break

        current_cursor = list_tools_page_result.nextCursor
    return all_tools


def _has_docstring_section(docstring: str, section: str) -> bool:
    """Check if a docstring already contains a specific section (Args, Returns, etc.)."""
    if not docstring:
        return False

    # Look for common docstring section patterns
    section_patterns = [
        f"\n{section}:",
        f"\n{section.lower()}:",
        f"\n{section.upper()}:",
        f"\n{section.capitalize()}:",
    ]

    return any(pattern in docstring for pattern in section_patterns)


def _create_async_function_from_mcp_tool(
    mcp_tool: MCPTool,
    session: ClientSession | None,
    connection: Connection | None = None,
    namespace: str | None = None,
) -> typing.Coroutine:
    """Create a dynamic async function from an MCP tool
    that can be wrapped by Synalinks tool.

    """
    properties = (
        mcp_tool.inputSchema.get("properties", {}) if mcp_tool.inputSchema else {}
    )
    required_params = (
        set(mcp_tool.inputSchema.get("required", [])) if mcp_tool.inputSchema else set()
    )

    parameters = []
    annotations = {}

    for param_name, param_schema in properties.items():
        param_type = _json_schema_to_python_type(param_schema)
        annotations[param_name] = param_type

        if param_name in required_params:
            param = inspect.Parameter(
                param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=param_type
            )
        else:
            param = inspect.Parameter(
                param_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=param_type,
                default=None,
            )

        parameters.append(param)

    name = f"{namespace}_{mcp_tool.name}" if namespace else mcp_tool.name

    description = mcp_tool.description or f"Execute {name} function"
    docstring = [description]

    if properties and not _has_docstring_section(description, "Args"):
        docstring.append("\nArgs:")

        for param_name, param_schema in properties.items():
            param_desc = param_schema.get("description", f"{param_name} parameter")
            docstring.append(f"    {param_name}: {param_desc}")

    docstring = "\n".join(docstring)

    signature = inspect.Signature(parameters)

    async def dynamic_function(**kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if session is None:
            # will create a session one on the fly
            async with create_session(connection) as tool_session:
                await tool_session.initialize()
                call_tool_result = await cast(ClientSession, tool_session).call_tool(
                    mcp_tool.name, filtered_kwargs
                )
        else:
            call_tool_result = await session.call_tool(mcp_tool.name, filtered_kwargs)

        tool_message = _convert_call_tool_result(call_tool_result)
        return tool_message

    dynamic_function.__name__ = name
    dynamic_function.__doc__ = docstring
    dynamic_function.__signature__ = signature
    dynamic_function.__annotations__ = annotations

    return dynamic_function


def _json_schema_to_python_type(schema: dict) -> type:
    """Convert JSON schema type to Python type."""
    schema_type = schema.get("type", "string")

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    return type_mapping.get(schema_type, str)


def convert_mcp_tool_to_synalinks_tool(
    session: ClientSession | None,
    tool: MCPTool,
    *,
    connection: Connection | None = None,
    namespace: str | None = None,
) -> Tool:
    """Convert an MCP tool to a Synalinks tool.

    NOTE: tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert
        connection: Optional connection config to use to create a new session
                    if a `session` is not provided
        namespace: Optional namespace to use for the tool name, if provided

    Returns:
        A Synalinks tool that wraps the MCP tool functionality
    """
    if session is None and connection is None:
        raise ValueError("Either a session or a connection config must be provided")

    function = _create_async_function_from_mcp_tool(
        tool, session, connection, namespace=namespace
    )
    return Tool(function)


async def load_mcp_tools(
    session: ClientSession | None,
    *,
    connection: Connection | None = None,
    namespace: str | None = None,
) -> list[Tool]:
    """Load all available MCP tools and convert them to Synalinks tools.

    Returns:
        A list of Synalinks tools with correct signature, annotations and schemas
    """
    if session is None and connection is None:
        raise ValueError("Either a session or a connection config must be provided")

    if session is None:
        # will create a session one on the fly
        async with create_session(connection) as tool_session:
            await tool_session.initialize()
            tools = await _list_all_tools(tool_session)
    else:
        tools = await _list_all_tools(session)

    converted_tools = [
        convert_mcp_tool_to_synalinks_tool(
            session, tool, connection=connection, namespace=namespace
        )
        for tool in tools
    ]
    return converted_tools
