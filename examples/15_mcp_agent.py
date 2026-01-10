"""
# MCP Agent

This example demonstrates how to build an autonomous agent that uses tools
from an MCP (Model Context Protocol) server. MCP is a standard protocol for
connecting AI models to external tools and data sources.

## Understanding MCP Integration

MCP enables seamless integration between language models and external tools
through a standardized protocol. Synalinks provides the `MultiServerMCPClient`
class to connect to one or more MCP servers and load their tools as native
Synalinks Tool modules.

Key benefits of using MCP:
- **Standardized Protocol**: Use tools from any MCP-compatible server
- **Multiple Servers**: Connect to multiple MCP servers simultaneously
- **Namespace Support**: Avoid tool name collisions with namespacing
- **Transport Flexibility**: Support for stdio, HTTP, SSE, and WebSocket

## Setting Up the MCP Server

First, you need an MCP server. This example uses a simple math server
(`mcp_math_server.py`) that provides basic arithmetic operations.

To run the server standalone (for testing):
```bash
uv run python examples/mcp_math_server.py
```

## Creating an MCP Agent

Connect to MCP servers using `MultiServerMCPClient`:

```python
client = synalinks.MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["examples/mcp_math_server.py"],
            "transport": "stdio",
        },
    }
)

# Load all tools from connected servers
tools = await client.get_tools()
```

Then use the tools with a `FunctionCallingAgent`:

```python
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.FunctionCallingAgent(
    data_model=NumericalFinalAnswer,
    tools=tools,
    language_model=language_model,
    autonomous=True,
)(inputs)
```

### Key Takeaways

- **MCP Protocol**: Synalinks supports the Model Context Protocol for
    standardized tool integration.
- **MultiServerMCPClient**: Connect to multiple MCP servers and load tools
    with automatic namespacing.
- **Transport Options**: Support for stdio (subprocess), HTTP, SSE, and
    WebSocket transports.
- **Seamless Integration**: MCP tools work identically to native Synalinks
    tools with full observability support.

## Program Visualization

![mcp_agent](../assets/examples/mcp_agent.png)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks

synalinks.enable_logging()


# Define the data models
class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class NumericalFinalAnswer(synalinks.DataModel):
    final_answer: float = synalinks.Field(
        description="The correct final numerical answer",
    )


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="mcp_math_agent",
    )

    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # ==========================================================================
    # Connect to MCP server and load tools
    # ==========================================================================
    print("Connecting to MCP server...")

    # Get the absolute path to the MCP server script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, "mcp_math_server.py")

    # Create the MCP client with stdio transport
    # The server will be started as a subprocess
    client = synalinks.MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [server_path],
                "transport": "stdio",
            },
        }
    )

    # Load all tools from the MCP server
    # Each tool call will start a new session with the server
    tools = await client.get_tools()

    print(f"Loaded {len(tools)} tools from MCP server:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # ==========================================================================
    # Create the autonomous agent with MCP tools
    # ==========================================================================
    print("\nCreating the MCP agent...")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=NumericalFinalAnswer,
        tools=tools,
        language_model=language_model,
        max_iterations=10,
        return_inputs_with_trajectory=True,
        autonomous=True,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="mcp_agent",
        description="An agent using MCP tools for math operations",
    )

    synalinks.utils.plot_program(
        agent,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # ==========================================================================
    # Run the agent
    # ==========================================================================
    print("\nRunning the agent...")

    # Test with a simple calculation
    input_query = Query(query="What is (15 + 25) * 3 - 10?")
    response = await agent(input_query)

    print("\nAgent response:")
    print(response.prettify_json())

    # ==========================================================================
    # Run another query to demonstrate multiple tool calls
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Running another query...")

    input_query = Query(query="Calculate 100 divided by 4, then multiply by 7")
    response = await agent(input_query)

    print("\nAgent response:")
    print(response.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
