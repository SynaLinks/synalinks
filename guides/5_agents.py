"""
# Agents

**Agents** represent a paradigm shift from simple LLM calls to autonomous
systems that can reason, plan, and take action. While a Generator produces
a single output, an Agent iteratively thinks, selects tools, executes them,
and uses the results to inform its next steps - continuing until it achieves
its goal or reaches a limit.

## The Agent Loop

The `FunctionCallingAgent` uses an internal **ChainOfThought** module to
reason about which tools to call. Here's the complete autonomous loop:

```mermaid
flowchart TD
    A[Input + Trajectory] --> B[ChainOfThought]
    B --> C[thinking + tool_calls]
    C --> D{tool_calls empty?}
    D -->|Yes| E[Final Generator]
    E --> F[Formatted Output]
    D -->|No| G[Execute Tools in Parallel]
    G --> H[Append Results to Trajectory]
    H --> I{max_iterations?}
    I -->|No| A
    I -->|Yes| E
```

At each iteration, the agent:

1. **Thinks**: Uses ChainOfThought to analyze the trajectory and decide next action
2. **Decides**: Returns `tool_calls` array (empty if done) with reasoning in `thinking`
3. **Acts**: Executes all requested tools in parallel using `asyncio.gather`
4. **Observes**: Appends tool results to trajectory for next iteration

## FunctionCallingAgent: The Primary Agent

The `FunctionCallingAgent` is Synalinks' main agent module. It uses the
language model's function calling capabilities to intelligently select and
invoke tools:

```python
import synalinks

agent = await synalinks.FunctionCallingAgent(
    data_model=Answer,           # Output schema
    language_model=lm,           # Which LLM to use
    tools=[tool1, tool2, tool3], # Available tools
    autonomous=True,             # Run until complete
    max_iterations=10,           # Safety limit
)(inputs)
```

## Defining Tools

Tools are async Python functions wrapped with `synalinks.Tool()`. They must:

1. Have type hints for all parameters
2. Have a complete docstring with an `Args:` section documenting every parameter
3. Be asynchronous (use `async def`)
4. **Have NO optional parameters** - all parameters must be required

```python
import synalinks

async def calculator(expression: str):
    \"\"\"Evaluate a mathematical expression.

    Args:
        expression (str): A mathematical expression like '2 + 2' or '15 * 23'.
    \"\"\"
    try:
        result = eval(expression)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

# Wrap the function as a Tool
calculator_tool = synalinks.Tool(calculator)
```

The `synalinks.Tool()` wrapper extracts the function's schema from its type
hints and docstring, making it available to the agent.

**Important Tool Constraints:**

- **No Optional Parameters**: OpenAI and other LLM providers require all
  tool parameters to be required in their JSON schemas. Do not use default
  values for parameters.

- **Complete Docstring Required**: Every parameter must be documented in the
  `Args:` section of the docstring. The Tool uses these descriptions to build
  the JSON schema sent to the LLM. Missing descriptions will raise a ValueError.

### Tool Design Best Practices

```mermaid
graph LR
    A[Good Tool Design] --> B[Clear Name]
    A --> C[Detailed Docstring]
    A --> D[Type Hints]
    A --> E[Error Handling]
    A --> F[Return Dict]
```

1. **Clear Names**: Use descriptive function names (e.g., `search_database`,
   not `search` or `db_query`)

2. **Detailed Docstrings**: The docstring is sent to the LLM - be specific
   about what the tool does, its parameters, and expected output

3. **Type Hints**: All parameters must have types. The types are converted
   to JSON schema for the LLM

4. **Error Handling**: Return error messages in the result dict rather than
   raising exceptions

5. **Return Dicts**: Tools should return dictionaries with meaningful keys

## Agent Modes

### Autonomous Mode

In autonomous mode, the agent runs until it decides to output a final answer
or reaches `max_iterations`:

```python
calculator_tool = synalinks.Tool(calculator)

outputs = await synalinks.FunctionCallingAgent(
    data_model=Answer,
    language_model=lm,
    tools=[calculator_tool],
    autonomous=True,       # Keep running until done
    max_iterations=10,     # Safety limit
)(inputs)
```

Use autonomous mode when:

- The task requires multiple tool calls
- You want the agent to figure out the workflow
- The number of steps is not known in advance

### Non-Autonomous Mode (Single Step)

In non-autonomous mode, the agent executes one iteration and returns:

```python
calculator_tool = synalinks.Tool(calculator)

outputs = await synalinks.FunctionCallingAgent(
    data_model=Answer,
    language_model=lm,
    tools=[calculator_tool],
    autonomous=False,      # Single step only
    max_iterations=1,
)(inputs)
```

Use non-autonomous mode when:

- You want manual control over each step
- You're building a human-in-the-loop system
- You need to inspect/modify state between steps

## Parallel Tool Calling

Modern LLMs support calling multiple tools in parallel. Synalinks agents
leverage this for efficiency:

```mermaid
graph LR
    A[Query] --> B[Agent]
    B --> C[Tool Call 1]
    B --> D[Tool Call 2]
    B --> E[Tool Call 3]
    C --> F[Results]
    D --> F
    E --> F
    F --> G[Continue/Output]
```

When the LLM determines that multiple tool calls are independent, it can
request them simultaneously. Synalinks executes these in parallel:

```python
# Agent might decide to call multiple tools at once
# Query: "What's 2+2 and what's 3*3?"
# Parallel calls: calculator("2+2"), calculator("3*3")
```

This significantly reduces latency for complex tasks.

## Trajectory Tracking

Use `return_inputs_with_trajectory=True` to include the full history of
tool calls in the output:

```python
calculator_tool = synalinks.Tool(calculator)

outputs = await synalinks.FunctionCallingAgent(
    data_model=Answer,
    language_model=lm,
    tools=[calculator_tool],
    autonomous=True,
    return_inputs_with_trajectory=True,  # Include history
)(inputs)

# Output includes:
# - Original input
# - All tool calls made
# - All tool results
# - Final answer
```

This is useful for:

- Debugging agent behavior
- Creating training data
- Auditing agent decisions

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

# =============================================================================
# Data Models
# =============================================================================

class Query(synalinks.DataModel):
    \"\"\"User request.\"\"\"
    query: str = synalinks.Field(description="User request")

class Answer(synalinks.DataModel):
    \"\"\"Final answer.\"\"\"
    answer: str = synalinks.Field(description="Final answer to the user")

# =============================================================================
# Tools (define async functions, then wrap with synalinks.Tool)
# =============================================================================

async def calculator(expression: str):
    \"\"\"Evaluate a mathematical expression.

    Args:
        expression (str): A mathematical expression like '2 + 2' or '15 * 23'.
    \"\"\"
    try:
        result = eval(expression)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

async def get_current_time():
    \"\"\"Get the current date and time.\"\"\"
    from datetime import datetime
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

async def convert_temperature(value: float, from_unit: str, to_unit: str):
    \"\"\"Convert temperature between Celsius and Fahrenheit.

    Args:
        value (float): The temperature value to convert.
        from_unit (str): Source unit ('celsius' or 'fahrenheit').
        to_unit (str): Target unit ('celsius' or 'fahrenheit').
    \"\"\"
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = (value * 9 / 5) + 32
        return {"result": f"{result:.1f}F"}
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5 / 9
        return {"result": f"{result:.1f}C"}
    else:
        return {"error": f"Cannot convert from {from_unit} to {to_unit}"}

# =============================================================================
# Main
# =============================================================================

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Wrap async functions as Tool objects
    calculator_tool = synalinks.Tool(calculator)
    time_tool = synalinks.Tool(get_current_time)
    temp_tool = synalinks.Tool(convert_temperature)

    # Create agent with tools
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=Answer,
        language_model=lm,
        tools=[calculator_tool, time_tool, temp_tool],
        autonomous=True,
        max_iterations=10,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="tool_agent",
    )

    # Test the agent
    result = await agent(Query(query="What is 15 * 23 + 7?"))
    print(f"Answer: {result['answer']}")

    result = await agent(Query(query="Convert 100 Fahrenheit to Celsius"))
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

- **Agent Loop**: Agents operate in an observe-think-act loop, iterating
  until they achieve their goal or reach a limit.

- **FunctionCallingAgent**: The primary agent module that uses LLM function
  calling to select and invoke tools intelligently.

- **Tool Requirements**: Tools are async functions with type hints and
  docstrings, wrapped with `synalinks.Tool()` before passing to the agent.

- **Autonomous vs Non-Autonomous**: Use autonomous mode for multi-step tasks,
  non-autonomous for single-step or human-in-the-loop workflows.

- **Parallel Tool Calling**: Agents can call multiple tools simultaneously
  for efficiency when the LLM determines calls are independent.

- **Error Handling in Tools**: Return error information in the result dict
  rather than raising exceptions, so the agent can reason about errors.

## API References

- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [Tool](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Tool%20module/)
- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User request."""

    query: str = synalinks.Field(description="User request")


class Answer(synalinks.DataModel):
    """Final answer."""

    answer: str = synalinks.Field(description="Final answer to the user")


# =============================================================================
# Tools (async functions to wrap with synalinks.Tool)
# =============================================================================


async def calculator(expression: str):
    """Evaluate a mathematical expression.

    Args:
        expression (str): A mathematical expression like '2 + 2' or '15 * 23'.
    """
    try:
        result = eval(expression)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


async def get_current_time():
    """Get the current date and time."""
    from datetime import datetime

    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


async def convert_temperature(value: float, from_unit: str, to_unit: str):
    """Convert temperature between Celsius and Fahrenheit.

    Args:
        value (float): The temperature value to convert.
        from_unit (str): Source unit ('celsius' or 'fahrenheit').
        to_unit (str): Target unit ('celsius' or 'fahrenheit').
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = (value * 9 / 5) + 32
        return {"result": f"{result:.1f}F"}
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5 / 9
        return {"result": f"{result:.1f}C"}
    else:
        return {"error": f"Cannot convert from {from_unit} to {to_unit}"}


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_5_agents",
    )

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # Autonomous Agent with Tools
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Autonomous Agent with Tools")
    print("=" * 60)

    # Wrap async functions as Tool objects
    calculator_tool = synalinks.Tool(calculator)
    time_tool = synalinks.Tool(get_current_time)
    temp_tool = synalinks.Tool(convert_temperature)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=Answer,
        language_model=lm,
        tools=[calculator_tool, time_tool, temp_tool],
        autonomous=True,
        max_iterations=10,
    )(inputs)

    agent_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="tool_agent",
    )

    print("\nQuery: What is 15 * 23 + 7?")
    result = await agent_program(Query(query="What is 15 * 23 + 7?"))
    print(f"Answer: {result['answer']}")

    print("\nQuery: Convert 100 Fahrenheit to Celsius")
    result = await agent_program(Query(query="Convert 100 Fahrenheit to Celsius"))
    print(f"Answer: {result['answer']}")

    print("\nQuery: What time is it right now?")
    result = await agent_program(Query(query="What time is it right now?"))
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Complex Multi-Tool Query (demonstrates parallel tool calling)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Complex Multi-Tool Query")
    print("=" * 60)

    result = await agent_program(
        Query(query="What is (25 * 4) + 10? Also, what's 32 Fahrenheit in Celsius?")
    )
    print(f"\nComplex query result: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
