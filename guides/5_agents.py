"""
# Guide 5: Agents

Agents are autonomous modules that can use tools to accomplish tasks.
This guide covers building agents with Synalinks.

## FunctionCallingAgent

The primary agent module for autonomous tool use:

```python
agent = synalinks.FunctionCallingAgent(
    data_model=Answer,
    language_model=lm,
    tools=[tool1, tool2],
    autonomous=True,
    max_iterations=10,
)
```

## Defining Tools

Tools are Python functions with type hints and docstrings:

```python
def calculator(expression: str) -> str:
    '''Evaluate a mathematical expression.'''
    return str(eval(expression))
```

## Agent Modes

- **Autonomous**: Agent runs until task complete
- **Non-Autonomous**: Agent executes one step and returns

## Parallel Tool Calling

Agents can call multiple tools in parallel for efficiency.

## Running the Example

```bash
uv run python guides/5_agents.py
```
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User request."""

    query: str = synalinks.Field(description="User request")


class Answer(synalinks.DataModel):
    """Final answer."""

    answer: str = synalinks.Field(description="Final answer to the user")


# =============================================================================
# STEP 2: Define Tools
# =============================================================================


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like '2 + 2' or '15 * 23'

    Returns:
        The result of the calculation as a string
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current date and time as a formatted string
    """
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.

    Args:
        value: The temperature value to convert
        from_unit: Source unit ('celsius' or 'fahrenheit')
        to_unit: Target unit ('celsius' or 'fahrenheit')

    Returns:
        The converted temperature as a string
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = (value * 9 / 5) + 32
        return f"{result:.1f}F"
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5 / 9
        return f"{result:.1f}C"
    else:
        return f"Cannot convert from {from_unit} to {to_unit}"


# =============================================================================
# STEP 3: Demonstrate Agents
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
    # 3.1: Basic Agent with Tools
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Autonomous Agent with Tools")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=Answer,
        language_model=lm,
        tools=[calculator, get_current_time, convert_temperature],
        autonomous=True,
        max_iterations=10,
    )(inputs)

    agent_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="tool_agent",
    )

    # Test with calculation
    print("\nQuery: What is 15 * 23 + 7?")
    result = await agent_program(Query(query="What is 15 * 23 + 7?"))
    print(f"Answer: {result['answer']}")

    # Test with temperature conversion
    print("\nQuery: Convert 100 Fahrenheit to Celsius")
    result = await agent_program(Query(query="Convert 100 Fahrenheit to Celsius"))
    print(f"Answer: {result['answer']}")

    # Test with current time
    print("\nQuery: What time is it right now?")
    result = await agent_program(Query(query="What time is it right now?"))
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # 3.2: Non-Autonomous Agent (Single Step)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Non-Autonomous Agent (Single Step)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=Answer,
        language_model=lm,
        tools=[calculator],
        autonomous=False,
        max_iterations=1,
    )(inputs)

    single_step_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="single_step_agent",
    )

    result = await single_step_program(Query(query="Calculate 2 + 2"))
    print(f"\nSingle step result: {result['answer']}")

    # -------------------------------------------------------------------------
    # 3.3: Complex Multi-Tool Query
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Complex Multi-Tool Query")
    print("=" * 60)

    result = await agent_program(
        Query(query="What is (25 * 4) + 10? Also, what's 32 Fahrenheit in Celsius?")
    )
    print(f"\nComplex query result: {result['answer']}")

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. TOOLS: Python functions with type hints and docstrings
2. AUTONOMOUS: Agent runs until task complete
3. NON-AUTONOMOUS: Single step execution
4. MAX_ITERATIONS: Safety limit for autonomous agents
5. PARALLEL CALLS: Agent can call multiple tools at once
6. ERROR HANDLING: Tools should return error messages, not raise
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
