"""
# Autonomous Agents

Autonomous agents represent a significant advancement in AI system design,
combining the power of language models with the ability to perform tasks
autonomously. This tutorial will guide you through building an autonomous
agent using Synalinks, capable of processing mathematical queries and
returning numerical answers.

## Understanding the Foundation

Autonomous agents address a fundamental limitation of traditional systems by
enabling them to perform tasks without constant human intervention. While
language models excel at generating coherent text, they often require
additional tools and logic to perform specific tasks autonomously. Autonomous
agents bridge this gap by dynamically processing information and executing
tasks based on predefined tools.

The architecture of an autonomous agent follows several core stages:

- The input stage captures the user's query or command.
- The processing stage uses predefined tools and logic to process the input
  and generate a response.
- The output stage returns the result to the user.

```mermaid
graph LR
    A[Query] --> B[Agent]
    B --> C{Need Tool?}
    C -->|Yes| D[Tool Call]
    D --> E[Tool Result]
    E --> B
    C -->|No| F[Final Answer]
```

Synalinks streamlines this complex process through its modular architecture,
allowing you to compose components with precision while maintaining flexibility
for different use cases.

## Creating an Autonomous Agent

Define tools as async functions with complete docstrings, then wrap them with `synalinks.Tool`:

```python
@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    \"\"\"Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate.
    \"\"\"
    result = eval(expression)
    return {"result": result, "log": "Successfully executed"}

tools = [synalinks.Tool(calculate)]
```

**Important Tool Constraints:**

- **No Optional Parameters**: All parameters must be required. OpenAI and
  other providers require all tool parameters to be required in JSON schemas.

- **Complete Docstring Required**: Every parameter must have a description
  in the `Args:` section of the docstring. The Tool uses these to build the
  JSON schema sent to the LLM.

Create the agent with `FunctionCallingAgent` in autonomous mode:

```python
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.FunctionCallingAgent(
    data_model=NumericalFinalAnswer,
    tools=tools,
    language_model=language_model,
    max_iterations=5,
    return_inputs_with_trajectory=True,
    autonomous=True,  # Agent runs until completion
)(inputs)
```

### Key Takeaways

- **Autonomous Task Execution**: Autonomous agents solve the fundamental
    problem of performing tasks without constant human intervention.
- **Synalinks Modular Implementation**: The framework simplifies the development
    of autonomous agents through composable components like `FunctionCallingAgent`.
- **Explicit Data Model Contracts**: Using structured `Query` and output models
    ensures type safety and predictable behavior.
- **Tool Integration**: Integrate tools like the calculate function into your
    autonomous agent for processing specific types of queries.
- **Dynamic Processing**: Autonomous agents dynamically process information
    and execute tasks based on predefined tools and logic.

## Program Visualization

![math_agent](../assets/examples/math_agent.png)

## API References

- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio

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


# Define the calculation tool
@synalinks.utils.register_synalinks_serializable()
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
            "log": (
                "Error: invalid characters in expression. "
                "The expression can only contain numbers, operators (+, -, *, /),"
                " parentheses, and spaces NOT letters."
            ),
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


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="autonomous_math_agent",
    )

    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # Define the tools available to the agent
    tools = [
        synalinks.Tool(calculate),
    ]

    # ==========================================================================
    # Create the autonomous agent
    # ==========================================================================
    print("Creating the autonomous agent...")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=NumericalFinalAnswer,
        tools=tools,
        language_model=language_model,
        max_iterations=5,
        return_inputs_with_trajectory=True,
        autonomous=True,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_agent",
        description="A math agent",
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
    print("Running the agent...")

    input_query = Query(query="How much is 152648 + 485?")
    response = await agent(input_query)

    print(response.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
