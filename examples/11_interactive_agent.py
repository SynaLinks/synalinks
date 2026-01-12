"""
# Interactive Agents

Interactive agents represent a significant advancement in AI system design,
enabling dynamic interactions with users through iterative processing and
validation of inputs. This tutorial will guide you through building an
interactive agent using Synalinks, capable of processing mathematical queries
and returning numerical answers through a series of interactions.

## Understanding the Foundation

Interactive agents address the need for dynamic and iterative processing of
user inputs, allowing for more flexible and responsive AI systems. Unlike
autonomous agents, interactive agents require user validation at each step,
ensuring accuracy and relevance in their responses.

The architecture of an interactive agent follows several core stages:

- The input stage captures the user's query or command.
- The processing stage uses predefined tools and logic to process the input
  and generate a response.
- The validation stage requires user input to validate the tool calls and
  their arguments.
- The output stage returns the result to the user.

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Tool

    User->>Agent: Query
    Agent->>User: Propose tool call
    User->>Agent: Validate/Edit args
    Agent->>Tool: Execute
    Tool-->>Agent: Result
    Agent->>User: Response
    Note over User,Agent: Repeat until done
```

Interactive agents are transforming the landscape of AI by facilitating
dynamic and iterative interactions with users.

## Exploring Interactive Agent Architecture

Synalinks offers a streamlined approach to building interactive agents, thanks
to its modular and flexible architecture. Each user interaction initiates a
new cycle in the Directed Acyclic Graph (DAG), and tool calls are executed
only after receiving user validation.

In non-autonomous mode (also called human in the loop or interactive mode), the
user needs to validate/edit the tool arguments and send it back to the agent. In
this mode, the agent requires a `ChatMessages` data model as input and outputs
`ChatMessages` back to the user (containing both tool results and assistant
response). The agent ignores the `max_iterations` argument, as it will only
perform one **step at a time**.

## Creating an Interactive Agent

Define tools as async functions with complete docstrings. **Important Tool Constraints:**

- **No Optional Parameters**: All parameters must be required. OpenAI and
  other providers require all tool parameters in their JSON schemas.

- **Complete Docstring Required**: Every parameter must be documented in the
  `Args:` section. The Tool extracts descriptions to build the JSON schema.

Use `ChatMessages` as input and set `autonomous=False`:

```python
inputs = synalinks.Input(data_model=synalinks.ChatMessages)
outputs = await synalinks.FunctionCallingAgent(
    tools=tools,
    language_model=language_model,
    return_inputs_with_trajectory=True,
    autonomous=False,  # Human-in-the-loop mode
)(inputs)
```

## Running the Conversation Loop

Process messages iteratively, validating tool calls at each step:

```python
input_messages = synalinks.ChatMessages(
    messages=[synalinks.ChatMessage(role="user", content="Calculate 2 + 2")]
)

for iteration in range(MAX_ITERATIONS):
    response = await agent(input_messages)
    assistant_message = response.get("messages")[-1]

    tool_calls = assistant_message.get("tool_calls")
    if not tool_calls:
        print(assistant_message.get("content"))  # Final response
        break

    # Display tool calls for user validation
    for tool_call in tool_calls:
        print(f"Tool: {tool_call.get('name')}({tool_call.get('arguments')})")

    # In a real app: let user approve/modify/reject here
    # After validation, append and continue
    input_messages.messages.append(assistant_message)
```

### Key Takeaways

- **Dynamic Interaction**: Interactive agents facilitate dynamic and iterative
    processing of user inputs.
- **Modular Design**: Synalinks modular architecture simplifies the development
    of interactive agents.
- **Structured Data Models**: The use of structured ChatMessages models ensures
    consistency and predictability.
- **Tool Integration and Validation**: Integration of tools and validation of
    their arguments provide a robust foundation.
- **User-Driven Processing**: Interactive agents dynamically process information
    and execute tasks based on user interactions.

## Program Visualization

![interactive_math_agent](../assets/examples/interactive_math_agent.png)

## API References

- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [ChatMessages (Base DataModels)](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# Activate logging for monitoring interactions
synalinks.enable_logging()

MAX_ITERATIONS = 5


# Define the calculation tool
@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Perform calculations based on a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
    """
    # Check for valid characters in the expression
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {
            "result": None,
            "log": (
                "Invalid characters detected in the expression. "
                "Only numbers, operators (+, -, *, /), parentheses, and spaces "
                "are allowed."
            ),
        }
    try:
        # Safely evaluate the mathematical expression
        result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
        return {
            "result": result,
            "log": "Calculation successful",
        }
    except Exception as e:
        return {
            "result": None,
            "log": f"Calculation error: {e}",
        }


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="interactive_math_agent",
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
    # Create the interactive agent
    # ==========================================================================
    print("Creating the interactive agent...")

    # Set up the input structure using ChatMessages
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Create the interactive agent
    # In non-autonomous mode, the agent returns ChatMessages containing both
    # tool results and assistant response
    outputs = await synalinks.FunctionCallingAgent(
        tools=tools,
        language_model=language_model,
        return_inputs_with_trajectory=True,
        autonomous=False,
    )(inputs)

    # Define the agent program
    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="interactive_math_agent",
        description="An agent designed to handle mathematical queries interactively",
    )

    synalinks.utils.plot_program(
        agent,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # ==========================================================================
    # Run the interactive conversation
    # ==========================================================================
    print("Running the interactive agent...")

    # Initialize the conversation with a user query
    input_messages = synalinks.ChatMessages(
        messages=[
            synalinks.ChatMessage(
                role="user",
                content="Calculate the sum of 152648 and 485.",
            )
        ]
    )

    # Process the conversation through multiple iterations
    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")
        response = await agent(input_messages)

        # The response contains all messages including tool results
        # The last message is the assistant response
        assistant_message = response.get("messages")[-1]

        # Check if the agent wants to call tools
        tool_calls = assistant_message.get("tool_calls")
        if not tool_calls:
            # No tool calls - the agent is done and has a final response
            print("\nAgent final response:")
            print(f"  {assistant_message.get('content')}")
            print("\nConversation complete.")
            break

        # =======================================================================
        # Human-in-the-loop validation step
        # In a real application, you would present these tool calls to the user
        # via a UI or CLI and let them approve, modify, or reject them.
        # =======================================================================
        print("\nAgent wants to call the following tools:")
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments")
            print(f"  [{i + 1}] {tool_name}({tool_args})")

        # Simulate user validation (in a real app, this would be interactive)
        print("\n[Simulated] User validates the tool calls...")
        print("[Simulated] Proceeding with execution...")

        # After validation, append the assistant message with tool_calls
        # The agent will execute the tools and continue in the next iteration
        input_messages.messages.append(assistant_message)


if __name__ == "__main__":
    asyncio.run(main())
