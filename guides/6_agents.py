"""
# Agents

Picture a language model as a person sitting at a desk. A plain
`Generator` (which you have seen since [Guide 1](https://synalinks.github.io/synalinks/guides/Getting%20Started/)) lets that person read
one question and write one answer, then stop. An **agent** gives them
more freedom: the person may also pick up the phone and call a
helper — a calculator, a search engine, a clock — read the helper's
reply, think again, and repeat until they decide they have enough
information to write a final answer.

A `Generator` maps one input to one structured output in a single LM
call. An agent generalizes that into a small **control loop**. At each
step:

1. The model proposes an *action* — usually one or more **tool calls**,
   meaning "please run this helper function with these arguments."
2. The framework runs the tools and collects the results
   (**observations**).
3. The running log of `(input, actions, observations)` is fed back
   into the next prompt.

We keep iterating until the model declines to act (it returns an
empty list of tool calls), at which point a final LM call — the
**terminal generator** — looks at the accumulated log and produces
the output object the caller asked for.

In principle, an agent could loop forever. To guarantee termination we
also impose a hard upper bound on iterations, called `max_iterations`.

Three ingredients are enough to build an agent:

- A **language model** that supports both structured output (returns
  JSON matching a schema) and **function calling** (can ask to invoke
  named tools with typed arguments).
- A finite set of **tools**: typed Python `async` functions that may
  have side effects (read a clock, query a database, hit a web API).
- A **trajectory**: an append-only log (you only ever add to the end,
  never rewrite earlier entries) of the agent's thoughts, tool calls,
  and tool results.

Everything in this guide is built from these three.

## The Agent Loop

At every step, the agent must make exactly one decision: "call more
tools, or stop?" It signals "stop" by producing an empty list of tool
calls.

Synalinks' `FunctionCallingAgent` wraps a `ChainOfThought` module (the
one you met in [Guide 4](https://synalinks.github.io/synalinks/guides/Modules/)). At each step that module must produce two
things: a free-form `thinking` field (so we — and downstream optimizers
— can see the model's intermediate reasoning), and a `tool_calls`
array (the list of helper calls it wants to make this turn). An empty
`tool_calls` array is the agent's only way to stop early; otherwise the
loop runs until the iteration cap.

```mermaid
flowchart TD
    A["Input + Trajectory"] --> B["ChainOfThought (decision module)"]
    B --> C["thinking + tool_calls[]"]
    C --> D{"tool_calls == []"}
    D -->|"yes"| E["Final Generator"]
    E --> F["Output (Answer schema)"]
    D -->|"no"| G["asyncio.gather(tool_i(args_i))"]
    G --> H["Append calls + results to trajectory"]
    H --> I{"iter < max_iterations"}
    I -->|"yes"| A
    I -->|"no"| E
```

Each iteration has four phases — **think → decide → act → observe**:

1. **Think.** The decision module reads the current trajectory and
   produces a structured object containing `thinking` and
   `tool_calls`.
2. **Decide.** If `tool_calls` is empty, control exits the loop.
   Otherwise the array tells us which tools to invoke and with what
   arguments.
3. **Act.** All requested calls are scheduled at the same time using
   `asyncio.gather` (Python's standard way to run several `async`
   tasks concurrently). Doing them in parallel is correct only when
   the tools in a single batch are *independent* — that is, the order
   in which they finish does not change the answer.
4. **Observe.** Each tool's return value is appended to the
   trajectory. Because the prompt on the next *think* step includes
   the trajectory, the model now sees the new observations.

Three failure modes are worth naming up front, so you can recognize
them when you see them:

- **Non-termination.** If the model never emits an empty
  `tool_calls`, the loop runs all the way to `max_iterations`. The
  terminal generator still runs and produces *some* output, but it
  may be incomplete.
- **Hallucinated calls.** The model may "hallucinate" — confidently
  invent — a tool that does not exist, or pass arguments of the wrong
  type. Whether this raises depends on the tool; we will talk about
  error handling below.
- **Skipped tools.** A weaker model may answer from its **priors**
  (knowledge baked in during pre-training) and never call any tool at
  all, even when the correct answer needs one. The "what time is it"
  example below reliably reproduces this on small open-weight models.

## FunctionCallingAgent: Constructing an Agent

```python
import synalinks

agent = await synalinks.FunctionCallingAgent(
    data_model=Answer,           # terminal output schema (Pydantic DataModel)
    language_model=lm,           # backing LM (must support structured output)
    tools=[tool1, tool2, tool3], # finite tool set, fixed at construction time
    autonomous=True,             # iterate until termination signal
    max_iterations=10,           # hard upper bound on loop steps
)(inputs)
```

Note that `data_model` only constrains the **final** output. The
intermediate think-decide steps use a different, fixed schema with
`thinking` and `tool_calls` fields. Keeping the two schemas separate
is what lets you reuse the same generic agent for many different
output shapes — you only need to change `data_model`.

## Defining Tools

A tool is just an `async` Python function with typed parameters and a
JSON-serializable return value (typically a `dict`). The
`synalinks.Tool(fn)` wrapper **introspects** the function — that is,
it reads the function's type hints and docstring at runtime — and
builds a JSON schema describing the tool. That schema is what the LM
sees when deciding whether to call this tool.

Four requirements, each either enforced by the framework or relied on
by it:

1. **Type hints on every parameter.** Used to generate the JSON
   Schema `type` field (e.g. `"string"`, `"number"`).
2. **A Google-style `Args:` block describing every parameter.** Each
   description becomes that parameter's `description` in the schema,
   and the agent reads it to decide what to pass.
3. **`async def`, not `def`.** The agent runs tools concurrently with
   `await`; a synchronous function would block the event loop and
   freeze every other tool in the batch.
4. **Optional parameters are supported.** A parameter with a default
   value is omitted from the schema's `required` list and its default
   is emitted in the schema, so the LM may leave it out and your
   function's default applies. (The "every property must be required"
   rule some providers enforce only applies to *strict structured
   output*, a separate path from tool calling.)

```python
import synalinks

async def calculator(expression: str):
    \"\"\"Evaluate a mathematical expression.

    Args:
        expression (str): A mathematical expression like '2 + 2' or '15 * 23'.
    \"\"\"
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

calculator_tool = synalinks.Tool(calculator)
```

Two non-obvious points to keep in mind:

- **The docstring is part of the program.** It is not just
  documentation for human readers. The LM reads it to decide *when* to
  call this tool, so a vague docstring leads to worse tool selection.
  Treat the docstring with the same care as the function's signature.
- **Return errors as values, not exceptions.** Writing
  `return {"error": "..."}` lets the agent see the failure on its next
  *think* step and try something different. Raising an exception, by
  contrast, aborts the whole agent run.

### Tool Design Checklist

```mermaid
graph LR
    A["Tool"] --> B["Descriptive name"]
    A --> C["Complete Args docstring"]
    A --> D["Typed parameters"]
    A --> E["Errors as values"]
    A --> F["Dict-shaped return"]
```

1. **Names are part of the prompt.** The LM sees the function name
   when choosing tools. `search_pubmed` beats `search`; `search`
   beats `do_query`.
2. **Docstrings are sent verbatim** — copied as-is into the prompt.
   Specify units, formats, and edge cases.
3. **Type hints are non-negotiable.** Without them, schema generation
   fails.
4. **Surface errors as data.** Return `{"error": msg}`; do not raise.
5. **Always return a dict.** A bare scalar return value (e.g. just a
   number) makes the trajectory harder for the LM to parse on later
   steps.

## Agent Modes

### Autonomous

```python
calculator_tool = synalinks.Tool(calculator)

outputs = await synalinks.FunctionCallingAgent(
    data_model=Answer,
    language_model=lm,
    tools=[calculator_tool],
    autonomous=True,
    max_iterations=10,
)(inputs)
```

Use autonomous mode when you do not know in advance how many steps
will be needed and you trust the model to stop on its own. The agent
owns the control flow — it decides itself when to keep looping and
when to stop.

### Non-Autonomous (Single Step)

```python
calculator_tool = synalinks.Tool(calculator)

outputs = await synalinks.FunctionCallingAgent(
    data_model=Answer,
    language_model=lm,
    tools=[calculator_tool],
    autonomous=False,
    max_iterations=1,
)(inputs)
```

In non-autonomous mode the loop body runs exactly once. The caller
(your code, or a surrounding program) owns the control flow. This is
useful for human-in-the-loop systems (a human approves each step),
step-by-step debugging, or wiring the agent into a larger controller
such as a planner or critic.

## Parallel Tool Calling

If the LM puts several entries in `tool_calls` on the same turn, the
agent runs them at the same time rather than one after another:

```mermaid
graph LR
    A["Query"] --> B["Decision step"]
    B --> C["tool_a(args_a)"]
    B --> D["tool_b(args_b)"]
    B --> E["tool_c(args_c)"]
    C --> F["results[]"]
    D --> F
    E --> F
    F --> G["Next iteration or terminate"]
```

Running calls in parallel only gives the right answer when the calls
do not depend on each other. The agent has no way to check this; it
is the model's responsibility to put dependent calls on *separate*
turns. This is a **soft contract** — something the framework requests
but cannot enforce — and small models often break it. For example, a
weak model may ask the calculator to multiply two numbers in parallel
that should have been computed sequentially.

## Trajectory Tracking

Setting `return_inputs_with_trajectory=True` makes the output include the
original input plus the full trace (every decision and every observation):

```python
calculator_tool = synalinks.Tool(calculator)

outputs = await synalinks.FunctionCallingAgent(
    data_model=Answer,
    language_model=lm,
    tools=[calculator_tool],
    autonomous=True,
    return_inputs_with_trajectory=True,
)(inputs)
```

Use this output shape to:

- inspect why a particular answer was produced (debugging),
- collect supervision data for in-context optimisers (training signals
  built from real runs),
- audit which tools were called and with what arguments.

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    \"\"\"User request.\"\"\"
    query: str = synalinks.Field(description="User request")

class Answer(synalinks.DataModel):
    \"\"\"Final answer.\"\"\"
    answer: str = synalinks.Field(description="Final answer to the user")

async def calculator(expression: str):
    \"\"\"Evaluate a mathematical expression.

    Args:
        expression (str): A mathematical expression like '2 + 2' or '15 * 23'.
    \"\"\"
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression, {"__builtins__": None}, {})
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
        return {"result": f"{(value * 9 / 5) + 32:.1f}F"}
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        return {"result": f"{(value - 32) * 5 / 9:.1f}C"}
    else:
        return {"error": f"Cannot convert from {from_unit} to {to_unit}"}

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/qwen3:8b")

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

    agent = synalinks.Program(inputs=inputs, outputs=outputs, name="tool_agent")

    result = await agent(Query(query="What is 15 * 23 + 7?"))
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (from a run against `ollama/mistral:latest`):

```
============================================================
Example 1: Autonomous Agent with Tools
============================================================

Query: What is 15 * 23 + 7?
Answer: 343

Query: Convert 100 Fahrenheit to Celsius
Answer: The temperature in Celsius is 37.78

Query: What time is it right now?
Answer: I'm not currently able to provide real-time information. However, I can tell you what time it was when this conversation started, which was [insert time]. If you'd like to know the time in a specific location, please let me know the city and country, and I'll do my best to provide the current time for you.

============================================================
Example 2: Complex Multi-Tool Query
============================================================

Complex query result: (25 * 4) + 10 = 110, 32 Fahrenheit in Celsius is 0
```

Read this output carefully: it illustrates exactly the failure modes named
above.

- The arithmetic queries return the correct numeric answer, but `llama3.2`
  is small enough that we cannot tell from the final string alone whether
  the `calculator` tool was actually used, or whether the model just
  computed the answer from memory. With this LM, both are plausible.
- The temperature conversion reports `37.78`, but the tool itself would
  have returned `37.8` (it uses the format string `{:.1f}`, which rounds
  to one decimal). So either the terminal generator rephrased the tool's
  output and lost precision, or the tool was never called and the model
  computed the value itself with different rounding.
- The "what time is it" query is the textbook *skipped tool* failure: the
  model answers from its priors ("I cannot access real time") instead of
  calling `get_current_time`. Stronger models reliably pick this tool;
  small open-weight chat models often do not. This is a limitation of the
  model, not a bug in the framework.

Small open-weight models tend to be unreliable at tool-calling. The agent
framework is the same regardless of model strength; if you need reliable
tool use, pick a model that was trained for it.

## Take-Home Summary

- An **agent is a bounded loop** over (think, decide, act, observe).
  The only ways it can stop are `tool_calls == []` (the model chose
  to stop) or `iter == max_iterations` (the loop ran out of budget).
- **`FunctionCallingAgent` keeps two schemas separate**: the decision
  schema (`thinking` + `tool_calls`, used every turn) and the final
  output schema (`data_model`, used once at the end).
- **Tools are typed `async` functions** wrapped with
  `synalinks.Tool()`. Their docstrings and type hints **are** the
  prompt the LM sees about them.
- **Return errors as values** (`{"error": ...}`) instead of raising,
  so the agent can read the error on the next step and react.
- **Parallel tool calls** are only correct when the model groups
  truly independent calls together; the framework cannot check this
  for you.
- **Tool-calling quality is dominated by the underlying LM.** Expect
  skipped tools, hallucinated tools, and wrong-typed arguments, and
  design defensively.

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
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression, {"__builtins__": None}, {})
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
    synalinks.enable_logging()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_6_agents",
    # )

    lm = synalinks.LanguageModel(model="ollama/qwen3:8b")

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
    agent_program.summary()

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
