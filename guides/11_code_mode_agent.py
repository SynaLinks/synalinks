# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Code-Mode Agent

The `CodeModeAgent` is an alternative to `FunctionCallingAgent` that reasons
by **writing and executing Python** instead of emitting JSON tool calls.
Each turn, the language model produces a snippet that runs inside a
persistent, sandboxed REPL. State (variables, imports, function
definitions) accumulates across turns, so the agent can probe data, build
intermediate values, and iterate — the same way a human would at a Python
prompt.

## Why Code Mode?

Function calling forces the LM to express every operation as a discrete
tool invocation with rigid JSON arguments. That works well for simple
lookups, but it's awkward when the task naturally composes:

- "Fetch these three pages **in parallel**, then merge and rank them"
- "Call the tool, **filter** the results where ``score > 0.8``, average them"
- "Retry with the second-best candidate **if** the first returns empty"

In function-calling mode each of these steps becomes a separate round
trip. In code mode the LM writes an `async def main(): ...` that orchestrates
all of them in a single snippet, with real control flow, local variables,
and `asyncio.gather` for parallelism.

```mermaid
flowchart TD
    A[Input + Trajectory] --> B[Code Generator]
    B --> C["python_code: Python snippet"]
    C --> G[Execute in Monty REPL]
    G --> H["observation: stdout / stderr / result / error"]
    H --> S{submit called?}
    S -->|Yes, valid payload| F[Submitted payload → Output]
    S -->|No| I[Append to Trajectory]
    I --> J{max_iterations?}
    J -->|No| A
    J -->|Yes| E[Final Generator]
    E --> F
```

At each iteration the agent:

1. **Thinks** (optionally via ChainOfThought) and emits ONE Python snippet
2. **Executes** the snippet in the persistent sandbox
3. **Observes** stdout, stderr, return value, or error
4. **Terminates** by calling the always-present ``submit`` tool with the
   final payload (the canonical exit). If the LM forgets, the loop
   continues until ``max_iterations`` and the final generator formats the
   trajectory into the target schema as a fallback.

## Minimal Example

```python
import synalinks

class Query(synalinks.DataModel):
    query: str

class Answer(synalinks.DataModel):
    answer: str

language_model = synalinks.LanguageModel(model="ollama/mistral")
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.CodeModeAgent(
    data_model=Answer,
    language_model=language_model,
    max_iterations=5,
)(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs)
```

No tools are required — the agent can reason purely through arithmetic,
string manipulation, and the whitelisted stdlib.

## The Sandbox

Snippets run inside a ``Sandbox`` — an abstract base class with one
built-in implementation, ``MontySandbox``, backed by
`Monty <https://github.com/pydantic/monty>`_. Monty is a restricted
Python interpreter designed for LM-authored code; knowing its
constraints is essential when writing instructions, examples, or
debugging observations.

The agent itself stays **stateless**: by default, every ``call()``
builds a fresh ``MontySandbox`` internally. To share state across calls
— the foundation of interactive / human-in-the-loop code mode — the
caller constructs a ``MontySandbox`` explicitly and hands it in via
the ``sandbox`` kwarg (see *Interactive Mode* below).

### Persistent State

Variables, imports and function definitions persist across turns:

```python
# Turn 1
x = 100

# Turn 2 — `x` is still bound
print(x + 1)        # -> 101
```

This is what makes code mode efficient for multi-step tasks. The LM can
break work into bite-sized snippets without re-deriving state each turn.

### Input Binding

The user input is bound as a dict named `inputs` on the **first turn only**:

```python
# Turn 1
query = inputs["query"]
```

If you need the input later, stash it in your own variable — ``inputs``
is only injected once.

### Allowed Stdlib

Only this subset is importable:

    sys, os, typing, asyncio, re, datetime, json, math, pathlib

Even inside those modules, filesystem / environment / network access is
stubbed out: `open()`, `os.system`, `os.listdir`, `os.environ`,
`os.path`, `sys.argv`, `Path.read_text` are **not** available. `asyncio`
is also reduced — only `asyncio.run` and `asyncio.gather` exist. There
are no time primitives (no `sleep`, no `wait_for`).

Third-party libraries cannot be imported.

To reach anything beyond this — files, network, NumPy, your database,
a vendor SDK — the LM must call a **tool** you've bound to the agent.
Tool bodies are plain Python running on the host process; the
sandbox's whitelist applies only to the LM-authored code. See
*Binding Tools — The Bridge to the Host* below.

### Language Restrictions

- No `class` statements
- No `match` statements

Everything else works: functions, comprehensions, async def, decorators,
exceptions, etc.

## Binding Tools — The Bridge to the Host

Tools are the **only** bridge between the sandbox and the outside
world. The sandboxed code itself can't read files, open sockets, or
import third-party libraries — but a tool's body is plain Python that
runs in the host process, with **full host privileges**. When the LM
``await``s a tool inside the sandbox, control crosses out of the
restricted REPL, the tool executes on the host (filesystem, network,
NumPy, your database client, an HTTP API, anything Python can do),
and the return value is marshalled back into the sandbox as a dict.

```text
   ┌──────────────── MontySandbox (restricted) ─────────────────┐
   │                                                            │
   │   async def main():                                        │
   │       data = await fetch_url(url="https://api.example")    │   ← LM-authored code
   │       rows = await query_db(sql="SELECT ...")              │
   │       return {"summary": summarize(data, rows)}            │
   │                  │                  ▲                      │
   └──────────────────┼──────────────────┼──────────────────────┘
                      │ await            │ dict result
                      ▼                  │
   ┌──────── External functions (HOST, full privileges) ────────┐
   │   async def fetch_url(url):  # uses httpx, real network    │
   │   async def query_db(sql):   # uses psycopg, real DB       │
   └────────────────────────────────────────────────────────────┘
```

This is the whole design: **the sandbox locks the LM's reasoning code
into a small, well-defined surface; the tools you bind decide which
host capabilities the LM can reach and on what terms**. Want the agent
to hit the web? Bind an HTTP tool. Want it to talk to your vector
store? Bind a search tool. The LM never gets ambient access — every
host effect goes through a tool you wrote, with a signature you
control.

Any `synalinks.Tool` passed to `CodeModeAgent` is exposed inside the
REPL as a global **async callable**:

```python
@synalinks.saving.register_synalinks_serializable()
async def triple(x: int) -> int:
    \"\"\"Triple an integer.

    Args:
        x (int): the integer to triple.
    \"\"\"
    return x * 3

agent = synalinks.CodeModeAgent(
    data_model=Answer,
    language_model=lm,
    tools=[synalinks.Tool(triple)],
)
```

Inside a snippet, the LM calls it with `await` from an async entry point:

```python
import asyncio

async def main():
    result = await triple(x=7)
    return result["result"]

tripled = asyncio.run(main())
print(tripled)     # -> 21
```

A more realistic tool reaches outside the sandbox — that's the point.
Decorate the function with
``@synalinks.saving.register_synalinks_serializable()`` so the agent
(and any tool wired to it) round-trips cleanly through
``get_config`` / ``from_config``:

```python
@synalinks.saving.register_synalinks_serializable()
async def fetch_url(url: str) -> dict:
    \"\"\"Fetch a URL and return its body.

    Args:
        url (str): the URL to GET.
    \"\"\"
    import httpx          # third-party, unreachable from inside the sandbox
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return {"status": resp.status_code, "body": resp.text}

agent = synalinks.CodeModeAgent(
    data_model=Answer,
    language_model=lm,
    tools=[synalinks.Tool(fetch_url)],
)
```

The body of `fetch_url` is just Python. It runs on the host. It can
import any installed package, hit the real network, read files,
connect to a database — exactly the things the LM-authored snippet
*can't* do directly. The Monty sandbox is the firewall; tools are the
named, audited holes through it.

**Three things to remember when writing / debugging tool use:**

1. **Tools are async**: calling ``triple(x=7)`` without ``await`` returns
   a coroutine object, not the value. The LM must drive them through
   ``asyncio.run(main())``.

2. **Return values are always dicts**: a tool wrapping
   ``async def f(x) -> int`` yields ``{"result": <value>}``. A tool that
   already returns a dict yields that dict directly. Index the field
   before using it (``result["result"]``, not ``result`` itself).

3. **Naming gotcha**: each tool is registered under
   ``tool.name == tool._func.__name__``. ``Tool(_my_helper)`` shows up
   in the sandbox as ``_my_helper`` — rename the function rather than
   trying to alias it.

**Security implication.** Because tool bodies run with host
privileges, the set of bound tools defines the agent's effective
capability surface. Treat tool authorship like designing a public API:
validate arguments, scope credentials narrowly, and don't pass through
free-form shell strings or SQL. A tool that wraps ``subprocess.run``
or ``eval`` on LM-supplied input erases the sandbox boundary
completely.

## Termination

The agent always exposes a built-in async ``submit`` tool inside the
sandbox. Calling it is **the** way to end a run:

```python
import asyncio

async def main():
    # ... compute everything you need ...
    await submit(result={"answer": "42"})

asyncio.run(main())
```

When ``submit`` is invoked:

- The payload is captured as the final answer and the loop stops on
  that turn — no extra final-formatting LM call.
- If a target ``schema`` / ``data_model`` is configured, the payload is
  validated against it. Validation failures come back as an observation
  (``"submit validation failed: ..."``) and the LM can retry on the
  next turn.
- In schemaless mode any dict is accepted and appended to the
  trajectory as the final assistant message.

If the LM never calls ``submit``:

- Empty ``python_code`` is **not** a graceful exit any more — it gets
  fed back as a reminder observation (``"(no code emitted) Call the
  ``submit`` tool ..."``) and the loop continues.
- Once ``max_iterations`` is exhausted, the agent falls back to a
  single ``final_generator`` LM call that formats the accumulated
  trajectory into the target schema (or, schemaless, emits a final
  assistant ``ChatMessage`` appended to the trajectory).

In short: ``submit`` is the canonical, one-round-trip termination.
``max_iterations`` is the safety net, paid for with one extra LM call.

## Error Recovery

Sandbox errors (both ``MontyError`` and arbitrary Python exceptions) are
**caught** and surfaced back to the LM as observations, not raised:

```
stdout:
(nothing)
error: ZeroDivisionError: division by zero
```

The LM sees the error on the next turn and can revise its approach.
This is a core feature — the agent self-corrects without crashing the
surrounding `Program`.

## Time Budget

``timeout`` (seconds, default 5) is a **per-snippet** execution budget:
every ``run`` call on a ``MontySandbox`` starts with a fresh clock, so
idle LM latency between interactive turns and time spent on earlier
snippets do **not** eat into the budget of the current one. Monty's
native limit is cumulative across the REPL's lifetime; the sandbox
transparently rolls the REPL over via ``dump()``/``load()`` before each
call to restore per-snippet semantics (sub-millisecond overhead).

A snippet that hangs or spins exhausts its budget and surfaces as an
observation — never an exception in the outer program. The budget
applies to the snippet as a whole, including any tool calls it
dispatches.

When you inject your own ``MontySandbox`` via the ``sandbox`` kwarg,
that sandbox's ``timeout`` wins — the agent's ``timeout`` argument only
applies to the sandbox it builds internally.

## Interactive Mode

``CodeModeAgent`` is a drop-in replacement for ``FunctionCallingAgent``
and supports the same ``autonomous`` flag:

- ``autonomous=True`` (default): one ``call()`` runs the full
  think-execute-observe loop up to ``max_iterations`` and produces a
  structured final answer.
- ``autonomous=False``: one ``call()`` runs **a single** code turn,
  returns the updated trajectory, and leaves the next step to the
  caller. Requires a ``ChatMessages`` input.

The catch specific to code mode: REPL state (variables, imports,
function defs) lives in the ``Sandbox``. By default every ``call()``
builds a fresh one — fine for autonomous, but in interactive mode that
would throw away everything your prior snippet built. The fix is to
**hand the agent a sandbox you own**:

```python
import synalinks

agent = synalinks.CodeModeAgent(
    data_model=Answer,
    language_model=lm,
    tools=[...],
    autonomous=False,
)
sandbox = synalinks.MontySandbox(timeout=10)

# Turn 1
trajectory = synalinks.ChatMessages(
    messages=[synalinks.ChatMessage(role="user", content="set up")],
)
trajectory = await agent(trajectory, sandbox=sandbox)

# Turn 2 — same sandbox, state persists
trajectory = synalinks.ChatMessages(
    messages=list(trajectory.get("messages")) + [
        synalinks.ChatMessage(role="user", content="continue"),
    ],
)
trajectory = await agent(trajectory, sandbox=sandbox)

# New conversation → fresh sandbox
fresh = synalinks.MontySandbox(timeout=10)
await agent(other_trajectory, sandbox=fresh)
```

The agent itself stays stateless — concurrent calls are safe, the
module serializes cleanly, and there's no hidden "current session".
The orchestrator (you) decides when a conversation starts, ends, or
branches.

### Persisting Sandbox State

``MontySandbox`` is a ``SynalinksSaveable``. The full REPL namespace
(variables, imports, user-defined functions) round-trips through:

- ``dump() -> bytes`` / ``MontySandbox.load(bytes)``
- ``get_config()`` / ``MontySandbox.from_config(...)`` (state is
  base64-encoded in the config dict so it's JSON-safe)

That means you can store sandbox state alongside the conversation
trajectory in a database, rehydrate it between requests, or ship it
across processes:

```python
# Between turns, persist both
trajectory_json = trajectory.get_json()
sandbox_blob = sandbox.dump()
# ... store in DB / Redis / disk ...

# Later: restore and continue
sandbox = synalinks.MontySandbox.load(sandbox_blob)
trajectory = synalinks.ChatMessages(**trajectory_json)
trajectory = await agent(trajectory, sandbox=sandbox)
```

## Chain of Thought

Set ``use_chain_of_thought=True`` to wrap the code generator so it
produces a ``thinking`` field alongside ``python_code``. The thinking
text is prepended to the assistant message in the trajectory:

```python
agent = synalinks.CodeModeAgent(
    data_model=Answer,
    language_model=lm,
    tools=[...],
    use_chain_of_thought=True,
)
```

Useful for traceability and when the LM benefits from "working out loud"
before committing to a snippet.

## Trajectory

When ``return_inputs_with_trajectory=True`` (default), the output is the
final answer **concatenated with** the full ``ChatMessages`` trajectory:
every assistant code block, every observation. This gives you:

- Debuggable traces of what the LM did and what the sandbox reported
- Training data for optimizers
- Auditing for production deployments

Pass ``return_inputs_with_trajectory=False`` when you only need the
structured answer and want to cut token bloat downstream.

## Code Mode vs Function Calling

| Aspect              | FunctionCallingAgent              | CodeModeAgent                        |
|---------------------|-----------------------------------|--------------------------------------|
| Action shape        | JSON tool call                    | Python snippet                       |
| Parallelism         | LM decides, provider schedules    | ``asyncio.gather`` in one snippet    |
| Control flow        | Separate turn per branch          | ``if`` / ``for`` / ``try`` inline    |
| Intermediate state  | Passed through trajectory         | Persistent variables in the sandbox  |
| Interactive mode    | ``autonomous=False`` — stateless  | ``autonomous=False`` + caller-owned ``Sandbox`` |
| Best for            | Small, discrete actions           | Composition, transformation, retry   |

Reach for `CodeModeAgent` when the task naturally wants control flow,
batch transformations, or stateful exploration. Stick with
`FunctionCallingAgent` when every step is a standalone, well-defined
tool call — that case is simpler and uses less output-token budget.

For HITL / interactive flows, both agents support ``autonomous=False``.
Code mode adds one extra piece: hand the agent a ``MontySandbox``
through the ``sandbox`` kwarg and the orchestrator owns REPL state
between calls.

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

@synalinks.saving.register_synalinks_serializable()
async def fetch_price(ticker: str) -> float:
    \"\"\"Return the (mock) current price for a ticker.

    Args:
        ticker (str): stock ticker symbol, e.g. 'AAPL'.
    \"\"\"
    prices = {"AAPL": 189.5, "MSFT": 412.1, "NVDA": 925.0}
    return prices.get(ticker.upper(), 0.0)

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="gemini/gemini-2.0-flash")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.CodeModeAgent(
        data_model=Answer,
        language_model=lm,
        tools=[synalinks.Tool(fetch_price)],
        max_iterations=5,
        timeout=10,
        use_chain_of_thought=True,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="portfolio_agent",
    )

    # Query the LM will naturally express in code:
    #   prices = asyncio.run(gather all fetch_price calls)
    #   total = sum(prices)
    result = await agent(Query(
        query="What's the total price of AAPL, MSFT and NVDA combined?"
    ))
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

- **Code as the action space**: the LM writes Python snippets, not JSON.
  Control flow, parallel tool calls, and intermediate transformations
  become first-class.

- **Persistent REPL within a call**: state accumulates across turns of
  the autonomous loop, so the LM can break work into bite-sized
  snippets without re-deriving context each iteration.

- **Sandboxed**: ``Sandbox`` is an abstract base; the built-in
  ``MontySandbox`` restricts the stdlib, blocks filesystem / network,
  and bans ``class`` / ``match``. Errors and timeouts surface as
  observations instead of crashing the program.

- **Tools as async globals — the bridge to the host**: bound tools
  appear inside the sandbox as async callables returning dicts. Their
  **bodies run on the host** with full Python privileges (filesystem,
  network, third-party libraries), so the set of tools you bind
  defines exactly which external capabilities the LM can reach.
  Scripts must ``await`` them inside ``async def main()`` and drive
  with ``asyncio.run(...)``.

- **``submit`` for termination**: the LM ends a run by calling the
  always-present ``submit`` tool with the final payload — schema-validated
  when a target is configured. Hitting ``max_iterations`` without
  ``submit`` triggers a one-shot final-formatting LM call as a fallback.

- **Interactive mode** is opt-in via ``autonomous=False`` + a
  caller-owned ``MontySandbox`` passed through the ``sandbox`` kwarg.
  The agent stays stateless; the orchestrator owns session lifecycle
  and can ``dump()`` / ``load()`` sandbox state alongside the
  trajectory.

- **When to pick code mode**: composition, batching, conditional retry.
  For single discrete tool calls, ``FunctionCallingAgent`` is simpler.

## API References

- [CodeModeAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/CodeModeAgent%20module/)
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
#
# Inside the code-mode sandbox these appear as global async callables.
# Every tool returns a **dict**: one wrapping a primitive-returning
# coroutine yields {"result": <value>}; one that already returns a dict
# yields that dict directly.
# =============================================================================


@synalinks.saving.register_synalinks_serializable()
async def fetch_price(ticker: str) -> float:
    """Return the (mock) current price for a ticker.

    Args:
        ticker (str): stock ticker symbol, e.g. 'AAPL'.
    """
    prices = {"AAPL": 189.5, "MSFT": 412.1, "NVDA": 925.0}
    return prices.get(ticker.upper(), 0.0)


@synalinks.saving.register_synalinks_serializable()
async def currency_rate(base: str, quote: str) -> float:
    """Return the (mock) exchange rate from `base` to `quote`.

    Args:
        base (str): source currency code, e.g. 'USD'.
        quote (str): target currency code, e.g. 'EUR'.
    """
    rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09}
    return rates.get((base.upper(), quote.upper()), 1.0)


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/qwen3")

    # -------------------------------------------------------------------------
    # Example 1: Code-mode agent composing two tools in one snippet
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: CodeModeAgent with parallel tool calls")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.CodeModeAgent(
        data_model=Answer,
        language_model=lm,
        tools=[
            synalinks.Tool(fetch_price),
            synalinks.Tool(currency_rate),
        ],
        max_iterations=5,
        timeout=10,
        use_chain_of_thought=True,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="portfolio_agent",
    )

    print("\nQuery: total USD price of AAPL, MSFT and NVDA")
    result = await agent(
        Query(query="What is the total USD price of AAPL, MSFT and NVDA combined?")
    )
    print(f"Answer: {result['answer']}")

    print("\nQuery: same total, but in EUR")
    result = await agent(
        Query(
            query=(
                "What is the total price of AAPL, MSFT and NVDA in EUR? "
                "Fetch all three prices, sum them, then convert USD->EUR."
            )
        )
    )
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Example 2: Tool-less code-mode agent (pure computation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: CodeModeAgent with no tools (pure Python)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.CodeModeAgent(
        data_model=Answer,
        language_model=lm,
        max_iterations=3,
    )(inputs)

    calc_agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="pure_code_agent",
    )

    print("\nQuery: standard deviation of a list")
    result = await calc_agent(
        Query(
            query=(
                "Compute the standard deviation of [4, 8, 15, 16, 23, 42]. Use `math`."
            )
        )
    )
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Example 3: Interactive mode with a caller-owned sandbox
    #
    # The orchestrator (this script) owns the sandbox; the agent stays
    # stateless. State persists across two `call()` invocations because we
    # hand the same MontySandbox in both times.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Interactive mode — caller owns the Sandbox")
    print("=" * 60)

    interactive_agent = synalinks.CodeModeAgent(
        data_model=Answer,
        language_model=lm,
        autonomous=False,
        max_iterations=1,
    )
    sandbox = synalinks.MontySandbox(timeout=10)

    # Turn 1: seed some REPL state
    trajectory = synalinks.ChatMessages(
        messages=[
            synalinks.ChatMessage(
                role="user",
                content="Compute and store `total = 15 * 23 + 7`. Print nothing.",
            ),
        ],
    )
    trajectory = await interactive_agent(trajectory, sandbox=sandbox)
    print(f"\nTurn 1 messages: {len(trajectory.get('messages'))}")

    # Turn 2: reuse `total` — only works because the sandbox survives.
    messages = list(trajectory.get("messages")) + [
        synalinks.ChatMessage(
            role="user",
            content="Now print `total * 2`.",
        ).get_json(),
    ]
    trajectory = synalinks.ChatMessages(
        messages=[synalinks.ChatMessage(**m) for m in messages],
    )
    trajectory = await interactive_agent(trajectory, sandbox=sandbox)
    print(f"Turn 2 messages: {len(trajectory.get('messages'))}")
    last_tool = [m for m in trajectory.get("messages") if m.get("role") == "tool"][-1]
    print(f"Last observation: {last_tool.get('content')!r}")


if __name__ == "__main__":
    asyncio.run(main())
