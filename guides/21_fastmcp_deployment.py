# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Deployment: FastMCP

The previous guide,
[FastAPI Deployment](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/), exposed a Synalinks
program to **HTTP** callers — browsers, mobile apps, other backend
services. This guide exposes the same kind of program to a *very*
different kind of caller: a **language model client**.

If you have used Claude Desktop or Cursor and noticed that they can
call external tools — search the web, edit a file, query a database
— that's the world we are stepping into.

## What Is MCP?

**MCP** stands for *Model Context Protocol*. It is a small, open
protocol — published by Anthropic in 2024 — that lets a language
model client discover and call **tools** that you define. From the
LM's point of view, your program looks like a function it can
choose to invoke. From your point of view, the LM and the wire are
somebody else's problem; you only have to write the function.

**[FastMCP][fastmcp]** is the standard Python framework for writing
MCP servers. It plays the same role that FastAPI plays for REST: you
write typed Python functions, the framework handles the protocol.

[fastmcp]: https://gofastmcp.com/

If you've read the FastAPI guide, the next sentence will sound
familiar — and that's the point: most of what you know transfers.

## The Same Trick, Repeated

A Synalinks `Program` is an async function. An MCP **tool** is an
async function. So once again the adapter is small (we'll show
where `program` comes from in the *Lifespan* section just below;
ignore the missing import for now):

```python
@mcp.tool
async def solve(query: str) -> NumericalAnswer:
    \"\"\"Solve a word problem.\"\"\"
    result = await program(Query(query=query))    # `program`: see below
    return NumericalAnswer.model_validate(result.get_json())
```

(As in the sibling FastAPI guide, snippets here are illustrative;
the complete file lives under `## Source` at the bottom.)

FastMCP cares about three things in that function:

- **Its name** (`solve`) becomes the tool name shown to the LM
  client.
- **Its argument and return types** become the tool's JSON schema.
  Scalar types (`str`, `int`, `float`, `bool`) appear as flat,
  top-level tool parameters — that is what LM clients invoke best.
  Synalinks DataModels are Pydantic models, so FastMCP accepts them
  directly *too*; use them when an argument is genuinely
  structured. For the single-string input here we take a `str` and
  wrap it into our internal `Query` model on the next line — that
  way the LM sees a clean `query: string` parameter, not a nested
  object with a `query.query` field.
- **Its docstring** becomes the description the LM sees when it is
  deciding *whether* to call this tool. Write it as a prompt to the
  model, not as a developer-facing API note. ("Use this tool when
  the user asks…" reads well to an LM; "REST endpoint for solving"
  does not.)

A note on **shape, not type**: the program's *input* schema
(`Query`) and the tool's *argument* schema are allowed to differ.
FastAPI happens to make them line up — REST conventionally wraps a
body in a JSON object, and `Query(query="...")` matches that
exactly. MCP is a function-call protocol, so flat keyword
arguments are the natural shape. The tool body bridges the two.

## Two Steps, Not One: Build, Then Serve

The same warning that opened the FastAPI guide applies here: do not
let your MCP server build the program on first start. Building
means talking to an LM provider, which is exactly the kind of work
a server should not be doing during boot. If the build half-fails,
you have a half-initialised server confidently exposing a broken
tool.

So we keep the same two-step shape:

1. **Build the artifact once.** This file exposes a regular
   `build_and_save_program(path)` function and a `build` CLI verb
   that calls it. Run it from a shell, a CI job, or a Python REPL
   — wherever you prepare other ML artifacts:

   ```bash
   # From a shell:
   python -m guides.20_fastmcp_deployment build
   ```

   ```python
   # Or, equivalently, from a notebook / REPL. (The module name starts
   # with a digit, so import it via importlib rather than `import`.)
   import asyncio, importlib
   mod = importlib.import_module("guides.20_fastmcp_deployment")
   asyncio.run(mod.build_and_save_program(mod.PROGRAM_PATH))
   ```

2. **Serve.** The server's only startup job is to *load*
   `math_agent.json`. If the file is missing, it fails loudly — not
   silently.

## Lifespan: Loading The Program Once

Like FastAPI, FastMCP has a **lifespan**: an `async` context
manager that runs once at server startup and once at shutdown.
Whatever you `yield` from the lifespan becomes available on
`Context.lifespan_context` inside every tool call. (Don't worry too
much about the word "context" here — it's just FastMCP's name for
"the bundle of information available during one tool call.")

If you read the FastAPI guide, you've seen this exact pattern. One
small surface difference: the FastAPI lifespan receives `app:
FastAPI`, the FastMCP lifespan receives `server` (the `FastMCP`
instance). Same idea, different framework-supplied argument.

```python
@asynccontextmanager
async def lifespan(server):
    if not PROGRAM_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRAM_PATH} not found. Run "
            f"`python -m guides.20_fastmcp_deployment build` first."
        )
    program = synalinks.Program.load(str(PROGRAM_PATH))
    yield {"program": program}
```

To reach the yielded dict from inside a tool, take a `Context`
argument — FastMCP will inject it automatically as long as the
*type* is `Context`:

```python
from fastmcp import Context

@mcp.tool
async def solve(query: str, ctx: Context) -> NumericalAnswer:
    \"\"\"Solve a word problem expressed in natural language.\"\"\"
    program = ctx.lifespan_context["program"]
    result = await program(Query(query=query))
    answer_json = result.out_mask(mask=["messages"]).get_json()
    return NumericalAnswer.model_validate(answer_json)
```

The `out_mask` + `get_json()` + `model_validate(...)` step is the
same as in the FastAPI guide. The reason is the same: `await
program(...)` returns a generic `JsonDataModel` (on purpose, so
internal modules can reshape data freely), and an agent in
particular also returns its trajectory (`messages`) for training
and observability. At the boundary of *your* system you want the
typed view back, so we drop the trajectory with `out_mask` and
validate the rest against the response model.

## Transports

`mcp.run()` decides *how* the tool talks to its client. There are
two transports worth using today, plus a third kept for
backwards compatibility:

- **stdio** (default). The tool reads requests from standard input
  and writes responses to standard output, the same way command-
  line programs read and write text. The client launches your
  process as a subprocess. This is how Claude Desktop and Cursor
  use local MCP tools. If in doubt, this is the right choice.
- **http** — a regular HTTP server (streamable). Use this when the
  client is on a different machine. Replace `mcp.run()` at the
  bottom of this file with
  `mcp.run(transport="http", host="127.0.0.1", port=8001)`, or use
  `fastmcp run --transport http ...` from the FastMCP CLI.
- *(legacy)* **sse** (Server-Sent Events) — an older HTTP-based
  transport. Still supported, but the MCP spec is moving to
  streamable HTTP. New deployments should prefer `http`.

## Running The Server

Prerequisites: an LM you have access to. The build function
defaults to `gemini/gemini-3.1-flash-lite-preview`, which expects
a `GEMINI_API_KEY` env var. If you don't have one, edit
`build_and_save_program` and change the model string — a free
local option is `"ollama/llama3.2:latest"` (see
[Getting Started](https://synalinks.github.io/synalinks/guides/Getting%20Started/) for setup). `fastmcp`
itself is already a Synalinks dependency, so nothing extra to
install.

This module follows the standard FastMCP conventions from the
[official quickstart][fastmcp-quickstart]: a module-level `mcp`
object, `@mcp.tool` decorators, and `mcp.run()` inside
`if __name__ == "__main__":`. So:

```bash
# stdio (default) -- the way LM clients usually launch you:
python -m guides.20_fastmcp_deployment

# or run with the FastMCP CLI:
fastmcp run guides/20_fastmcp_deployment.py
```

Connecting a local MCP client (Claude Desktop, Cursor, …) is a
matter of adding a config entry that points at this script with
`stdio` as the transport. The exact format depends on the client;
their docs cover it.

If you're new to FastMCP, read the
[official quickstart][fastmcp-quickstart] alongside this guide. The
*shape* of the module — `FastMCP("name")`, `@mcp.tool`,
`mcp.run()` — is copied verbatim from there; only the *body* of
the tool is Synalinks-specific.

[fastmcp-quickstart]: https://gofastmcp.com/getting-started/quickstart

## Error Handling

There are *two* layers of "tool" in this file and the rules differ
slightly between them. Keeping them straight saves a lot of
confusion:

- The **inner tool**, `calculate`, is called by the agent *inside*
  the program. The agent loop is forgiving: if `calculate` returns
  `{"result": None, "log": "Error: ..."}`, the agent reads the log,
  tries a different approach, and keeps going. That's why
  `calculate` returns a dict with a `log` field instead of raising.
- The **MCP tool**, `solve`, is the outer boundary that talks to a
  language-model client. Here, errors are *part of the protocol*:
  if `solve` raises, FastMCP captures the exception, sends a
  structured error back to the client, and the LM on the other
  side gets a chance to react — by retrying with corrected input
  or by telling its user something went wrong.

Two practical consequences for the *outer* MCP tool:

- **Do not swallow exceptions inside a tool.** Let them propagate.
  An LM that sees a clear error can fix its own mistake; an LM
  that sees an empty response can only guess.
- **Crash early on bad input.** A clear `ValueError("expression
  must contain at least one digit")` is more useful to the model
  than a generic 500 deep inside the call.

By default FastMCP forwards the exception message verbatim to the
client. This is the right default for development — the LM (and
you, reading the logs) get useful information. In production you
will eventually want to pass `mask_error_details=True` to
`FastMCP(...)`: the *client* gets a generic message ("internal
error") while the *server* logs the real one. The trade-off is
real, in both directions:

- *Unmasked* leaks debugging detail to anything that talks to your
  server.
- *Masked* starves the LM of the retry signal that would have let
  it fix its own mistake.

Pick by deployment, not by reflex.

## Bonus: Re-using A FastAPI App As An MCP Server

If you already have the FastAPI server from the previous guide,
FastMCP can lift its endpoints into MCP tools for you in one line:

```python
import importlib
from fastmcp import FastMCP

# Your FastAPI module from the previous guide (its name starts with a
# digit, so import it via importlib rather than `import`).
fastapi_app = importlib.import_module("guides.19_fastapi_deployment")

mcp = FastMCP.from_fastapi(
    fastapi_app.app,
    name="math-agent",
)

if __name__ == "__main__":
    mcp.run()
```

Each FastAPI route becomes one MCP tool, with the route's Pydantic
request/response models reused as the tool's schema. This is the
cheapest way to make a single program reachable from both audiences.

One caveat worth knowing about: the auto-lifted tool inherits the
FastAPI route's request *body* shape. In our case the FastAPI
handler takes `query: Query`, so the auto-generated MCP tool ends
up with a nested `query.query` parameter — the exact shape we
hand-flattened to `query: str` above. If you go this route and
want a clean LM-facing schema, either flatten on the FastAPI side
(at the cost of less idiomatic REST), or accept the nested form
and trust the LM to fill it in.

## Concurrency Note

(Safe to skip on a first read — only matters once you train.)

Same caveat as the FastAPI guide: the program is shared across
tool invocations. Serving them only *reads* the program's
trainable variables (the configurable knobs an optimiser tunes
during training — see the [Training](https://synalinks.github.io/synalinks/guides/Training/) guide). The
optimiser is the only thing that ever *writes* to them, and it
doesn't run at tool-call time. If you intend to expose training
itself as a tool, give each training run its own program copy so
they don't fight over the same knobs.

## Take-Home Summary

- **MCP** lets a language model client call your code as a tool.
  **FastMCP** is the Python framework for it.
- An MCP **tool** is an `async def` whose **name**, **signature**,
  and **docstring** become the schema and prompt the LM sees.
  Prefer **scalar argument types** (`str`, `int`, `float`, `bool`)
  so the LM sees flat top-level parameters; reach for DataModels
  only when an argument is genuinely structured. The tool body
  bridges the wire shape (what the LM sends) and the program shape
  (what your `Program` expects).
- **Build once, serve forever.** The server loads a prepared
  artifact; building inside the server is the wrong place.
- **Load the program in the lifespan; reach it via
  `Context.lifespan_context`.** Same separation as FastAPI: setup
  out of the tool body, work in the tool body.
- **Tool errors are part of the protocol.** Raise honestly; mask
  details in production only when the trade-off is right for that
  deployment.
- **`FastMCP.from_fastapi(app)`** is the one-line trick if you
  already have a FastAPI server and want to expose it to LM
  callers without writing a second adapter.

## What To Learn Next

- [FastAPI Deployment](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/) — the sibling
  guide.
- [Observability](https://synalinks.github.io/synalinks/guides/Observability/) — production tracing applies
  here too; the spans nest cleanly under MCP tool calls if you
  enable it inside the lifespan.
- FastMCP's [getting started][fastmcp-start] — for transports
  beyond stdio, authentication, resources, prompts, and middleware.

[fastmcp-start]: https://gofastmcp.com/getting-started/welcome

## API References

- [Program.save / Program.load](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/Program%20Saving%20API/Program%20saving%20and%20loading/)
- [enable_observability](https://synalinks.github.io/synalinks/Synalinks%20API/Observability%20API/)
- [FastMCP quickstart](https://gofastmcp.com/getting-started/quickstart)
- [FastMCP server reference](https://gofastmcp.com/servers/fastmcp)
- [MCP specification](https://modelcontextprotocol.io/)
"""

# --8<-- [start:source]
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import Context, FastMCP
from pydantic import ValidationError

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """The user-facing tool input."""

    query: str = synalinks.Field(description="The user query")


class NumericalAnswer(synalinks.DataModel):
    """The user-facing tool output."""

    answer: float = synalinks.Field(description="The correct final numerical answer")


# =============================================================================
# Tool used by the agent. Distinct from the MCP tool below, which is the
# wire-facing endpoint we expose to language-model clients.
# =============================================================================


# The decorator registers `calculate` under its qualified name so that
# `Program.load(...)` can find it again after a server restart. Without
# it, loading would fail with "Could not locate function 'calculate'".
@synalinks.saving.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators
            (+, -, *, /), parentheses, and spaces.
    """
    # Whitelist the allowed characters first; this is what makes the
    # `eval` on the next-to-last line safe. NEVER `eval` arbitrary user
    # input — `eval(some_string)` is a remote-code-execution hazard in
    # the general case.
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {"result": None, "log": "Error: invalid characters in expression"}
    try:
        result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
        return {"result": result, "log": "Successfully executed"}
    except Exception as e:
        return {"result": None, "log": f"Error: {e}"}


# =============================================================================
# Build step -- explicit, run via `python -m ... build`. NOT called from
# the server lifespan: production deployments should load a known artifact,
# not re-build on the fly.
# =============================================================================


# Where the prepared program lives on disk. Override with the
# `MATH_AGENT_PATH` env var; defaults to `math_agent.json` in the cwd.
PROGRAM_PATH = Path(os.environ.get("MATH_AGENT_PATH", "math_agent.json"))


async def build_and_save_program(path: Path) -> "synalinks.Program":
    """Build the math agent and persist it to ``path``."""
    # Reset Synalinks's global registry — useful when re-running the
    # build in a long-lived REPL or notebook so state from a previous
    # build doesn't leak in.
    synalinks.clear_session()
    synalinks.set_default_language_model("gemini/gemini-3.1-flash-lite-preview")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=NumericalAnswer,
        tools=[synalinks.Tool(calculate)],
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_agent",
        description="A math agent",
    )
    program.save(str(path))
    return program


# =============================================================================
# FastMCP server -- loads a prepared artifact, fails fast if it is missing
# =============================================================================


@asynccontextmanager
async def lifespan(server):
    """Load the prepared program; refuse to start if it is missing."""
    if not PROGRAM_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRAM_PATH} not found. Run "
            f"`python -m guides.20_fastmcp_deployment build` first."
        )
    program = synalinks.Program.load(str(PROGRAM_PATH))
    yield {"program": program}


mcp = FastMCP("math-agent", lifespan=lifespan)


@mcp.tool
async def solve(query: str, ctx: Context) -> NumericalAnswer:
    """Solve an arithmetic word problem expressed in natural language.

    Use this tool whenever the user asks a question that needs
    careful arithmetic — counting, totals, averages, or any
    multi-step calculation. Pass the user's question verbatim as
    ``query``; the tool will return a single numerical answer.
    """
    program = ctx.lifespan_context["program"]
    result = await program(Query(query=query))
    if result is None:
        # Surfaced to the LM as a structured tool error; the model
        # can decide to apologise, retry, or rephrase.
        raise ValueError("Guard rejected input or output")
    # An agent returns its trajectory (`messages`) alongside the answer;
    # mask it out so the strict response model only sees its own fields.
    answer_json = result.out_mask(mask=["messages"]).get_json()
    try:
        return NumericalAnswer.model_validate(answer_json)
    except ValidationError as e:
        raise ValueError(f"Program produced off-schema output: {e}") from e


# =============================================================================
# Standard FastMCP runner. Equivalent to `fastmcp run` on this file.
# See https://gofastmcp.com/getting-started/quickstart.
# =============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import asyncio

        asyncio.run(build_and_save_program(PROGRAM_PATH))
    else:
        mcp.run()
