"""A Synalinks agent exposed as an MCP server with FastMCP.

MCP (Model Context Protocol) lets a language-model client — Claude Desktop,
Cursor, … — discover and call your code as a tool. This server exposes a
Synalinks `FunctionCallingAgent` as one MCP tool, `solve`.

Build once, serve forever — two steps, never one:

    uv run python main.py build   # 1. build the agent, save it to disk
    uv run python main.py         # 2. serve it over stdio (what clients launch)

The server's only startup job is to *load* the prepared artifact and fail fast
if it is missing — it never rebuilds on the fly. See the guide:
https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/

The model is read from the MODEL env var; it defaults to vLLM. MODEL is baked
into the saved program at *build* time, so set it before `build`. Set
MLFLOW_TRACKING_URI to trace tool calls to MLflow.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Context
from fastmcp import FastMCP
from pydantic import ValidationError

import synalinks

# Used at build time; override via the MODEL env var (e.g. in `.env`).
DEFAULT_MODEL = "vllm/Qwen/Qwen3-4B"

# Where the prepared program is written/loaded. Override with PROGRAM_PATH.
PROGRAM_PATH = Path(os.environ.get("PROGRAM_PATH", "program.json"))


def _configure_env() -> None:
    """Defaults that must be set before the LM is created.

    vLLM's OpenAI endpoint lives at /v1; set a correct default (overridable via
    .env / real env). Synalinks' own fallback omits /v1, which 404s.
    """
    os.environ.setdefault("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1")


def _enable_observability() -> None:
    """Enable MLflow tracing when MLFLOW_TRACKING_URI is set (no-op otherwise)."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        synalinks.enable_observability(
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "synalinks_traces"),
        )


# =============================================================================
# Data models — the agent's structured input/output.
# =============================================================================


class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")


class NumericalAnswer(synalinks.DataModel):
    answer: float = synalinks.Field(description="The correct final numerical answer")


# =============================================================================
# Tool used by the agent (inner) — distinct from the MCP tool (outer) below.
# =============================================================================


# The decorator registers `calculate` under its qualified name so the agent can
# be restored by Program.load after a restart.
@synalinks.saving.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. May contain numbers, operators (+, -, *, /), parentheses
            and spaces.
    """
    # Whitelist allowed characters first; this is what makes the `eval` safe.
    # NEVER `eval` arbitrary user input in the general case.
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {"result": None, "log": "Error: invalid characters in expression"}
    try:
        result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
        return {"result": result, "log": "Successfully executed"}
    except Exception as e:
        return {"result": None, "log": f"Error: {e}"}


async def build_and_save_program(path: Path) -> "synalinks.Program":
    """Build the agent once and persist it to ``path``.

    Run this as a *separate* step (CLI verb, CI job, REPL) — never from the
    server lifespan. Building does not call the LM (it just composes modules and
    records schemas), so it works offline — no model needed until request time.
    """
    synalinks.clear_session()
    lm = synalinks.LanguageModel(model=os.environ.get("MODEL", DEFAULT_MODEL))

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=NumericalAnswer,
        tools=[synalinks.Tool(calculate)],
        language_model=lm,
    )(inputs)
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_agent",
        description="A tool-using math agent.",
    )
    program.save(str(path))
    return program


# =============================================================================
# FastMCP server — loads a prepared artifact, fails fast if it is missing.
# =============================================================================


@asynccontextmanager
async def lifespan(server):
    """Load the prepared program once; refuse to start if it is missing."""
    load_dotenv()
    _configure_env()
    _enable_observability()  # before loading any module
    if not PROGRAM_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRAM_PATH} not found. Build it first: "
            f"`uv run python main.py build`."
        )
    program = synalinks.Program.load(str(PROGRAM_PATH))
    # Whatever we yield is available on `Context.lifespan_context` per tool call.
    yield {"program": program}


mcp = FastMCP("synalinks-agent", lifespan=lifespan)


@mcp.tool
async def solve(query: str, ctx: Context) -> NumericalAnswer:
    """Solve an arithmetic word problem expressed in natural language.

    Use this tool whenever the user asks a question that needs careful
    arithmetic — counting, totals, averages, or any multi-step calculation.
    Pass the user's question verbatim as ``query``; the tool returns a single
    numerical answer.
    """
    program = ctx.lifespan_context["program"]
    result = await program(Query(query=query))
    if result is None:
        # Surfaced to the LM as a structured tool error; do not swallow it.
        raise ValueError("Guard rejected input or output")
    # An agent returns its trajectory (`messages`) alongside the answer; mask it
    # out so the strict response model only sees its own fields.
    answer_json = result.out_mask(mask=["messages"]).get_json()
    try:
        return NumericalAnswer.model_validate(answer_json)
    except ValidationError as e:
        raise ValueError(f"Program produced off-schema output: {e}") from e


# Standard FastMCP runner. `build` is a sibling verb (see the module docstring).
# For a remote client, swap `mcp.run()` for
# `mcp.run(transport="http", host="127.0.0.1", port=8001)`.
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import asyncio

        load_dotenv()
        _configure_env()
        asyncio.run(build_and_save_program(PROGRAM_PATH))
    else:
        mcp.run()
