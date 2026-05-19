# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Deployment: FastAPI

Every guide so far has ended at the same line: `await program(...)`.
That line lives inside *your* Python script, and the caller of that
line is *you*. Real applications are different: the caller is
usually someone — or something — that cannot reach into your
process. It might be:

- a web page open in your friend's browser,
- a phone app written in Swift or Kotlin,
- another backend service written in Go.

None of those can `import` your Python file. They need a *network
protocol*: an agreed-upon way of sending bytes across a wire and
getting bytes back. **HTTP** is the protocol that powers the web.
**REST** is a popular *style* of designing HTTP APIs. And
**[FastAPI][fastapi]** is the standard Python framework for writing
REST APIs.

This guide wraps a Synalinks program in a small, real, runnable
FastAPI server.

[fastapi]: https://fastapi.tiangolo.com/

The sibling guide,
[FastMCP Deployment](FastMCP%20Deployment.md), wraps the same kind
of program for a *different* kind of caller — a language model
client. The two adapters look strikingly similar; pick by who's
calling.

## The Whole Trick: A Program Is An Async Function

Before any framework code, recall what a Synalinks program *is*.
Once built, it is an object you `await` with a typed input, and
that gives you a typed output back:

```python
result = await program(Query(query="..."))
```

Two things to read past, in case they are new:

- **`await f(...)`** — read it as `f(...)` — same idea as calling
  a normal function, except that Python may pause internally and
  let other work run before the result comes back. The
  `async`/`await` keywords mark the call as possibly-interrupted.
  We use them throughout because the LM provider talks to us over
  the network, and that network call is what gets paused.
- **`Query(query="...")`** — `Query` is the class (defined later
  in the file) and `query` is its single field. The outer and
  inner names are the same on purpose: a request body labelled
  `query` is the most natural REST shape. The repeated word is a
  coincidence, not a recursion.

That is the entire contract. Everything FastAPI adds is plumbing
around those two lines: parse the incoming JSON into a `Query`
instance, run that `await`, serialise the answer back to JSON.
Holding this picture in your head will keep the rest of the guide
simple.

(Code snippets in this guide are illustrative — they may omit
imports for brevity. The complete file lives under `## Source` at
the bottom.)

## Why DataModels Save You Work

A Synalinks `DataModel` inherits from `pydantic.BaseModel`. That
single fact does most of the work. (Inheritance is the same plain
Python idea you learned with `class Dog(Animal)`: any place that
expects an `Animal` accepts a `Dog`. Here, any place that expects a
`BaseModel` accepts a `DataModel`.)

What you get for free because of that inheritance:

- FastAPI **already** parses JSON request bodies into Pydantic
  models — so your `Query` class is a valid request body.
- FastAPI **already** turns Pydantic models back into JSON — so
  `NumericalAnswer` is a valid response body.
- FastAPI **already** publishes interactive docs at `/docs` with
  the JSON shape of each endpoint. Your input and output schemas
  appear there automatically.

You don't write the schemas twice; the ones you wrote for the LM
are reused on the wire.

## Two Steps, Not One: Build, Then Serve

It's tempting to write a server that builds the program the first
time it starts. *Don't.* Building a Synalinks program means setting
a default language model, instantiating an agent, calling an LM
provider during construction — all of which are operations a
production server has no business doing. If the agent build fails
halfway through your first request, you have a half-initialised
server that confidently returns wrong answers.

The right pattern is **two steps**:

1. **Build the program once**, in a separate preparation step.
   This file exposes a regular `build_and_save_program(path)`
   function and a `build` CLI verb that calls it. Run it from a
   shell, a CI job, or a Python REPL — wherever you prepare other
   ML artifacts:

   ```bash
   # From a shell:
   python -m guides.19_fastapi_deployment build
   ```

   ```python
   # Or, equivalently, from a notebook / REPL:
   import asyncio
   from guides.fastapi_deployment_19 import (
       build_and_save_program, PROGRAM_PATH,
   )
   asyncio.run(build_and_save_program(PROGRAM_PATH))
   ```

   That writes `math_agent.json` to disk.

2. **Serve.** The server's only job at startup is to *load* the
   prepared file. If the file isn't there, the server fails fast
   and loudly. Silently regenerating it is exactly the kind of
   "convenient" mistake that hides real problems.

This separation is the same principle behind machine-learning
deployments more generally: training (or any model preparation)
lives in one place, inference lives in another, and the wire
between them is a versioned artifact on disk.

## Loading The Program Once: The Lifespan Pattern

Even *loading* a file is too expensive to repeat on every request.
FastAPI gives you a hook called a **lifespan**: code that runs once
when the server starts, and once when it stops. Everything in
between is request-serving.

A lifespan is an `async` function decorated with
`@asynccontextmanager`. Three new words at once, so let's unpack
them in order:

- A **generator** is a function whose body contains the keyword
  `yield`. When you call it, it does *not* run all the way
  through — it runs up to the first `yield`, pauses there, and
  resumes from that point the next time it's asked to.
- A **context manager** is an object with a "setup" half and a
  "teardown" half, used with `with`: do setup, hand control to the
  block inside `with`, run teardown when the block exits.
- The **`@asynccontextmanager`** decorator turns an async
  generator into a context manager. The code *before* `yield` is
  the setup (runs at server startup), the `yield` itself is the
  moment FastAPI takes over and starts serving requests, and the
  code *after* `yield` is the teardown (runs at shutdown).

In code:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not PROGRAM_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRAM_PATH} not found. Run "
            f"`python -m guides.19_fastapi_deployment build` first."
        )
    program = synalinks.Program.load(str(PROGRAM_PATH))
    yield {"program": program}
    # teardown goes here (close clients, flush traces, …)
```

Two things to remember:

- `@asynccontextmanager` is the standard recipe; you don't usually
  write context managers from scratch.
- The `dict` you `yield` is merged into `request.state` for every
  request, so each handler can reach the program as
  `request.state.program` without using a global variable.

## The Handler

The handler is short. Its job is: receive a parsed input, await the
program, project the answer out of the agent's full output, return
a typed response. We'll meet `out_mask` for the first time — keep
reading, it's explained right below the snippet:

```python
@app.post("/solve")
async def solve(request: Request, query: Query) -> NumericalAnswer:
    result = await request.state.program(query)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Guard rejected input or output",
        )
    answer_json = result.out_mask(mask=["messages"]).get_json()
    try:
        return NumericalAnswer.model_validate(answer_json)
    except ValidationError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Program produced off-schema output: {e}",
        )
```

A sentence each on the moving parts:

- **`async def`** — the handler is async because the program is
  async. FastAPI awaits it directly. No thread pool, no
  `asyncio.run`, no `nest_asyncio`. If those terms mean nothing to
  you yet, ignore them — the point is: it just works.
- **`query: Query`** — because `query` is typed as a Pydantic
  model, FastAPI knows to read the request body, parse it as JSON,
  and validate it against `Query`. If the body is malformed, the
  handler never runs; FastAPI sends back `422 Unprocessable
  Entity` automatically.
- **`-> NumericalAnswer`** — modern FastAPI uses the return
  annotation as the response schema. (Older tutorials show
  `response_model=NumericalAnswer` as an extra decorator argument;
  it still works, but it's redundant when you have the annotation.)
- **`if result is None`** — a Synalinks guard (see
  [Input Guard](Input%20Guard.md) /
  [Output Guard](Output%20Guard.md)) returns `None` when it refuses
  the call. We translate that into HTTP `422 Unprocessable Entity`,
  which means "I understood the request, my own rules just won't
  let me process it." (Why not `502 Bad Gateway`? `502` means an
  *upstream service* failed. The guard didn't fail; *we* declined.)
- **`result.out_mask(mask=["messages"]).get_json()`** — the line
  that needs two sentences. Two independent facts conspire here:

    1. `await program(...)` always returns a generic JSON
       container (`JsonDataModel`), not a `NumericalAnswer`.
       Synalinks does this on purpose so internal modules can
       reshape data freely without breaking the rest of the graph.
    2. An *agent* in particular also includes a **trajectory** in
       its output — the running list of `messages` it exchanged
       with the LM — because that trajectory is essential for
       training, debugging, and observability.

    Together, the result has more fields than `NumericalAnswer`
    wants. `out_mask(mask=["messages"])` returns a *view* of the
    result with the trajectory dropped; `.get_json()` hands back
    the raw dict for the next step.
- **`NumericalAnswer.model_validate(answer_json)`** — now that the
  payload has only the fields the response model knows about,
  `model_validate` turns it into a `NumericalAnswer` instance. The
  `try/except` is defensive: in theory constrained decoding stops
  off-schema output before it reaches you, but a `502` on this
  branch documents the invariant ("if we ever produce bad output,
  that's an upstream problem, not the caller's fault").

## Running The Server

Prerequisites:

- An LM you have access to. The build function defaults to
  `gemini/gemini-3.1-flash-lite-preview`, which expects a
  `GEMINI_API_KEY` env var. If you don't have one, edit
  `build_and_save_program` and change the model string — a free
  local option is `"ollama/llama3.2:latest"` (see
  [Getting Started](Getting%20Started.md) for setup).
- The FastAPI toolkit installed:
  `uv pip install "fastapi[standard]" uvicorn`. The `[standard]`
  extra is what gives you the `fastapi` CLI used below; plain
  `fastapi` works too but only with the `uvicorn` long form.

This module follows the standard FastAPI conventions from the
[official tutorial][fastapi-first-steps]: a module-level `app`
object plus a normal `if __name__ == "__main__":` runner. So you
have the usual choices:

```bash
# Recommended (modern FastAPI CLI):
fastapi dev guides/19_fastapi_deployment.py

# Or the long form (works with any process manager):
uvicorn guides.19_fastapi_deployment:app --reload

# Or just run the file directly:
python -m guides.19_fastapi_deployment
```

In a separate terminal:

```bash
# Interactive docs (FastAPI ships these for free):
open http://127.0.0.1:8000/docs

# Liveness probe -- should always say "ok":
curl http://127.0.0.1:8000/healthz

# An actual call:
curl -X POST http://127.0.0.1:8000/solve \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is 12 * 7?"}'
```

If you're new to FastAPI, read the
[official tutorial][fastapi-first-steps] alongside this guide. The
*shape* of the module — `app`, decorators, lifespan, `__main__` —
is copied verbatim from there; only the *body* of the handler is
Synalinks-specific.

[fastapi-first-steps]: https://fastapi.tiangolo.com/tutorial/first-steps/

## Production Posture (For Later)

`uvicorn --reload` is great while you're developing. For real
deployments you'll want a few extras. None of them are specific to
Synalinks — they apply to any Python web service — so don't worry
about understanding them now. Just know they're the next things to
read about:

- **A process manager.** The standard recipe is gunicorn driving
  uvicorn workers:
  `gunicorn -k uvicorn.workers.UvicornWorker -w 4 module:app`.
  One worker handles many in-flight requests on its event loop;
  add workers only when CPU-heavy work (parsing, embeddings, local
  models) starts to dominate.
- **Timeouts.** Set one on the LM call inside the program *and* one
  on the HTTP server. Without them, a stuck upstream pinned by one
  request can stall every other request.
- **CORS** — if a browser is the caller. Without it, the browser
  blocks the request before it even leaves the page. Wrap the app
  with `fastapi.middleware.cors.CORSMiddleware`.
- **Authentication.** FastAPI ships [security helpers][security]
  for API keys, OAuth, JWTs, and friends. Synalinks doesn't opine.
- **Observability.** Call `synalinks.enable_observability(...)`
  *inside the lifespan*, before the first request handler runs.
  Every span Synalinks records is then forwarded to MLflow (or
  whichever sink you wired up). See
  [Observability](Observability.md).

[security]: https://fastapi.tiangolo.com/tutorial/security/

## Concurrency Note

(Safe to skip on a first read — only matters once you train.)

`request.state.program` is *one* Program instance shared across
every concurrent request. The reason it is safe to share is that
serving requests only *reads* the program's **trainable
variables** (the configurable knobs an optimiser tunes during
training — see the [Training](Training.md) guide). The optimiser
is the only thing that ever *writes* to them, and the optimiser
doesn't run at request time. If you ever expose training itself as
an endpoint, give each training job its own program copy so they
don't fight over the same knobs.

## Take-Home Summary

- A `Program` is an async function. **FastAPI deployment** means
  "take an HTTP request off the wire, await the program, put the
  answer back on the wire."
- **DataModels are Pydantic models.** FastAPI uses them directly
  for request bodies, response bodies, and the `/docs` schemas.
  You don't write them twice.
- **Build once, serve forever.** Build the program in a *separate*
  step; the server's only startup job is to load the saved file —
  and to fail fast if the file isn't there.
- **Load the program inside the lifespan, never inside a handler.**
  `yield`ing a `dict` from the lifespan puts the program on
  `request.state` for every request.
- **Use 422 for guard rejections, 502 for off-schema output.** The
  status codes communicate *who* caused the problem; pick them
  honestly.

## What To Learn Next

- [FastMCP Deployment](FastMCP%20Deployment.md) — same shape, but
  the caller is a language model rather than an HTTP client.
- [Observability](Observability.md) — production without tracing
  is debugging in the dark.
- FastAPI's [tutorial][fastapi] for query params, dependencies,
  background tasks, and the wider surface area. (It's good. Use it.)

## API References

- [Program.save / Program.load](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/Program%20Saving%20API/Program%20saving%20and%20loading/)
- [enable_observability](https://synalinks.github.io/synalinks/Synalinks%20API/Observability%20API/)
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
"""

# --8<-- [start:source]
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """The user-facing request body."""

    query: str = synalinks.Field(description="The user query")


class NumericalAnswer(synalinks.DataModel):
    """The user-facing response body."""

    answer: float = synalinks.Field(description="The correct final numerical answer")


# =============================================================================
# Tool used by the agent
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
# FastAPI app -- loads a prepared artifact, fails fast if it is missing
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prepared program; refuse to start if it is missing."""
    if not PROGRAM_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRAM_PATH} not found. Run "
            f"`python -m guides.19_fastapi_deployment build` first."
        )
    program = synalinks.Program.load(str(PROGRAM_PATH))
    # Yielding a dict places its keys on `request.state` for every request.
    yield {"program": program}


app = FastAPI(
    title="math-agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe -- cheap and credential-free."""
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request, query: Query) -> NumericalAnswer:
    """Solve a word problem expressed in natural language."""
    result = await request.state.program(query)
    if result is None:
        # A guard refused; that is an application-level no, not an
        # upstream failure, so 422 is the honest status code.
        raise HTTPException(
            status_code=422,
            detail="Guard rejected input or output",
        )
    # An agent returns its trajectory (`messages`) alongside the answer;
    # mask it out so the strict response model only sees its own fields.
    answer_json = result.out_mask(mask=["messages"]).get_json()
    try:
        return NumericalAnswer.model_validate(answer_json)
    except ValidationError as e:
        # Constrained decoding should prevent this; if it ever fires,
        # the LM produced bad data, so 502 is correct.
        raise HTTPException(
            status_code=502,
            detail=f"Program produced off-schema output: {e}",
        )


# =============================================================================
# Standard FastAPI runner. Equivalent to:
#   fastapi dev guides/19_fastapi_deployment.py
#   uvicorn guides.19_fastapi_deployment:app --reload
# See https://fastapi.tiangolo.com/tutorial/first-steps/.
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import asyncio

        asyncio.run(build_and_save_program(PROGRAM_PATH))
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
