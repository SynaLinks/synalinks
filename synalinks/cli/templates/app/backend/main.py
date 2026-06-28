"""The app backend: a Synalinks program served as a REST API with FastAPI.

Build once, serve forever — two steps, never one:

    uv run python main.py build   # 1. build the program, save it to disk
    uv run python main.py         # 2. serve the saved artifact

(or, while developing: `fastapi dev main.py`).

The model is read from the MODEL env var (see `.env`); it defaults to vLLM.
NOTE: MODEL is baked into the saved program at *build* time, so set it before
`build` — changing it at serve time has no effect (rebuild to switch models).
No GPU? Set MODEL=ollama/mistral:latest. To use a hosted model,
`cp .env.template .env`, add the matching key, and set MODEL there.

See the guide:
https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic import ValidationError

import synalinks

# Used at build time; override via the MODEL env var (e.g. in `.env`).
DEFAULT_MODEL = "vllm/Qwen/Qwen3-4B"


def _enable_observability() -> None:
    """Enable MLflow tracing when MLFLOW_TRACKING_URI is set.

    Must run before any module is built/loaded. No-op when unset, so the app
    runs fine without an MLflow server.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        synalinks.enable_observability(
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "synalinks_traces"),
        )


# Where the prepared program is written/loaded. Override with PROGRAM_PATH.
PROGRAM_PATH = Path(os.environ.get("PROGRAM_PATH", "program.json"))


# Structured I/O — DataModels are Pydantic models, so FastAPI reuses them as
# request/response bodies and publishes their schemas at /docs for free.
class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="The question to answer")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer to the question")


async def build_and_save_program(path: Path) -> "synalinks.Program":
    """Build the program once and persist it to ``path``.

    Run this as a *separate* step (CLI verb, CI job, REPL) — never from the
    server lifespan. A production server should load a known artifact, not
    construct one (and maybe fail) on its first request.
    """
    synalinks.clear_session()
    lm = synalinks.LanguageModel(model=os.environ.get("MODEL", DEFAULT_MODEL))

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa",
        description="Answers a question.",
    )
    program.save(str(path))
    return program


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prepared program once; refuse to start if it is missing."""
    load_dotenv()
    _enable_observability()  # before loading any module
    if not PROGRAM_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRAM_PATH} not found. Build it first: "
            f"`uv run python main.py build`."
        )
    program = synalinks.Program.load(str(PROGRAM_PATH))
    # Keys of the yielded dict land on `request.state` for every request,
    # so handlers reach the program without a global variable.
    yield {"program": program}


app = FastAPI(title="synalinks-app-backend", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — cheap and credential-free."""
    return {"status": "ok"}


@app.post("/answer")
async def answer(request: Request, question: Question) -> Answer:
    """Answer a question."""
    result = await request.state.program(question)
    if result is None:
        # A Synalinks guard refused the call — an application-level "no",
        # not an upstream failure, so 422 is the honest status code.
        raise HTTPException(status_code=422, detail="Guard rejected input or output")
    try:
        return Answer.model_validate(result.get_json())
    except ValidationError as e:
        # Constrained decoding should prevent this; if it ever fires the LM
        # produced off-schema output, which is an upstream (502) problem.
        raise HTTPException(
            status_code=502, detail=f"Program produced off-schema output: {e}"
        )


# Standard FastAPI runner. Equivalent to `fastapi dev main.py` /
# `uvicorn main:app`. `build` is a sibling verb (see the module docstring).
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import asyncio

        load_dotenv()
        asyncio.run(build_and_save_program(PROGRAM_PATH))
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
