"""An OpenAI chat-completions-compatible endpoint backed by a Synalinks agent.

`POST /v1/chat/completions` accepts the OpenAI Chat Completions request shape and
returns the OpenAI response shape, so existing OpenAI clients work unchanged by
pointing their base URL at this server:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    client.chat.completions.create(
        model="synalinks-agent",
        messages=[{"role": "user", "content": "What is 12 * 7?"}],
    )

Under the hood the messages are fed to a Synalinks `FunctionCallingAgent` that
can call tools (here, a calculator) before answering.

Build once, serve forever — two steps, never one:

    uv run python main.py build   # 1. build the agent, save it to disk
    uv run python main.py         # 2. serve the saved artifact

Building does not call the LM (it just composes modules and records schemas), so
the build step runs fully offline — no model needed until request time. The
model is read from the MODEL env var (see `.env`); it defaults to vLLM.

See the guide:
https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel
from pydantic import Field

import synalinks

# Override via the MODEL env var (e.g. in `.env` or docker-compose).
DEFAULT_MODEL = "vllm/Qwen/Qwen3-4B"

# Where the prepared program is written/loaded. Override with PROGRAM_PATH.
PROGRAM_PATH = Path(os.environ.get("PROGRAM_PATH", "program.json"))


def _enable_observability() -> None:
    """Enable MLflow tracing when MLFLOW_TRACKING_URI is set.

    Must run before any module is built/loaded. No-op when unset, so the app
    runs fine without an MLflow server.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        synalinks.enable_observability(
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "synalinks_traces"),
        )


# =============================================================================
# Tool the agent can call
# =============================================================================


# The decorator registers `calculate` under its qualified name so the agent can
# serialize/restore the tool reference.
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
    """Build the chat agent and persist it to ``path``.

    Run this as a *separate* step (CLI verb, CI job, REPL). It builds a
    FunctionCallingAgent over ``ChatMessages`` and saves it — no LM call
    happens here, so it works offline (e.g. at Docker image-build time).
    """
    synalinks.clear_session()
    lm = synalinks.LanguageModel(model=os.environ.get("MODEL", DEFAULT_MODEL))

    inputs = synalinks.Input(data_model=synalinks.ChatMessages)
    outputs = await synalinks.FunctionCallingAgent(
        tools=[synalinks.Tool(calculate)],
        language_model=lm,
    )(inputs)
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="chat_agent",
        description="A tool-using chat agent.",
    )
    program.save(str(path))
    return program


# =============================================================================
# OpenAI-compatible request / response schemas (a minimal, useful subset)
# =============================================================================


class Message(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    # Accepted for client compatibility; this starter server ignores them.
    stream: Optional[bool] = False
    temperature: Optional[float] = None


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# FastAPI app
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prepared agent once; refuse to start if it is missing."""
    load_dotenv()
    _enable_observability()  # before loading any module
    if not PROGRAM_PATH.exists():
        raise FileNotFoundError(
            f"{PROGRAM_PATH} not found. Build it first: "
            f"`uv run python main.py build`."
        )
    app.state.agent = synalinks.Program.load(str(PROGRAM_PATH))
    yield


app = FastAPI(title="synalinks-chat-api", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — cheap and credential-free."""
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completion, served by a Synalinks agent."""
    if body.stream:
        # Streaming is out of scope for this starter; clients should disable it.
        raise HTTPException(
            status_code=400, detail="This server does not support stream=true"
        )

    chat_messages = synalinks.ChatMessages(
        messages=[{"role": m.role, "content": m.content or ""} for m in body.messages]
    )
    result = await request.app.state.agent(chat_messages)
    if result is None:
        # A Synalinks guard refused the call — application-level "no" (422).
        raise HTTPException(status_code=422, detail="Guard rejected input or output")

    # The agent returns its full trajectory; the last message is the answer.
    messages = result.get("messages") or []
    final = messages[-1] if messages else {}
    content = final.get("content") if hasattr(final, "get") else None

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=body.model or os.environ.get("MODEL", DEFAULT_MODEL),
        choices=[Choice(message=Message(role="assistant", content=content or ""))],
    )


# Standard FastAPI runner. `build` is a sibling verb (see the module docstring).
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import asyncio

        load_dotenv()
        asyncio.run(build_and_save_program(PROGRAM_PATH))
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
