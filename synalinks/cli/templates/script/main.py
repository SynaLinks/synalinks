"""A minimal Synalinks program — your starting point.

Run it:

    uv sync
    uv run python main.py

The default MODEL is served by vLLM — start it with `vllm serve Qwen/Qwen3-4B`.
To use another model, edit MODEL below (and `cp .env.template .env` for keys /
endpoints). Set MLFLOW_TRACKING_URI to trace runs to MLflow.
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks

MODEL = "vllm/Qwen/Qwen3-4B"


def _enable_observability():
    """Enable MLflow tracing when MLFLOW_TRACKING_URI is set (no-op otherwise)."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        synalinks.enable_observability(
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "synalinks_traces"),
        )


# Structured input/output — every Synalinks module reads and writes DataModels.
class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="The question to answer")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer to the question")


async def build_program(lm):
    """A one-module program: question in, answer out."""
    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)
    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa",
        description="Answers a question.",
    )


async def main():
    load_dotenv()
    # vLLM's OpenAI endpoint lives at /v1; set a correct default (overridable
    # via .env / real env). Synalinks' own fallback omits /v1, which 404s.
    os.environ.setdefault("HOSTED_VLLM_API_BASE", "http://localhost:8000/v1")
    _enable_observability()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model=MODEL)
    program = await build_program(lm)

    result = await program(Question(question="What is the capital of France?"))
    print(result.get_json())


if __name__ == "__main__":
    asyncio.run(main())
