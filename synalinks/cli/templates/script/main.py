"""A minimal Synalinks program — your starting point.

Run it:

    uv sync
    uv run python main.py

The default MODEL is local Ollama (no API key). To use a hosted model,
`cp .env.template .env`, add the matching key, and edit MODEL below.
"""

import asyncio

from dotenv import load_dotenv

import synalinks

MODEL = "ollama/llama3.2:latest"


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
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model=MODEL)
    program = await build_program(lm)

    result = await program(Question(question="What is the capital of France?"))
    print(result.get_json())


if __name__ == "__main__":
    asyncio.run(main())
