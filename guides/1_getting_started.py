"""
# Guide 1: Getting Started with Synalinks

Welcome to Synalinks! This guide will help you get up and running quickly.

## Installation

```bash
# Using pip
pip install synalinks

# Or using uv (recommended)
uv pip install synalinks
```

## Key Concepts

### 1. No Manual Prompting

In Synalinks, you don't write prompts. Instead, define:
- **Input Data Models**: What goes into your program
- **Output Data Models**: What comes out

The framework constructs prompts from your data model definitions.

### 2. Data Models

Use `synalinks.DataModel` with `Field` descriptions to guide the LLM:

```python
class MyOutput(synalinks.DataModel):
    reasoning: str = synalinks.Field(description="Your step by step reasoning")
    result: str = synalinks.Field(description="The final result")
```

### 3. Constrained Structured Output

Synalinks uses constrained structured output to ensure LLM responses
always match your schema. No parsing errors!

### 4. Session Management

Always clear the session at script start for reproducible module naming:

```python
synalinks.clear_session()
```

## Environment Setup

Set up your API keys in a `.env` file:

```bash
# For OpenAI-compatible providers
OPENAI_API_KEY=your_key_here

# For Anthropic
ANTHROPIC_API_KEY=your_key_here

# For local models (Ollama)
# No key needed, just run: ollama serve
```

## Running the Example

```bash
uv run python guides/1_getting_started.py
```
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Your Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User question."""

    query: str = synalinks.Field(description="The user query to answer")


class Answer(synalinks.DataModel):
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step reasoning")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# STEP 2: Build and Run the Program
# =============================================================================


async def main():
    load_dotenv()

    # Clear session for reproducible naming
    synalinks.clear_session()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_1_getting_started",
    )

    # -------------------------------------------------------------------------
    # 2.1: Initialize Language Model
    # -------------------------------------------------------------------------
    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # 2.2: Build Program Using Functional API
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Building Your First Program")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
    )

    # -------------------------------------------------------------------------
    # 2.3: Run the Program
    # -------------------------------------------------------------------------
    print("\nRunning the program...")
    print("-" * 60)

    result = await program(Query(query="What is 2 + 2?"))

    print("\nResult:")
    print(f"Thinking: {result['thinking']}")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # 2.4: Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. NO MANUAL PROMPTING: Define data models, not prompts
2. FIELD DESCRIPTIONS: Guide the LLM with Field(description=...)
3. STRUCTURED OUTPUT: Responses always match your schema
4. CHAIN OF THOUGHT: Add a 'thinking' field to get reasoning
5. CLEAR SESSION: Use synalinks.clear_session() for reproducibility
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
