"""
# Getting Started with Synalinks

Welcome to Synalinks! This guide introduces you to the fundamental concepts
and helps you build your first AI-powered application.

## What is Synalinks?

Synalinks is a **neuro-symbolic framework** for building Language Model (LM)
applications. Unlike traditional approaches where you write prompts manually,
Synalinks lets you define **what** you want (using data models) and handles
the **how** (prompt construction, parsing, validation) automatically.

```mermaid
graph LR
    subgraph Traditional Approach
        A[Write Prompt] --> B[Call LLM API]
        B --> C[Parse Response]
        C --> D[Handle Errors]
    end
    subgraph Synalinks Approach
        E[Define DataModel] --> F[Synalinks]
        F --> G[Structured Output]
    end
```

## Why Synalinks?

Traditional LLM development has several pain points:

1. **Prompt Engineering**: Manually crafting prompts is tedious and error-prone
2. **Output Parsing**: LLM responses are unstructured text that needs parsing
3. **Schema Validation**: Ensuring responses match expected formats is difficult
4. **Reproducibility**: Results vary based on prompt wording

Synalinks solves these by:

- **Auto-generating prompts** from your data model definitions
- **Constraining outputs** to always match your schema (no parsing errors!)
- **Providing type safety** through Pydantic-based data models
- **Enabling training** to improve your programs over time

## Installation

```bash
# Using pip
pip install synalinks

# Or using uv (recommended for faster installs)
uv pip install synalinks
```

## Environment Setup

Synalinks works with any OpenAI-compatible API. Set up your credentials:

```bash
# Create a .env file in your project root
OPENAI_API_KEY=your-api-key-here

# For Anthropic models
ANTHROPIC_API_KEY=your-anthropic-key

# For local models (Ollama) - no key needed
# Just run: ollama serve
```

## Core Concepts

### 1. Data Models: The Foundation

In Synalinks, everything revolves around **Data Models**. A DataModel defines
the structure of your inputs and outputs using Python classes:

```python
import synalinks

class Question(synalinks.DataModel):
    \"\"\"The input to our program.\"\"\"
    question: str = synalinks.Field(
        description="The question to answer"
    )

class Answer(synalinks.DataModel):
    \"\"\"The output from our program.\"\"\"
    thinking: str = synalinks.Field(
        description="Step-by-step reasoning process"
    )
    answer: str = synalinks.Field(
        description="The final answer"
    )
```

The `description` parameter is crucial - it tells the LLM what each field
should contain. Think of it as documentation that the AI reads.

### 2. The Generator Module

The `Generator` is the core module that transforms inputs into outputs:

```python
generator = synalinks.Generator(
    data_model=Answer,           # What structure to output
    language_model=language_model,  # Which LLM to use
)
```

### 3. Programs: Composable Pipelines

A `Program` wraps your modules into a reusable, trainable unit:

```python
program = synalinks.Program(
    inputs=inputs,
    outputs=outputs,
    name="my_program",
)
```

## Complete Example

Here's a complete, runnable example that demonstrates the core concepts:

```python
import asyncio
from dotenv import load_dotenv
import synalinks

# Step 1: Define your data models
class Question(synalinks.DataModel):
    \"\"\"Input: A question from the user.\"\"\"
    question: str = synalinks.Field(
        description="The question to answer"
    )

class Answer(synalinks.DataModel):
    \"\"\"Output: An answer with reasoning.\"\"\"
    thinking: str = synalinks.Field(
        description="Your step-by-step reasoning process"
    )
    answer: str = synalinks.Field(
        description="The final answer based on your reasoning"
    )

async def main():
    # Load environment variables (API keys)
    load_dotenv()

    # Clear session for reproducible module naming
    synalinks.clear_session()

    # Step 2: Initialize the language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1-mini"  # Or "anthropic/claude-3-haiku", etc.
    )

    # Step 3: Build the program using the Functional API
    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
        description="A simple question-answering program",
    )

    # Step 4: Run the program
    result = await program(
        Question(question="What is the capital of France?")
    )

    # Step 5: Access the structured output
    print(f"Thinking: {result['thinking']}")
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding the Output

When you run the program, you get a **structured dictionary** that exactly
matches your `Answer` data model:

```python
{
    "thinking": "France is a country in Western Europe. Its capital city...",
    "answer": "Paris"
}
```

This is guaranteed by Synalinks' constrained generation - the LLM is forced
to produce valid JSON that matches your schema.

## Key Takeaways

- **No Manual Prompting**: Define data models instead of writing prompts.
  Synalinks automatically constructs effective prompts from your field
  descriptions.

- **Structured Output**: Every response is guaranteed to match your schema.
  No more parsing errors or malformed responses.

- **Field Descriptions Matter**: The `description` parameter guides the LLM.
  Write clear, specific descriptions for best results.

- **Chain of Thought**: Adding a "thinking" field encourages the LLM to
  reason step-by-step, improving accuracy on complex tasks.

- **Session Management**: Use `synalinks.clear_session()` at the start of
  scripts for reproducible module naming.

## Next Steps

Now that you understand the basics, continue to:

- [Guide 2: Data Models](Data%20Models.md) - Deep dive into data model features
- [Guide 3: Programs](Programs.md) - Learn about different program architectures
- [Guide 4: Modules](Modules.md) - Explore the available modules

## API References

- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


class Question(synalinks.DataModel):
    """Input: A question from the user."""

    question: str = synalinks.Field(description="The question to answer")


class Answer(synalinks.DataModel):
    """Output: An answer with reasoning."""

    thinking: str = synalinks.Field(
        description="Your step-by-step reasoning process"
    )
    answer: str = synalinks.Field(
        description="The final answer based on your reasoning"
    )


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_1_getting_started",
    )

    language_model = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
        description="A simple question-answering program",
    )

    result = await program(Question(question="What is the capital of France?"))

    print(f"Thinking: {result['thinking']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
