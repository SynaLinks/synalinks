"""
# First Steps with Synalinks

Welcome to Synalinks! This lesson covers the essential concepts you need
to understand before building AI applications.

## Installation

```bash
# Using pip
pip install synalinks

# Or using uv (recommended)
uv pip install synalinks
```

## Key Concepts

### 1. No Traditional Prompting

In Synalinks, you don't write prompts manually. Instead, you define:

- **Input Data Models**: What data goes into your program
- **Output Data Models**: What data comes out

```mermaid
graph LR
    A[Input DataModel] --> B[Synalinks]
    B --> C[Auto-Generated Prompt]
    C --> D[LLM]
    D --> E[Output DataModel]
```

The framework automatically constructs prompts from your data model definitions.

### 2. Data Models and Fields

Data models define the structure of your inputs and outputs. Use `Field` to
add descriptions that help the LLM understand what each field should contain:

```python
class Answer(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step reasoning"
    )
    answer: str = synalinks.Field(
        description="The final answer"
    )
```

### 3. Constrained Structured Output

Synalinks uses **constrained structured output** to ensure LLM responses
always match your data model specification. No parsing errors!

### 4. Session Management

Always clear the session at the start of scripts to ensure reproducible
module naming:

```python
synalinks.clear_session()
```

## Building a Simple Program

Here's a complete example that creates a question-answering program:

```python
import asyncio
from dotenv import load_dotenv
import synalinks

# Define input data model
class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query to answer")

# Define output data model with chain-of-thought
class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Your step by step thinking process")
    answer: str = synalinks.Field(description="The correct answer based on your thinking")

async def main():
    load_dotenv()
    synalinks.clear_session()

    # Initialize a language model
    language_model = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Build the program using the Functional API
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="chain_of_thought_qa",
    )

    # Run the program
    result = await program(Query(query="What is 2 + 2?"))
    print(f"Thinking: {result['thinking']}")
    print(f"Answer: {result['answer']}")

asyncio.run(main())
```

By adding a `thinking` field to our output model, we instruct the LLM to show
its reasoning - this is called "Chain of Thought" prompting, achieved simply
by defining the output structure!

### Key Takeaways

- **No Prompt Engineering**: Define data models instead of writing prompts -
    the framework generates prompts automatically from your schemas.
- **Structured Output**: All LLM responses are guaranteed to match your
    data model specification through constrained generation.
- **Field Descriptions**: Use descriptive `Field` annotations to guide the
    LLM on what each field should contain.
- **Chain of Thought**: Add a "thinking" field to your output model to get
    step-by-step reasoning from the LLM.

## API References

- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Input](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Input%20module/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Check Installation and Clear Session
# =============================================================================


def setup():
    """Setup Synalinks for use."""
    # Check version
    print(f"Synalinks version: {synalinks.__version__}")

    # Clear the global context for reproducible naming
    # This ensures modules get consistent names across runs
    synalinks.clear_session()


# =============================================================================
# STEP 2: Define Data Models
# =============================================================================
# Data models are the core of Synalinks. They define:
# - The structure of data flowing through your program
# - Instructions to the LLM via field descriptions


class Query(synalinks.DataModel):
    """The input to our program - a user's question.

    The docstring becomes part of the schema description.
    """

    query: str = synalinks.Field(
        description="The user query to answer",
    )


class Answer(synalinks.DataModel):
    """A simple answer from the LLM."""

    answer: str = synalinks.Field(
        description="The correct answer to the query",
    )


class AnswerWithThinking(synalinks.DataModel):
    """An answer with step-by-step reasoning.

    By adding a 'thinking' field, we instruct the LLM to show its work.
    This is called "Chain of Thought" prompting - but we achieve it
    simply by defining the output structure!
    """

    thinking: str = synalinks.Field(
        description="Your step by step thinking process",
    )
    answer: str = synalinks.Field(
        description="The correct answer based on your thinking",
    )


# =============================================================================
# STEP 3: Understand the Prompt Template
# =============================================================================


def show_prompt_template():
    """Display the default prompt template."""
    print("=" * 60)
    print("Default Prompt Template")
    print("=" * 60)
    print()
    print("Synalinks automatically constructs prompts using this template:")
    print()
    print(synalinks.default_prompt_template())
    print()
    print("-" * 60)
    print("The template uses Markdown headers for structure.")
    print("Your data model schemas are automatically inserted!")
    print()


# =============================================================================
# STEP 4: Build and Run a Simple Program
# =============================================================================


async def main():
    load_dotenv()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="lesson_0_first_steps",
    )

    # Setup
    setup()

    # Show the prompt template
    show_prompt_template()

    # Initialize a language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1-mini",
    )

    # -------------------------------------------------------------------------
    # Example 1: Simple Answer (no thinking)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Simple Answer")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)

    simple_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_qa",
    )

    result = await simple_program(Query(query="What is the capital of France?"))
    print("\nQuery: What is the capital of France?")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Example 2: Answer with Thinking (Chain of Thought)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Answer with Thinking (Chain of Thought)")
    print("=" * 60)
    print("Just by adding a 'thinking' field to our output model,")
    print("the LLM will show its reasoning!\n")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)

    cot_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="chain_of_thought_qa",
    )

    result = await cot_program(
        Query(query="If I have 3 apples and give away 1, how many do I have?")
    )
    print("Query: If I have 3 apples and give away 1, how many do I have?")
    print(f"\nThinking: {result['thinking']}")
    print(f"\nAnswer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print("""
1. NO MANUAL PROMPTING: Define data models, not prompts
2. FIELD DESCRIPTIONS: Guide the LLM with Field(description=...)
3. STRUCTURED OUTPUT: Responses always match your schema
4. CHAIN OF THOUGHT: Add a 'thinking' field to get reasoning
5. CLEAR SESSION: Use synalinks.clear_session() for reproducibility
""")


if __name__ == "__main__":
    asyncio.run(main())
