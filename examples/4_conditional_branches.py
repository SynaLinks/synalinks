"""
# Lesson 4: Conditional Branches

In Lesson 3, you learned to make decisions. Now, let's use those decisions
to create **conditional branches** - programs that take different paths
based on the input.

## Why Conditional Branches?

Real-world applications often need different processing for different cases:
- Simple questions → Quick answer
- Complex questions → Detailed reasoning
- Urgent tickets → Fast track processing
- Normal tickets → Standard queue

## How Conditional Branches Work

The `Branch` module combines **decision-making** with **conditional execution**:

```mermaid
graph LR
    Input --> Decision
    Decision -->|easy| A[Branch A]
    Decision -->|medium| B[Branch B]
    Decision -->|hard| C[Branch C]
    A --> Outputs
    B --> Outputs
    C --> Outputs
```

Only ONE branch executes - the others output `None`.

```python
(easy_result, hard_result) = await synalinks.Branch(
    question="How complex is this query?",
    labels=["easy", "hard"],
    branches=[
        synalinks.Generator(data_model=SimpleAnswer, ...),  # For "easy"
        synalinks.Generator(data_model=DetailedAnswer, ...), # For "hard"
    ],
    language_model=language_model,
)(inputs)

# If query was "easy": easy_result has data, hard_result is None
# If query was "hard": easy_result is None, hard_result has data
```

## Handling None Outputs

Since unselected branches return `None`, you can use logical operators
to merge results (covered in Lesson 5):

```python
final_result = easy_result | hard_result  # Gets the non-None result
```

## Running the Example

```bash
uv run python examples/4_conditional_branches.py
```

## Program Visualization

![conditional_branches](../assets/examples/conditional_branches.png)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """A user query to process."""

    query: str = synalinks.Field(
        description="The user query",
    )


class Answer(synalinks.DataModel):
    """A simple, direct answer (for easy questions)."""

    answer: str = synalinks.Field(
        description="The correct answer",
    )


class AnswerWithThinking(synalinks.DataModel):
    """A detailed answer with reasoning (for hard questions)."""

    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )


# =============================================================================
# STEP 2: Build and Run the Program
# =============================================================================


async def main():
    load_dotenv()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="lesson_4_conditional_branches",
    )

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # -------------------------------------------------------------------------
    # 2.1: Create Conditional Branches
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Building a Program with Conditional Branches")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Create a Branch module that routes to different generators
    # based on the decision result
    (easy_branch, hard_branch) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],  # Maps to branches by index
        language_model=language_model,  # Required for the internal decision
        branches=[
            # Branch 0: For "easy" queries - simple answer
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
                name="easy_answerer",
            ),
            # Branch 1: For "difficult" queries - detailed reasoning
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
                name="hard_answerer",
            ),
        ],
    )(inputs)

    # Create program with both outputs
    # One will be None depending on which branch was taken
    program = synalinks.Program(
        inputs=inputs,
        outputs=[easy_branch, hard_branch],
        name="conditional_branches",
        description="Routes queries to different handlers based on difficulty",
    )

    # Generate visualization
    synalinks.utils.plot_program(
        program,
        to_folder="examples",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    # -------------------------------------------------------------------------
    # 2.2: Test with Different Queries
    # -------------------------------------------------------------------------
    print("\nTesting conditional branches...")
    print("-" * 60)

    # Test with an easy query
    easy_query = Query(query="What is 2 + 2?")
    results = await program(easy_query)
    print(f"\nQuery: '{easy_query.query}'")
    print(f"Easy branch result: {results[0]}")
    print(f"Hard branch result: {results[1]}")

    # Test with a difficult query
    hard_query = Query(
        query="Explain quantum entanglement and its implications for computing"
    )
    results = await program(hard_query)
    print(f"\nQuery: '{hard_query.query}'")
    print(f"Easy branch result: {results[0]}")
    if results[1]:
        print(f"Hard branch result: {results[1]['answer'][:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
