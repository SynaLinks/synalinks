"""
# Lesson 3: Making Decisions

Sometimes your AI needs to make a choice: Is this email spam or not? Is this
query simple or complex? Should we route to department A or B?

This lesson introduces the **Decision** module - a powerful way to classify
inputs into predefined categories.

## What is a Decision?

A Decision is essentially **single-label classification**. You provide:
1. A **question** - what you want to decide
2. **Labels** - the possible choices

The LLM will pick exactly ONE label from your choices.

```python
decision = await synalinks.Decision(
    question="Is this email spam?",
    labels=["spam", "not_spam"],
    language_model=language_model,
)(email_input)

print(decision["choice"])  # Either "spam" or "not_spam"
```

## How It Works

Behind the scenes, Synalinks:
1. Creates an **Enum schema** from your labels
2. Uses **constrained generation** to force the LLM to pick one label
3. Returns a structured output with `thinking` and `choice` fields

This guarantees you get exactly one of your labels - no ambiguity!

## Use Cases

- **Routing**: Send queries to different handlers based on type
- **Filtering**: Spam detection, content moderation
- **Triage**: Prioritize tasks by urgency
- **Classification**: Categorize documents, tickets, etc.

## Running the Example

```bash
uv run python examples/3_decisions.py
```

## Program Visualization

![decision_making](../assets/examples/decision_making.png)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """A user query to classify."""

    query: str = synalinks.Field(
        description="The user query",
    )


# =============================================================================
# STEP 2: Build and Run the Program
# =============================================================================


async def main():
    load_dotenv()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="lesson_3_decisions",
    )

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # -------------------------------------------------------------------------
    # 2.1: Create a Decision Module
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Building a Decision-Making Program")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Create a Decision module that classifies query difficulty
    # The LLM will output EXACTLY one of the labels
    outputs = await synalinks.Decision(
        question="Evaluate the difficulty to answer the provided query",
        labels=["easy", "difficult"],  # Only these two choices!
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="decision_making",
        description="Classifies queries by difficulty",
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
    # 2.2: Test the Decision Module
    # -------------------------------------------------------------------------
    print("\nTesting with different queries...")
    print("-" * 60)

    # Test with an easy query
    easy_query = Query(query="What is 2 + 2?")
    result = await program(easy_query)
    print(f"\nQuery: '{easy_query.query}'")
    print(f"Thinking: {result['thinking'][:60]}...")
    print(f"Choice: {result['choice']}")

    # Test with a difficult query
    hard_query = Query(
        query="Explain the implications of Gödel's incompleteness theorems"
    )
    result = await program(hard_query)
    print(f"\nQuery: '{hard_query.query}'")
    print(f"Thinking: {result['thinking'][:60]}...")
    print(f"Choice: {result['choice']}")


if __name__ == "__main__":
    asyncio.run(main())
