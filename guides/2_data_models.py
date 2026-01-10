"""
# Guide 2: Data Models

Data Models are the foundation of Synalinks. They define the structure of
inputs and outputs for your programs.

## Creating Data Models

Inherit from `synalinks.DataModel` and use `synalinks.Field` for descriptions:

```python
class UserQuery(synalinks.DataModel):
    query: str = synalinks.Field(description="The user's question to answer")
    context: str = synalinks.Field(description="Optional context for the question")
```

## Field Descriptions

Field descriptions are critical - they guide the LLM on what to generate.
The description becomes part of the prompt sent to the LLM.

## Supported Types

Synalinks supports these Python types:

- `str` - Strings
- `int` - Integers
- `float` - Numbers
- `bool` - Booleans
- `list[T]` - Arrays of type T
- `dict` - Objects (stored as JSON)
- `synalinks.Score` - Constrained float between 0.0 and 1.0 (Enum)

## Data Model Operations

Synalinks supports Python operators for combining data models:

| Operator | Module | Description |
|----------|--------|-------------|
| `+` | `Concat` | Merge fields (raises if either None) |
| `&` | `And` | Merge with None if either is None |
| `|` | `Or` | Merge with None fallback |
| `^` | `Xor` | None if both present |
| `~` | `Not` | Logical negation |

## Accessing Data

Access fields using dictionary syntax:

```python
result = await program(Query(query="What is Python?"))
answer = result['answer']
```

## Running the Example

```bash
uv run python guides/2_data_models.py
```
"""

import asyncio
from enum import Enum

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Various Data Model Types
# =============================================================================


class Query(synalinks.DataModel):
    """A user's question."""

    query: str = synalinks.Field(description="The user's question to answer")


class DetailedAnswer(synalinks.DataModel):
    """An answer with reasoning and confidence."""

    thinking: str = synalinks.Field(description="Step-by-step reasoning process")
    answer: str = synalinks.Field(description="The final answer based on reasoning")
    confidence: synalinks.Score = synalinks.Field(
        description="Confidence score from 0.0 to 1.0"
    )


# Using Enums for constrained choices
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentAnalysis(synalinks.DataModel):
    """Sentiment analysis result."""

    sentiment: Sentiment = synalinks.Field(description="The detected sentiment")
    explanation: str = synalinks.Field(
        description="Explanation for the sentiment classification"
    )


class CodeReview(synalinks.DataModel):
    """Code review output with lists."""

    issues: list[str] = synalinks.Field(
        description="List of potential issues found in the code"
    )
    suggestions: list[str] = synalinks.Field(
        description="Improvement suggestions for the code"
    )
    quality_score: int = synalinks.Field(
        description="Code quality score from 1 (poor) to 10 (excellent)"
    )


# =============================================================================
# STEP 2: Demonstrate Data Model Features
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_2_data_models",
    )

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # 2.1: Basic Data Model with Confidence Score
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Data Model with Confidence Score")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=DetailedAnswer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="detailed_qa",
    )

    result = await program(Query(query="What is the capital of France?"))

    print(f"\nThinking: {result['thinking']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")

    # -------------------------------------------------------------------------
    # 2.2: Using Enums for Constrained Outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Enum-constrained Sentiment Analysis")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=SentimentAnalysis,
        language_model=lm,
    )(inputs)

    sentiment_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="sentiment_analyzer",
    )

    result = await sentiment_program(
        Query(query="I love this product! It exceeded all my expectations.")
    )

    print(f"\nSentiment: {result['sentiment']}")
    print(f"Explanation: {result['explanation']}")

    # -------------------------------------------------------------------------
    # 2.3: Data Model Operators
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: Data Model Operators")
    print("=" * 60)

    # Create two data models
    dm1 = Query(query="Hello")
    dm2 = DetailedAnswer(
        thinking="Simple greeting",
        answer="Hi there!",
        confidence=synalinks.Score.VERY_GOOD,
    )

    # Concatenate with + operator
    combined = dm1.to_json_data_model() + dm2.to_json_data_model()

    print(f"\nCombined fields: {list(combined.get_json().keys())}")
    print(f"Query: {combined['query']}")
    print(f"Answer: {combined['answer']}")

    # -------------------------------------------------------------------------
    # 2.4: Masking Fields
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: Masking Fields")
    print("=" * 60)

    # InMask - keep only specified fields
    masked_in = await synalinks.InMask(keys=["answer", "confidence"])(
        dm2.to_json_data_model()
    )
    print(f"\nInMask (keep answer, confidence): {list(masked_in.get_json().keys())}")

    # OutMask - remove specified fields
    masked_out = await synalinks.OutMask(keys=["thinking"])(dm2.to_json_data_model())
    print(f"OutMask (remove thinking): {list(masked_out.get_json().keys())}")

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. FIELD DESCRIPTIONS: They become instructions for the LLM
2. USE ENUMS: Constrain outputs to valid choices
3. USE synalinks.Score: For confidence values between 0.0 and 1.0
4. OPERATORS: Combine data models with +, &, |, ^, ~
5. MASKING: Extract or exclude fields with InMask/OutMask
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
