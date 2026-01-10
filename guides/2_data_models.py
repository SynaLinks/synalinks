"""
# Data Models

Data Models are the cornerstone of Synalinks. They define the **contract**
between your application and the Language Model - specifying exactly what
structure your inputs and outputs should have.

## Why Data Models Matter

In traditional LLM development, you send text and receive text. This creates
several problems:

```mermaid
graph LR
    subgraph Without Data Models
        A[Text Prompt] --> B[LLM]
        B --> C[Unstructured Text]
        C --> D[Parse & Hope]
        D --> E[Runtime Errors?]
    end
```

With Synalinks Data Models:

```mermaid
graph LR
    subgraph With Data Models
        A[DataModel Input] --> B[Synalinks]
        B --> C[LLM with Schema]
        C --> D[Validated Output]
        D --> E[DataModel Instance]
    end
```

Data Models provide:

1. **Type Safety**: Know exactly what fields you'll receive
2. **Validation**: Invalid responses are rejected automatically
3. **Documentation**: Field descriptions guide the LLM
4. **IDE Support**: Autocomplete and type checking

## Creating Data Models

A Data Model is a Python class that inherits from `synalinks.DataModel`:

```python
import synalinks
from typing import Literal
from enum import Enum

class Rating(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10

class MovieReview(synalinks.DataModel):
    \"\"\"Analysis of a movie review.\"\"\"

    sentiment: Literal['positive', 'negative', 'neutral'] = synalinks.Field(
        description="The overall sentiment: positive, negative, or neutral"
    )
    key_points: list[str] = synalinks.Field(
        description="Main points mentioned in the review"
    )
    rating: Rating = synalinks.Field(
        description="Estimated rating from 1 to 10"
    )
```

### The Field Function

`synalinks.Field()` is where you communicate with the LLM. The `description`
parameter becomes part of the prompt, telling the model what to generate:

```python
# Good description - specific and actionable
answer: str = synalinks.Field(
    description="A concise answer in 1-2 sentences, based only on the provided context"
)

# Poor description - vague
answer: str = synalinks.Field(
    description="The answer"
)
```

## Supported Types

Synalinks supports these Python types:

| Type | JSON Schema | Example |
|------|-------------|---------|
| `str` | string | `"hello world"` |
| `int` | integer | `42` |
| `float` | number | `3.14` |
| `bool` | boolean | `true` |
| `list[T]` | array | `["a", "b", "c"]` |
| `dict` | object | `{"key": "value"}` |
| `Enum` | enum | Constrained choices |
| `synalinks.Score` | enum | 0.0 to 1.0 in steps |

### Using Enums for Constrained Outputs

When you need the LLM to choose from specific options, use Python Enums (similar to the Literal above):

```python
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskAnalysis(synalinks.DataModel):
    reasoning: str = synalinks.Field(
        description="Why this priority was assigned"
    )
    priority: Priority = synalinks.Field(
        description="The priority level of this task"
    )
```

The LLM is **forced** to output one of the enum values - it cannot hallucinate
invalid options.

### Using synalinks.Score for Confidence

For confidence scores or ratings between 0 and 1, use `synalinks.Score`:

```python
class Analysis(synalinks.DataModel):
    result: str = synalinks.Field(description="The analysis result")
    confidence: synalinks.Score = synalinks.Field(
        description="Confidence in the result (0.0 = uncertain, 1.0 = certain)"
    )
```

`synalinks.Score` is an enum with values: `NONE (0.0)`, `VERY_BAD (0.1)`,
`BAD (0.2)`, ..., `VERY_GOOD (0.9)`, `PERFECT (1.0)`.

## Data Model Operations

Synalinks provides operators for combining and manipulating data models:

```mermaid
graph TD
    A[DataModel A] --> C[Operator]
    B[DataModel B] --> C
    C --> D[Combined DataModel]
```

| Operator | Name | Behavior |
|----------|------|----------|
| `+` | Concat | Merge all fields from both models |
| `&` | And | Merge, but return None if either is None |
| `|` | Or | Return first non-None value for each field |
| `^` | Xor | Return None if both have the same field |
| `~` | Not | Logical negation (for boolean fields) |

### Example: Combining Results

```python
# Two different analysis results
result1 = Analysis1(summary="First analysis", score=0.8)
result2 = Analysis2(details="Additional details", tags=["a", "b"])

# Combine into one data model with all fields
combined = result1.to_json_data_model() + result2.to_json_data_model()
# Result: {summary, score, details, tags}
```

## Masking: Filtering Fields

Sometimes you need to extract or remove specific fields. Use masking operations:

### InMask: Keep Only Specified Fields

```python
# Keep only 'answer' and 'confidence', discard everything else
filtered = await synalinks.ops.in_mask(
    full_result,
    mask=["answer", "confidence"]
)
```

### OutMask: Remove Specified Fields

```python
# Remove 'thinking', keep everything else
filtered = await synalinks.ops.out_mask(
    full_result,
    mask=["thinking"]
)
```

This is particularly useful when:

- You want to hide intermediate reasoning from the final output
- You're training and only want to evaluate certain fields
- You're chaining modules and need to reshape data between them

## Complete Example

```python
import asyncio
from enum import Enum
from dotenv import load_dotenv
import synalinks

# Define an enum for constrained choices
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

# Input data model
class ReviewInput(synalinks.DataModel):
    \"\"\"A product review to analyze.\"\"\"
    review_text: str = synalinks.Field(
        description="The text of the product review"
    )

# Output data model with multiple field types
class ReviewAnalysis(synalinks.DataModel):
    \"\"\"Structured analysis of a review.\"\"\"
    sentiment: Sentiment = synalinks.Field(
        description="The overall sentiment of the review"
    )
    confidence: synalinks.Score = synalinks.Field(
        description="Confidence in the sentiment classification"
    )
    key_points: list[str] = synalinks.Field(
        description="Main points mentioned by the reviewer"
    )
    recommended: bool = synalinks.Field(
        description="Whether the reviewer would recommend the product"
    )

async def main():
    load_dotenv()
    synalinks.clear_session()

    language_model = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Build the program
    inputs = synalinks.Input(data_model=ReviewInput)
    outputs = await synalinks.Generator(
        data_model=ReviewAnalysis,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="review_analyzer",
    )

    # Run analysis
    result = await program(
        ReviewInput(
            review_text="This laptop is amazing! Fast processor, great screen, "
            "and the battery lasts all day. Only complaint is it runs a bit warm. "
            "Would definitely buy again!"
        )
    )

    # Access structured results
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Key Points: {result['key_points']}")
    print(f"Recommended: {result['recommended']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

- **Field Descriptions Are Instructions**: The `description` parameter tells
  the LLM what to generate. Write clear, specific descriptions.

- **Use Enums for Choices**: When the output must be one of several options,
  use Python Enums to constrain the LLM's output.

- **synalinks.Score for Confidence**: Use the built-in Score type for
  confidence values, ratings, or any 0-1 scale.

- **Operators Combine Models**: Use `+`, `&`, `|`, `^` to merge data models
  from different sources or processing branches.

- **Masking Filters Fields**: Use `in_mask` to keep specific fields or
  `out_mask` to remove them.

## API References

- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [JsonDataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20JsonDataModel%20class/)
- [Base DataModels](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
- [JSON Ops](https://synalinks.github.io/synalinks/Synalinks%20API/Ops%20API/JSON%20Ops/)
"""

import asyncio
from enum import Enum

from dotenv import load_dotenv

import synalinks


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewInput(synalinks.DataModel):
    """A product review to analyze."""

    review_text: str = synalinks.Field(description="The text of the product review")


class ReviewAnalysis(synalinks.DataModel):
    """Structured analysis of a review."""

    sentiment: Sentiment = synalinks.Field(
        description="The overall sentiment of the review"
    )
    confidence: synalinks.Score = synalinks.Field(
        description="Confidence in the sentiment classification"
    )
    key_points: list[str] = synalinks.Field(
        description="Main points mentioned by the reviewer"
    )
    recommended: bool = synalinks.Field(
        description="Whether the reviewer would recommend the product"
    )


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_2_data_models",
    )

    language_model = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    inputs = synalinks.Input(data_model=ReviewInput)
    outputs = await synalinks.Generator(
        data_model=ReviewAnalysis,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="review_analyzer",
    )

    result = await program(
        ReviewInput(
            review_text="This laptop is amazing! Fast processor, great screen, "
            "and the battery lasts all day. Only complaint is it runs a bit warm. "
            "Would definitely buy again!"
        )
    )

    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Key Points: {result['key_points']}")
    print(f"Recommended: {result['recommended']}")


if __name__ == "__main__":
    asyncio.run(main())
