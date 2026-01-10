"""
# Lesson 5a: Data Model Operators

In previous lessons, you learned to create branches where some outputs can be
`None`. How do we combine these outputs? This lesson introduces **operators**
for merging and manipulating data models.

## The Five Operators

Synalinks provides five Python operators for data models:

| Operator | Name | Behavior |
|----------|------|----------|
| `+` | Concatenation | Merge fields (fails if either is None) |
| `&` | Logical AND | Safe merge (returns None if either is None) |
| `|` | Logical OR | Returns first non-None, or merges if both exist |
| `^` | Logical XOR | Returns data only if exactly ONE is non-None |
| `~` | NOT (Invert) | Converts data to None |

## 1. Concatenation (`+`)

Combines fields from two data models into one:

```python
answer = Answer(answer="42")           # {"answer": "42"}
thinking = Thinking(thinking="...")    # {"thinking": "..."}

combined = answer + thinking  # {"answer": "42", "thinking": "..."}
```

**Warning**: Raises an exception if either operand is `None`!

## 2. Logical AND (`&`)

Safe concatenation - returns `None` if either input is `None`:

```python
result = data & possibly_none  # None if possibly_none is None
```

## 3. Logical OR (`|`)

Returns non-None value, or merges if both have data:

```python
(easy_result, hard_result) = await Branch(...)(inputs)
final = easy_result | hard_result  # Gets whichever has data
```

## 4. Logical XOR (`^`)

Returns data only if **exactly one** operand is non-None:

```python
result = a ^ b
# If a has data and b is None: returns a
# If a is None and b has data: returns b
# If both have data: returns None
# If both are None: returns None
```

## 5. NOT / Invert (`~`)

Converts any data to `None`:

```python
result = ~data  # Always returns None
```

Useful for conditional flows where you want to "cancel" a path.

## Truth Table

| A | B | A + B | A & B | A | B | A ^ B |
|---|---|-------|-------|--------|-------|
| Data | Data | Merged | Merged | Merged | None |
| Data | None | ERROR | None | A | A |
| None | Data | ERROR | None | B | B |
| None | None | ERROR | None | None | None |

| A | ~A |
|---|----|
| Data | None |
| None | None* |

*Note: ~None behavior depends on context (symbolic vs json)

## Running the Example

```bash
uv run python examples/5a_data_model_operators.py
```

## Program Visualizations

![concatenation](../assets/examples/concatenation.png)
![logical_or](../assets/examples/logical_or.png)
![logical_and](../assets/examples/logical_and.png)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """A user query."""

    query: str = synalinks.Field(description="The user query")


class Answer(synalinks.DataModel):
    """A simple answer."""

    answer: str = synalinks.Field(description="The correct answer")


class AnswerWithThinking(synalinks.DataModel):
    """An answer with reasoning."""

    thinking: str = synalinks.Field(description="Your step by step thinking")
    answer: str = synalinks.Field(description="The correct answer")


class Critique(synalinks.DataModel):
    """A critique of an answer."""

    critique: str = synalinks.Field(description="The critique of the answer")


# =============================================================================
# STEP 2: Demonstrate the Operators
# =============================================================================


async def main():
    load_dotenv()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="lesson_5a_data_model_operators",
    )

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # =========================================================================
    # EXAMPLE 1: Concatenation Operator (+)
    # =========================================================================
    print("=" * 60)
    print("Example 1: Concatenation Operator (+)")
    print("=" * 60)
    print("Combines fields from two data models into one.\n")

    inputs = synalinks.Input(data_model=Query)

    # Two parallel generators
    answer_1 = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        name="answer_generator",
    )(inputs)
    critique_1 = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        name="critique_generator",
    )(inputs)

    # Concatenate results - merges all fields
    outputs = answer_1 + critique_1

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="concatenation",
        description="Demonstrates field concatenation",
    )

    synalinks.utils.plot_program(program, to_folder="examples", show_module_names=True)

    # Test it
    result = await program(Query(query="What is 2 + 2?"))
    print("Combined result fields:", list(result.keys()))
    print(f"  answer: {result['answer'][:50]}...")
    print(f"  critique: {result['critique'][:50]}...")

    # =========================================================================
    # EXAMPLE 2: Logical OR Operator (|) - Merging Branch Outputs
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Logical OR Operator (|)")
    print("=" * 60)
    print("Returns the first non-None value. Perfect for branches!\n")

    inputs = synalinks.Input(data_model=Query)

    # Branch returns (easy_result, hard_result) - only ONE is non-None
    (easy, hard) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the query",
        labels=["easy", "difficult"],
        language_model=language_model,
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
            ),
        ],
        return_decision=False,
    )(inputs)

    # Use | to get whichever branch was active
    outputs = easy | hard

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="logical_or",
        description="Demonstrates merging branch outputs",
    )

    synalinks.utils.plot_program(program, to_folder="examples", show_module_names=True)

    # Test with easy query
    result = await program(Query(query="What is 1 + 1?"))
    print("Easy query result:", result["answer"][:50] if result else "None")

    # =========================================================================
    # EXAMPLE 3: Logical AND Operator (&) - Safe Concatenation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Logical AND Operator (&)")
    print("=" * 60)
    print("Safe merge - returns None if either input is None.\n")

    inputs = synalinks.Input(data_model=Query)

    # Branch with potentially None outputs
    (easy, hard) = await synalinks.Branch(
        question="Evaluate the difficulty to answer the query",
        labels=["easy", "difficult"],
        language_model=language_model,
        branches=[
            synalinks.Generator(data_model=Answer, language_model=language_model),
            synalinks.Generator(
                data_model=AnswerWithThinking, language_model=language_model
            ),
        ],
        return_decision=False,
    )(inputs)

    # Safe concatenation with inputs - won't fail if branch output is None
    easy_with_query = inputs & easy
    hard_with_query = inputs & hard

    # Add critique to each branch (if active)
    easy_critique = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(easy_with_query)
    hard_critique = await synalinks.Generator(
        data_model=Critique,
        language_model=language_model,
        return_inputs=True,
    )(hard_with_query)

    # Merge branch outputs with |
    outputs = easy_critique | hard_critique

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="logical_and",
        description="Demonstrates safe concatenation with branches",
    )

    synalinks.utils.plot_program(program, to_folder="examples", show_module_names=True)

    print("Pattern: Branch -> Safe concat (&) -> Process -> Merge (|)")

    # =========================================================================
    # EXAMPLE 4: NOT Operator (~) - Inverting Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 4: NOT Operator (~)")
    print("=" * 60)
    print("Converts any data to None. Useful for conditional cancellation.\n")

    # Demonstrate NOT with actual data
    sample_data = Answer(answer="42")
    inverted = ~sample_data
    print(f"Original: {sample_data}")
    print(f"After ~: {inverted}")

    # =========================================================================
    # EXAMPLE 5: XOR Operator (^) - Exclusive Or
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 5: XOR Operator (^)")
    print("=" * 60)
    print("Returns data only if exactly ONE operand is non-None.\n")

    data_a = Answer(answer="Answer A")
    data_b = Answer(answer="Answer B")

    # XOR truth table demonstration
    print("XOR Truth Table:")
    print(f"  data ^ None = has data: {(data_a ^ None) is not None}")
    print(f"  None ^ data = has data: {(None ^ data_b) is not None}")
    print(f"  data ^ data = is None: {(data_a ^ data_b) is None}")


if __name__ == "__main__":
    asyncio.run(main())
