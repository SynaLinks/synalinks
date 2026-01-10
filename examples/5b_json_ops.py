"""
# Lesson 5b: JSON Operations

In Lesson 5a, you learned about data model operators (`+`, `&`, `|`, `^`, `~`).
This lesson covers **JSON operations** - functions for transforming, filtering,
and reshaping data models.

## Categories of Operations

### 1. Masking Operations (Filtering Fields)

| Operation | Description | Example |
|-----------|-------------|---------|
| `in_mask` | Keep only specified fields | `ops.in_mask(data, mask=["answer"])` |
| `out_mask` | Remove specified fields | `ops.out_mask(data, mask=["thinking"])` |

### 2. Renaming Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `prefix` | Add prefix to field names | `ops.prefix(data, prefix="v1_")` |
| `suffix` | Add suffix to field names | `ops.suffix(data, suffix="_draft")` |

### 3. Aggregation Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `factorize` | Group similar fields into lists | `ops.factorize(combined)` |

### 4. Logical Operations (Function Form)

| Operation | Equivalent | Description |
|-----------|------------|-------------|
| `ops.concat` | `+` | Merge fields with custom naming |
| `ops.logical_and` | `&` | Safe merge |
| `ops.logical_or` | `|` | First non-None |
| `ops.logical_xor` | `^` | Exactly one non-None |

## Why Use These Operations?

1. **Data Preparation**: Transform data before passing to next module
2. **Field Selection**: Keep only relevant fields for downstream processing
3. **Conflict Resolution**: Rename fields to avoid collisions when merging
4. **Aggregation**: Combine multiple similar outputs into lists

## Running the Example

```bash
uv run python examples/5b_json_ops.py
```

## Program Visualizations

![in_mask_example](../assets/examples/in_mask_example.png)
![factorize_example](../assets/examples/factorize_example.png)
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


class AnswerWithThinking(synalinks.DataModel):
    """An answer with reasoning."""

    thinking: str = synalinks.Field(description="Your step by step thinking")
    answer: str = synalinks.Field(description="The correct answer")


class Answer(synalinks.DataModel):
    """A simple answer."""

    answer: str = synalinks.Field(description="The correct answer")


# =============================================================================
# STEP 2: Demonstrate JSON Operations
# =============================================================================


async def main():
    load_dotenv()

    # Enable observability for tracing (view traces at http://localhost:5000)
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="lesson_5b_json_ops",
    )

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # =========================================================================
    # EXAMPLE 1: In Mask - Keep Only Specific Fields
    # =========================================================================
    print("=" * 60)
    print("Example 1: In Mask - Keep Only Specific Fields")
    print("=" * 60)
    print("Filters a data model to keep only the specified fields.\n")

    inputs = synalinks.Input(data_model=Query)
    x = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    # Keep only the "answer" field, discard "thinking"
    outputs = await synalinks.ops.in_mask(x, mask=["answer"])

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="in_mask_example",
        description="Keep only the answer field",
    )

    synalinks.utils.plot_program(program, to_folder="examples")

    # Test it
    result = await program(Query(query="What is 2 + 2?"))
    print("Original fields: thinking, answer")
    print(f"After in_mask(['answer']): {list(result.keys())}")

    # =========================================================================
    # EXAMPLE 2: Out Mask - Remove Specific Fields
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Out Mask - Remove Specific Fields")
    print("=" * 60)
    print("Removes specified fields from a data model.\n")

    inputs = synalinks.Input(data_model=Query)
    x = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)
    # Remove the "thinking" field, keep everything else
    outputs = await synalinks.ops.out_mask(x, mask=["thinking"])

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="out_mask_example",
    )

    synalinks.utils.plot_program(program, to_folder="examples")

    result = await program(Query(query="What is 3 + 3?"))
    print("Original fields: thinking, answer")
    print(f"After out_mask(['thinking']): {list(result.keys())}")

    # =========================================================================
    # EXAMPLE 3: Prefix - Add Prefix to Field Names
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Prefix - Add Prefix to Field Names")
    print("=" * 60)
    print("Renames all fields by adding a prefix.\n")

    inputs = synalinks.Input(data_model=Query)
    x = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)
    outputs = await synalinks.ops.prefix(x, prefix="original_")

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="prefix_example",
    )

    synalinks.utils.plot_program(program, to_folder="examples")

    result = await program(Query(query="What is 4 + 4?"))
    print("Original fields: answer")
    print(f"After prefix('original_'): {list(result.keys())}")

    # =========================================================================
    # EXAMPLE 4: Suffix - Add Suffix to Field Names
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 4: Suffix - Add Suffix to Field Names")
    print("=" * 60)
    print("Renames all fields by adding a suffix.\n")

    inputs = synalinks.Input(data_model=Query)
    x = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)
    outputs = await synalinks.ops.suffix(x, suffix="_draft")

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="suffix_example",
    )

    synalinks.utils.plot_program(program, to_folder="examples")

    result = await program(Query(query="What is 5 + 5?"))
    print("Original fields: answer")
    print(f"After suffix('_draft'): {list(result.keys())}")

    # =========================================================================
    # EXAMPLE 5: Factorize - Group Similar Fields into Lists
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 5: Factorize - Group Similar Fields")
    print("=" * 60)
    print("Groups similar fields (answer, answer_1, answer_2) into a list.\n")

    inputs = synalinks.Input(data_model=Query)

    # Generate 3 different answers
    x1 = await synalinks.Generator(data_model=Answer, language_model=language_model)(
        inputs
    )
    x2 = await synalinks.Generator(data_model=Answer, language_model=language_model)(
        inputs
    )
    x3 = await synalinks.Generator(data_model=Answer, language_model=language_model)(
        inputs
    )

    # Concatenate: creates {answer, answer_1, answer_2}
    combined = x1 + x2 + x3

    # Factorize: groups into {answers: [...]}
    outputs = await synalinks.ops.factorize(combined)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="factorize_example",
    )

    synalinks.utils.plot_program(program, to_folder="examples")

    result = await program(Query(query="What is 6 + 6?"))
    print("After concat: answer, answer_1, answer_2")
    print(f"After factorize: {list(result.keys())}")
    if "answers" in result.keys():
        print(f"  answers has {len(result['answers'])} items")

    # =========================================================================
    # EXAMPLE 6: Concat Function (Named Operations)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 6: Concat Function with Custom Naming")
    print("=" * 60)
    print("Using function form allows custom names for operations.\n")

    inputs = synalinks.Input(data_model=Query)
    x1 = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)
    x2 = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)

    # Function form allows naming the operation
    outputs = await synalinks.ops.concat(
        x1,
        x2,
        name="combined_outputs",
        description="Merged simple and detailed answers",
    )

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="concat_function_example",
    )

    synalinks.utils.plot_program(program, to_folder="examples")

    print("Concat function allows custom naming for better traceability")


if __name__ == "__main__":
    asyncio.run(main())
