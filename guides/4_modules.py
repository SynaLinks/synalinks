"""
# Modules

**Modules** are the fundamental building blocks of Synalinks programs. Just as
neurons are the basic units of computation in a neural network, modules are
the basic units of computation in a Synalinks program. Each module performs
a specific transformation on data, and modules can be composed together to
build complex LM applications.

## Core Modules

### Input: The Entry Point

Every program starts with an `Input` module. It defines the schema of data
entering your program:

```python
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

# Define the entry point - no computation happens here
inputs = synalinks.Input(data_model=Query)
```

The `Input` module doesn't transform data - it simply marks where data enters
the computation graph. Think of it as the "x" in f(x).

### Generator: The Core LLM Module

The `Generator` is the heart of Synalinks. It takes input data and uses a
language model to produce structured output:

```mermaid
graph LR
    A[Input DataModel] --> B[Generator]
    B --> C[LLM Call]
    C --> D[Constrained Decoding]
    D --> E[Output DataModel]
```

```python
import synalinks

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

outputs = await synalinks.Generator(
    data_model=Answer,
    language_model=language_model,
    instructions="Be concise and accurate.",  # Extra guidance
)(inputs)
```

Key Generator features:

- **Constrained Output**: Output always matches your DataModel schema
- **Automatic Prompting**: Synalinks constructs the prompt from your schema
- **Instructions Parameter**: Add extra guidance without modifying the schema
- **Trainable**: Instructions and examples can be optimized

### Identity: Pass-Through

The `Identity` module passes data unchanged. Useful for creating parallel
paths or as a placeholder:

```python
# Just pass the data through
unchanged = await synalinks.Identity()(inputs)
```

### Tool: Wrap Any Function

The `Tool` module wraps async Python functions for use in programs:

```python
import synalinks

@synalinks.saving.register_synalinks_serializable()
async def search_web(query: str):
    \"\"\"Search the web for information.

    Args:
        query (str): The search query.
    \"\"\"
    # Your search implementation
    return {"results": [...]}

tool = synalinks.Tool(search_web)
```

**Important Tool Constraints:**

- **No Optional Parameters**: All parameters must be required. OpenAI and
  other LLM providers require all tool parameters to be required in their
  JSON schemas. Do not use default values.

- **Complete Docstring Required**: Every parameter must be documented in
  the `Args:` section. The Tool extracts descriptions from the docstring
  to build the JSON schema. Missing descriptions raise a ValueError.

## Control Flow Modules

### Decision: Single-Label Classification

The `Decision` module classifies input into one of several categories,
enabling intelligent routing:

```mermaid
graph LR
    A[Input] --> B[Decision]
    B --> C{Which Label?}
    C -->|math| D[Math Handler]
    C -->|general| E[General Handler]
    C -->|code| F[Code Handler]
```

```python
decision = await synalinks.Decision(
    question="What type of question is this?",
    labels=["math", "general", "code"],
    language_model=language_model,
)(inputs)

# Result: {"choice": "math"} or {"choice": "general"} etc.
```

### Branch: Conditional Execution

The `Branch` module combines decision-making with routing. It takes a question,
labels, and corresponding branch modules:

```python
(math_output, general_output) = await synalinks.Branch(
    question="Is this a math or general question?",
    labels=["math", "general"],
    branches=[
        synalinks.Generator(
            data_model=Answer,
            language_model=lm,
            instructions="You are a math expert.",
        ),
        synalinks.Generator(
            data_model=Answer,
            language_model=lm,
            instructions="You are a general knowledge expert.",
        ),
    ],
    language_model=lm,
)(inputs)

# Combine with OR - only the selected branch produces output
outputs = math_output | general_output
```

The Branch module:

1. Asks the question to classify the input
2. Routes to the corresponding branch (others return None)
3. Use OR to combine the outputs

### Action: Context Injection

The `Action` module executes with injected context from previous steps:

```python
outputs = await synalinks.Action(
    language_model=language_model,
    data_model=Answer,
    context_key="search_results",  # Inject under this key
)(inputs, search_results)
```

## Merging Modules

When you have multiple data streams, merging modules combine them:

```mermaid
graph LR
    A[DataModel A] --> C[Merge Module]
    B[DataModel B] --> C
    C --> D[Combined DataModel]
```

### Concat (+): Merge All Fields

Combines all fields from multiple DataModels:

```python
# Merge two outputs
merged = await synalinks.Concat()([output_a, output_b])
# Result has all fields from both A and B
```

### Logical And (&): Merge with None Check

Like Concat, but returns None if any input is None:

```python
# Only merge if both exist
merged = await synalinks.And()([output_a, output_b])
# Result: merged DataModel or None
```

### Logical Or (|): ignore None

Returns the first non-None value:

```python
# Fallback pattern
result = await synalinks.Or()([primary, fallback])
# Returns primary if not None, else fallback
```

### Xor (^): Exclusive Choice

Returns None if both inputs are provided:

```python
# For guard patterns
result = await synalinks.Xor()([warning, data])
# If warning exists, data becomes None
```

## Masking Modules

Masking modules filter fields from DataModels:

### InMask: Keep Specified Fields

```python
# Keep only "answer" field, drop everything else
filtered = await synalinks.InMask(mask=["answer"])(full_output)
```

### OutMask: Remove Specified Fields

```python
# Remove "thinking" field, keep everything else
filtered = await synalinks.OutMask(mask=["thinking"])(full_output)
```

Masking is useful for:

- Hiding intermediate reasoning from final output
- Reducing token usage in downstream modules
- Focusing training on specific fields

## Test-Time Compute Modules

These modules use extra computation at inference time to improve quality:

### ChainOfThought: Automatic Reasoning

Adds a "thinking" field to encourage step-by-step reasoning:

```python
outputs = await synalinks.ChainOfThought(
    data_model=Answer,
    language_model=language_model,
)(inputs)

# Result includes both "thinking" and "answer" fields
print(result['thinking'])  # Step-by-step reasoning
print(result['answer'])    # Final answer
```

The thinking field is automatically added to your schema - you don't need
to include it in your DataModel.

### SelfCritique: Self-Evaluation

Generates output, then evaluates it with a reward function:

```python
outputs = await synalinks.SelfCritique(
    data_model=Answer,
    language_model=language_model,
    reward=synalinks.LMAsJudge(language_model=language_model),
)(inputs)
```

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

# =============================================================================
# Data Models
# =============================================================================

class Query(synalinks.DataModel):
    \"\"\"User question.\"\"\"
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    \"\"\"Simple answer.\"\"\"
    answer: str = synalinks.Field(description="The answer")

class AnswerWithThinking(synalinks.DataModel):
    \"\"\"Answer with reasoning.\"\"\"
    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")

# =============================================================================
# Main
# =============================================================================

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # Generator Example
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Module 1: Generator")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(inputs=inputs, outputs=outputs)
    result = await program(Query(query="What is Python?"))
    print(f"Generator output: {result['answer'][:100]}...")

    # -------------------------------------------------------------------------
    # Branch Example (in docstring)
    # -------------------------------------------------------------------------
    print("\\n" + "=" * 60)
    print("Module 2: Branch")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    (math_out, general_out) = await synalinks.Branch(
        question="Is this a math or general question?",
        labels=["math", "general"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                instructions="Show your calculations.",
            ),
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
            ),
        ],
        language_model=lm,
    )(inputs)

    outputs = math_out | general_out

    program = synalinks.Program(inputs=inputs, outputs=outputs)
    result = await program(Query(query="What is 15 * 23?"))
    print(f"Math result: {result['answer']}")

    # -------------------------------------------------------------------------
    # ChainOfThought Example
    # -------------------------------------------------------------------------
    print("\\n" + "=" * 60)
    print("Module 3: ChainOfThought")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(inputs=inputs, outputs=outputs)
    result = await program(Query(query="If I have 3 apples and give 1 away?"))
    print(f"Thinking: {result['thinking'][:100]}...")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Masking Example
    # -------------------------------------------------------------------------
    print("\\n" + "=" * 60)
    print("Module 4: Masking")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    full_output = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
    )(inputs)

    # Keep only the answer field
    masked = await synalinks.InMask(mask=["answer"])(full_output)

    program = synalinks.Program(inputs=inputs, outputs=masked)
    result = await program(Query(query="What is 1+1?"))
    print(f"Masked fields: {list(result.get_json().keys())}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Takeaways

- **Input**: Defines the entry point of your program. Every program needs at least one.

- **Generator**: The core LLM module. Takes input, produces structured output
  matching your DataModel schema. Supports instructions and is trainable.

- **Decision + Branch**: Enable intelligent routing. Decision classifies,
  Branch routes to the appropriate handler.

- **ChainOfThought**: Automatically adds step-by-step reasoning to improve
  accuracy on complex tasks.

- **Merging Operators**: `+` (Concat), `&` (And), `|` (Or), `^` (Xor) combine
  DataModels in different ways for different use cases.

- **InMask/OutMask**: Filter fields to hide intermediate work or focus on
  specific outputs.

## API References

- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Decision](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Decision%20module/)
- [Branch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Branch%20module/)
- [ChainOfThought](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/ChainOfThought%20module/)
- [InMask/OutMask](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Masking%20modules/)
- [Tool](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Tool%20module/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User question."""

    query: str = synalinks.Field(description="User question")


class Answer(synalinks.DataModel):
    """Simple answer."""

    answer: str = synalinks.Field(description="The answer")


class AnswerWithThinking(synalinks.DataModel):
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_4_modules",
    )

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # Generator Module
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Module 1: Generator")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="generator_demo",
    )

    result = await program(Query(query="What is Python?"))
    print(f"\nGenerator output: {result['answer'][:100]}...")

    # -------------------------------------------------------------------------
    # Decision Module
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 2: Decision")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    decision = await synalinks.Decision(
        question="What type of query is this?",
        labels=["factual", "opinion", "calculation"],
        language_model=lm,
    )(inputs)

    decision_program = synalinks.Program(
        inputs=inputs,
        outputs=decision,
        name="decision_demo",
    )

    result = await decision_program(Query(query="What is 2+2?"))
    print(f"\nDecision output: {result['choice']}")

    result = await decision_program(Query(query="Is Python a good language?"))
    print(f"Decision output: {result['choice']}")

    # -------------------------------------------------------------------------
    # Branch Module
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 3: Branch (includes decision-making)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Branch combines decision-making with routing
    (math_output, general_output) = await synalinks.Branch(
        question="Is this a math or general question?",
        labels=["math", "general"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                instructions="You are a math expert. Show your calculations.",
            ),
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                instructions="You are a general knowledge expert.",
            ),
        ],
        language_model=lm,
    )(inputs)

    # Use OR to combine - only selected branch produces output
    outputs = math_output | general_output

    branch_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="branch_demo",
    )

    result = await branch_program(Query(query="What is 15 * 23?"))
    print(f"\nMath branch result: {result['answer']}")

    result = await branch_program(Query(query="Who wrote Hamlet?"))
    print(f"General branch result: {result['answer']}")

    # -------------------------------------------------------------------------
    # ChainOfThought Module
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 4: ChainOfThought")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    cot_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="cot_demo",
    )

    result = await cot_program(Query(query="If I have 3 apples and give 1 away?"))
    print(f"\nThinking: {result['thinking'][:100]}...")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Merging Modules
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 5: Concat (Merging)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    branch_a = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        name="expert_a",
        instructions="You are expert A, brief answers.",
    )(inputs)

    branch_b = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
        name="expert_b",
        instructions="You are expert B, detailed answers.",
    )(inputs)

    merged = await synalinks.Concat()([branch_a, branch_b])

    merge_program = synalinks.Program(
        inputs=inputs,
        outputs=merged,
        name="merge_demo",
    )

    result = await merge_program(Query(query="What is AI?"))
    print(f"\nMerged fields: {list(result.get_json().keys())}")

    # -------------------------------------------------------------------------
    # Masking Modules
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 6: InMask and OutMask")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    full_output = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
    )(inputs)

    masked = await synalinks.InMask(mask=["answer"])(full_output)

    mask_program = synalinks.Program(
        inputs=inputs,
        outputs=masked,
        name="mask_demo",
    )

    result = await mask_program(Query(query="What is 1+1?"))
    print(f"\nMasked output fields: {list(result.get_json().keys())}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
