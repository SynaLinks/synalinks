"""
# Guide 4: Modules

Modules are the building blocks of Synalinks programs. This guide covers
all the core modules available.

## Core Modules

| Module | Description |
|--------|-------------|
| `Input` | Defines program entry point |
| `Generator` | LLM inference with structured outputs |
| `Identity` | Pass-through (no-op) |
| `Not` | Logical negation |

## Control Flow Modules

| Module | Description |
|--------|-------------|
| `Decision` | Single-label classification for routing |
| `Branch` | Conditional execution based on decisions |
| `Action` | Execute actions with context injection |

## Merging Modules

| Module | Operator | Description |
|--------|----------|-------------|
| `Concat` | `+` | Merge fields |
| `And` | `&` | Merge with None if either None |
| `Or` | `|` | Merge with None fallback |
| `Xor` | `^` | None if both present |

## Masking Modules

| Module | Description |
|--------|-------------|
| `InMask` | Keep only specified fields |
| `OutMask` | Remove specified fields |

## Test Time Compute Modules

| Module | Description |
|--------|-------------|
| `ChainOfThought` | Generator with automatic thinking field |
| `SelfCritique` | Self-evaluation with reward scoring |

## Running the Example

```bash
uv run python guides/4_modules.py
```
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
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
# STEP 2: Demonstrate Core Modules
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
    # 2.1: Generator Module
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
    # 2.2: Decision Module
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
    # 2.3: Branch Module
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 3: Decision + Branch")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    decision = await synalinks.Decision(
        question="Is this a math or general question?",
        labels=["math", "general"],
        language_model=lm,
    )(inputs)

    outputs = await synalinks.Branch(
        branches={
            "math": synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                hint="You are a math expert. Show your calculations.",
            ),
            "general": synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                hint="You are a general knowledge expert.",
            ),
        },
    )(inputs, decision)

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
    # 2.4: ChainOfThought Module
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
    # 2.5: Merging Modules
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 5: Concat (Merging)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Two parallel branches
    branch_a = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        name="expert_a",
        hint="You are expert A, brief answers.",
    )(inputs)

    branch_b = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
        name="expert_b",
        hint="You are expert B, detailed answers.",
    )(inputs)

    # Merge outputs using Concat
    merged = await synalinks.Concat()([branch_a, branch_b])

    merge_program = synalinks.Program(
        inputs=inputs,
        outputs=merged,
        name="merge_demo",
    )

    result = await merge_program(Query(query="What is AI?"))
    print(f"\nMerged fields: {list(result.get_json().keys())}")

    # -------------------------------------------------------------------------
    # 2.6: Masking Modules
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 6: InMask and OutMask")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    full_output = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
    )(inputs)

    # Keep only answer field
    masked = await synalinks.InMask(keys=["answer"])(full_output)

    mask_program = synalinks.Program(
        inputs=inputs,
        outputs=masked,
        name="mask_demo",
    )

    result = await mask_program(Query(query="What is 1+1?"))
    print(f"\nMasked output fields: {list(result.get_json().keys())}")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. GENERATOR: Core module for LLM inference
2. DECISION: Route to different branches based on classification
3. BRANCH: Execute different modules based on decision
4. CHAINOFTHOUGHT: Automatic reasoning with thinking field
5. CONCAT: Merge multiple data models
6. INMASK/OUTMASK: Filter fields in/out
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
