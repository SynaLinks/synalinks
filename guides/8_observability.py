"""
# Guide 8: Observability

Synalinks provides built-in observability features to monitor, debug, and
trace your LM applications.

## Enabling Observability

```python
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",
    experiment_name="my_experiment",
)
```

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

## What Gets Traced

- **Program calls** - Input/output data, execution time
- **Module calls** - Each module's inputs and outputs
- **LLM calls** - Prompts, responses, token usage
- **Tool calls** - Agent tool invocations
- **Training runs** - Metrics, parameters, artifacts

## Logging Levels

```python
synalinks.enable_logging(level="DEBUG")  # Detailed
synalinks.enable_logging(level="INFO")   # Standard
synalinks.enable_logging(level="WARNING") # Quiet
```

## Debugging Tools

- `program.summary()` - Print program structure
- `synalinks.plot_program()` - Visualize program graph
- `program.trainable_variables` - Inspect trainable state

## Running the Example

```bash
uv run python guides/8_observability.py
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
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# STEP 2: Demonstrate Observability Features
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # -------------------------------------------------------------------------
    # 2.1: Enable Observability
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Enable Observability")
    print("=" * 60)

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_8_observability",
    )

    print("\nObservability enabled!")
    print("View traces at: http://localhost:5000")
    print("\nTo start MLflow UI, run:")
    print("  mlflow ui --port 5000")

    # -------------------------------------------------------------------------
    # 2.2: Enable Logging
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Enable Logging")
    print("=" * 60)

    synalinks.enable_logging(level="INFO")
    print("\nLogging enabled at INFO level")

    # -------------------------------------------------------------------------
    # 2.3: Create and Run Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Create and Run Program (traces will be recorded)")
    print("=" * 60)

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="traced_qa",
    )

    result = await program(Query(query="What is Python used for?"))

    print(f"\nAnswer: {result['answer'][:100]}...")

    # -------------------------------------------------------------------------
    # 2.4: Program Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Program Summary")
    print("=" * 60)

    program.summary()

    # -------------------------------------------------------------------------
    # 2.5: Inspect Trainable Variables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Inspect Trainable Variables")
    print("=" * 60)

    print("\nTrainable variables:")
    for var in program.trainable_variables:
        print(f"  - {var.name}")

    # -------------------------------------------------------------------------
    # 2.6: Visualize Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Visualize Program")
    print("=" * 60)

    synalinks.utils.plot_program(
        program,
        to_folder="guides",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    print("\nProgram visualization saved to guides/traced_qa.png")

    # -------------------------------------------------------------------------
    # 2.7: Multiple Traced Calls
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Multiple Traced Calls")
    print("=" * 60)

    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is deep learning?",
    ]

    print("\nRunning multiple queries (all will be traced):")
    for q in queries:
        result = await program(Query(query=q))
        print(f"  Q: {q[:30]}... -> A: {result['answer'][:40]}...")

    # -------------------------------------------------------------------------
    # 2.8: Tracing Structure
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Tracing Structure")
    print("=" * 60)

    print(
        """
Each trace in MLflow contains:

Program Call
├── Module: Input
├── Module: Generator
│   └── LLM Call: gpt-4.1-mini
│       ├── Prompt (full text)
│       ├── Response (full text)
│       └── Tokens: input/output count
└── Output Data
"""
    )

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. ENABLE_OBSERVABILITY: Start tracing to MLflow
2. MLFLOW UI: View traces at http://localhost:5000
3. ENABLE_LOGGING: Control log verbosity
4. SUMMARY: Print program structure
5. PLOT_PROGRAM: Visualize computation graph
6. TRAINABLE_VARIABLES: Inspect learned state
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
