"""
# Observability

**Observability** is the ability to understand the internal state of a system
by examining its outputs. In LM applications, this means tracking every prompt,
response, token usage, and decision - enabling you to debug issues, optimize
performance, and monitor production systems.

## Why Observability Matters

LM applications are inherently non-deterministic and complex. Without
observability, you're flying blind:

```mermaid
graph LR
    subgraph Without Observability
        A[Input] --> B[Black Box]
        B --> C[Output]
        C --> D["Why did it fail?"]
    end
    subgraph With Observability
        E[Input] --> F[Traced Pipeline]
        F --> G[Output]
        H[Traces] --> I[Debug & Optimize]
    end
```

Observability enables:

1. **Debugging**: See exactly what prompts were sent and responses received
2. **Performance Monitoring**: Track latency, token usage, and costs
3. **Quality Assurance**: Identify and fix problematic outputs
4. **Optimization**: Find bottlenecks and improve efficiency

## Enabling Observability

Synalinks uses MLflow for tracing and metrics:

```python
import synalinks

synalinks.enable_observability(
    tracking_uri="http://localhost:5000",  # MLflow server
    experiment_name="my_experiment",        # Group related runs
)
```

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

Then open http://localhost:5000 in your browser.

## What Gets Traced

Every operation in your program is automatically traced:

```mermaid
graph TD
    A[Program Call] --> B[Trace]
    B --> C[Module: Input]
    B --> D[Module: Generator]
    D --> E[LLM Call]
    E --> F[Prompt]
    E --> G[Response]
    E --> H[Token Count]
    B --> I[Module: Branch]
    I --> J[Selected Path]
```

### Trace Contents

| Component | What's Captured |
|-----------|-----------------|
| Program | Input/output DataModels, execution time |
| Module | Each module's inputs, outputs, parameters |
| LLM Call | Full prompt, response, model name, tokens |
| Tool Call | Tool name, arguments, result |
| Training | Metrics, hyperparameters, artifacts |

## Logging Levels

Control log verbosity for debugging:

```python
import synalinks

# Detailed logging - every LLM call logged
synalinks.enable_logging(log_level="debug")

# Standard logging - key events only
synalinks.enable_logging(log_level="info")

# Quiet logging - warnings and errors only
synalinks.enable_logging(log_level="warning")
```

## Debugging Tools

### Program Summary

Inspect your program's structure:

```python
program.summary()
```

Output:
```
Program: qa_program
description: 'A `Functional` program is a `Program` defined as a directed graph
of modules.'
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Module (type)               ┃ Output Schema         ┃    Variable # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_module (InputModule)  │ Query                 │             0 │
├─────────────────────────────┼───────────────────────┼───────────────┤
│ generator (Generator)       │ Answer                │             2 │
└─────────────────────────────┴───────────────────────┴───────────────┘
 Total variables: 2
 Trainable variables: 2
```

### Program Visualization

Generate a visual graph of your program:

```python
synalinks.utils.plot_program(
    program,
    to_folder="output",
    show_module_names=True,
    show_trainable=True,
)
```

This creates a PNG file showing the computation graph.

### Trainable Variables

Inspect what can be optimized:

```python
for var in program.trainable_variables:
    print(f"Variable: {var.name}")
    print(f"  Value: {var.value}")
```

## MLflow Integration

### Viewing Traces

Navigate to the MLflow UI at http://localhost:5000:

1. Select your experiment from the sidebar
2. Click on a run to see its traces
3. Expand traces to see individual module calls
4. Click on LLM calls to see full prompts/responses

### Trace Structure

Each trace shows the hierarchical execution:

```
Program Call: qa_program
├── Module: Input
│   ├── Input: {"query": "What is Python?"}
│   └── Duration: 0.001s
├── Module: Generator
│   ├── LLM Call: openai/gpt-4.1-mini
│   │   ├── Prompt: [full text]
│   │   ├── Response: [full text]
│   │   ├── Input tokens: 150
│   │   └── Output tokens: 50
│   └── Duration: 1.2s
└── Output: {"answer": "Python is..."}
```

### Training Metrics

During training, MLflow captures:

- Per-epoch metrics (reward, accuracy)
- Validation metrics
- Hyperparameters (optimizer settings, epochs)
- Artifacts (saved program checkpoints)

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    \"\"\"User question.\"\"\"
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    \"\"\"Answer with reasoning.\"\"\"
    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")

async def main():
    load_dotenv()
    synalinks.clear_session()

    # Enable observability
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="observability_demo",
    )

    # Enable logging
    synalinks.enable_logging(log_level="info")

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Create program
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

    # Print summary
    program.summary()

    # Run program - traces are automatically captured
    result = await program(Query(query="What is Python?"))
    print(f"Answer: {result['answer']}")

    # Visualize program
    synalinks.utils.plot_program(
        program,
        show_module_names=True,
    )

    # Inspect trainable variables
    print("\\nTrainable variables:")
    for var in program.trainable_variables:
        print(f"  - {var.name}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Production Monitoring

For production deployments, set up persistent MLflow tracking:

```python
import synalinks

# Use a remote MLflow server
synalinks.enable_observability(
    tracking_uri="http://mlflow.your-domain.com:5000",
    experiment_name="production",
)
```

Monitor key metrics:

- **Latency**: Time per request
- **Token Usage**: Cost per request
- **Error Rate**: Failed requests
- **Quality Scores**: If using LMAsJudge or similar

## Key Takeaways

- **MLflow Integration**: Synalinks uses MLflow for comprehensive tracing
  and metrics. All traces are automatically captured.

- **enable_observability()**: Call this at startup with your MLflow server
  URI and experiment name.

- **Trace Hierarchy**: Traces show the full execution path from program
  to individual LLM calls.

- **Debugging Tools**: Use `program.summary()`, `plot_program()`, and
  `trainable_variables` for inspection.

- **Logging Levels**: Control verbosity with `enable_logging()` - use
  DEBUG for development, WARNING for production.

- **Production Ready**: Point to a remote MLflow server for production
  monitoring and alerting.

## API References

- [enable_observability](https://synalinks.github.io/synalinks/Synalinks%20API/Observability%20API/)
- [enable_logging](https://synalinks.github.io/synalinks/Synalinks%20API/Observability%20API/)
- [plot_program](https://synalinks.github.io/synalinks/Synalinks%20API/Utils%20API/)
- [Program.summary](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
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
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # -------------------------------------------------------------------------
    # Enable Observability
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
    # Enable Logging
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Enable Logging")
    print("=" * 60)

    synalinks.enable_logging(log_level="info")
    print("\nLogging enabled at INFO level")

    # -------------------------------------------------------------------------
    # Create and Run Program
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
    # Program Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Program Summary")
    print("=" * 60)

    program.summary()

    # -------------------------------------------------------------------------
    # Inspect Trainable Variables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Inspect Trainable Variables")
    print("=" * 60)

    print("\nTrainable variables:")
    for var in program.trainable_variables:
        print(f"  - {var.name}")

    # -------------------------------------------------------------------------
    # Visualize Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Visualize Program")
    print("=" * 60)

    synalinks.utils.plot_program(
        program,
        to_folder="guides",
        show_module_names=True,
        show_trainable=True,
        show_schemas=False,
    )

    print("\nProgram visualization saved to guides/traced_qa.png")

    # -------------------------------------------------------------------------
    # Multiple Traced Calls
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
    # Tracing Structure
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


if __name__ == "__main__":
    asyncio.run(main())
