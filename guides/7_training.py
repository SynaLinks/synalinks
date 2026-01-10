"""
# Training

**Training** in Synalinks is fundamentally different from traditional machine
learning. Instead of updating model weights through backpropagation, Synalinks
uses **in-context learning optimization** - improving your programs by
optimizing the prompts, instructions, and examples that guide the language
model.

## The Philosophy of In-Context Learning

Traditional ML updates weights; Synalinks updates context:

```mermaid
graph LR
    subgraph Traditional ML
        A[Data] --> B[Backprop]
        B --> C[Update Weights]
        C --> D[Better Model]
    end
    subgraph Synalinks Training
        E[Data] --> F[Evaluate]
        F --> G[Update Context]
        G --> H[Better Prompts/Examples]
    end
```

This approach has key advantages:

1. **No Gradient Computation**: Works with any LLM API
2. **Interpretable**: You can read and understand what was learned
3. **Modular**: Each module learns independently
4. **Fast**: No heavy computation - just prompt optimization

## What Gets Optimized

Each Generator module has two trainable variables:

```mermaid
graph TD
    A[Generator] --> B[instruction_variable]
    A --> C[examples_variable]
    B --> D["System prompt optimization"]
    C --> E["Few-shot example selection"]
```

- **instruction_variable**: The system prompt or instruction prefix
- **examples_variable**: Few-shot examples injected into the prompt

## The Training Loop

Training follows a familiar pattern:

```python
import synalinks

# 1. Create your program
program = synalinks.Program(inputs=inputs, outputs=outputs)

# 2. Compile with optimizer and reward
program.compile(
    optimizer=synalinks.optimizers.RandomFewShot(nb_max_examples=3),
    reward=synalinks.ExactMatch(key="answer"),
)

# 3. Train
history = await program.fit(
    x=training_data,
    epochs=5,
    validation_data=test_data,
    verbose=1,
)

# 4. Save the trained program
program.save("trained_program.json")
```

## Training Data Format

Training data consists of separate NumPy arrays for inputs (x) and expected
outputs (y):

```python
import numpy as np

x_train = np.array(
    [
        InputModel(field="value"),
        InputModel(field="value2"),
        # ... more examples
    ],
    dtype="object",
)
y_train = np.array(
    [
        OutputModel(result="expected"),
        OutputModel(result="expected2"),
        # ... more examples
    ],
    dtype="object",
)
```

Both arrays must contain DataModel instances matching your program's input and
output schemas.

## Optimizers

Optimizers determine how trainable variables are updated:

### RandomFewShot

Randomly samples `k` examples from training data to use as few-shot prompts:

```python
optimizer = synalinks.optimizers.RandomFewShot(nb_max_examples=3)
```

- Simple and effective
- Good baseline for most tasks
- Low computational overhead

### OMEGA (Optimizing Memory with Evolution and Gradient Alignment)

Advanced evolutionary optimizer that:

- Maintains a population of prompt variants
- Uses fitness-based selection
- Applies crossover and mutation
- Converges to high-performing prompts

```python
optimizer = synalinks.OMEGA(
    population_size=10,
    mutation_rate=0.1,
)
```

## Rewards

Rewards measure how well outputs match expected values:

### ExactMatch

Returns 1.0 if field values match exactly, 0.0 otherwise:

```python
reward = synalinks.ExactMatch(key="answer")
```

Best for:

- Classification tasks
- Factual QA with known answers
- Tasks where partial credit doesn't make sense

### CosineSimilarity

Uses embedding similarity between outputs and expected values:

```python
reward = synalinks.CosineSimilarity(
    embedding_model=embedding_model,
    key="answer",
)
```

Best for:

- Open-ended generation
- Semantic similarity matters
- Multiple valid phrasings

### LMAsJudge

Uses another LLM to evaluate output quality:

```python
reward = synalinks.LMAsJudge(
    language_model=judge_model,
    instructions="accuracy, helpfulness, clarity",
)
```

Best for:

- Complex evaluation criteria
- Subjective quality assessment
- When exact matching is too strict

## Metrics

Track performance during training:

```python
program.compile(
    optimizer=optimizer,
    reward=reward,
    metrics=[
        synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward"),
    ],
)
```

The training history contains all tracked metrics:

```python
history = await program.fit(x=data, epochs=5)

print(history.history.keys())
# ['mean_reward', 'val_mean_reward']
```

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

# =============================================================================
# Data Models
# =============================================================================

class MathProblem(synalinks.DataModel):
    \"\"\"A math problem.\"\"\"
    problem: str = synalinks.Field(description="The math problem to solve")

class MathAnswer(synalinks.DataModel):
    \"\"\"A math answer.\"\"\"
    thinking: str = synalinks.Field(description="Step by step calculation")
    answer: str = synalinks.Field(description="The numerical answer only")

# =============================================================================
# Main
# =============================================================================

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # Prepare Training Data
    # -------------------------------------------------------------------------
    train_data = [
        (MathProblem(problem="2 + 3"), MathAnswer(thinking="2 + 3 = 5", answer="5")),
        (MathProblem(problem="5 * 4"), MathAnswer(thinking="5 * 4 = 20", answer="20")),
        (MathProblem(problem="10 - 3"), MathAnswer(thinking="10 - 3 = 7", answer="7")),
        (MathProblem(problem="8 / 2"), MathAnswer(thinking="8 / 2 = 4", answer="4")),
        (MathProblem(problem="3 + 3 + 3"), MathAnswer(thinking="3 + 3 + 3 = 9", answer="9")),
        (MathProblem(problem="7 * 2"), MathAnswer(thinking="7 * 2 = 14", answer="14")),
    ]

    test_data = [
        (MathProblem(problem="4 + 5"), MathAnswer(thinking="4 + 5 = 9", answer="9")),
        (MathProblem(problem="6 * 3"), MathAnswer(thinking="6 * 3 = 18", answer="18")),
    ]

    # -------------------------------------------------------------------------
    # Create Program
    # -------------------------------------------------------------------------
    inputs = synalinks.Input(data_model=MathProblem)
    outputs = await synalinks.Generator(
        data_model=MathAnswer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_solver",
    )

    # -------------------------------------------------------------------------
    # Compile with Optimizer and Reward
    # -------------------------------------------------------------------------
    reward = synalinks.ExactMatch(key="answer")

    program.compile(
        optimizer=synalinks.optimizers.RandomFewShot(nb_max_examples=3),
        reward=reward,
        metrics=[
            synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward"),
        ],
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    history = await program.fit(
        x=train_data,
        epochs=2,
        validation_data=test_data,
        verbose=1,
    )

    print(f"Training history: {list(history.history.keys())}")

    # -------------------------------------------------------------------------
    # Test Trained Program
    # -------------------------------------------------------------------------
    result = await program(MathProblem(problem="9 + 1"))
    print(f"9 + 1 = {result['answer']}")

    # -------------------------------------------------------------------------
    # Save and Load
    # -------------------------------------------------------------------------
    program.save("trained_math.json")
    loaded = synalinks.Program.load("trained_math.json")

    result = await loaded(MathProblem(problem="100 / 10"))
    print(f"100 / 10 = {result['answer']}")

    import os
    if os.path.exists("trained_math.json"):
        os.remove("trained_math.json")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

### Start Simple

Begin with `RandomFewShot` and `ExactMatch`:

```python
program.compile(
    optimizer=synalinks.optimizers.RandomFewShot(nb_max_examples=3),
    reward=synalinks.ExactMatch(key="answer"),
)
```

Only move to more complex optimizers/rewards if needed.

### Use Quality Training Data

- Ensure examples are correct and representative
- Include edge cases and variations
- Balance difficulty levels

### Monitor Validation Metrics

Always use validation data to detect overfitting:

```python
history = await program.fit(
    x=train_data,
    validation_data=val_data,  # Always include this
    epochs=5,
)
```

### Save Checkpoints

Save your program after training to preserve learned state:

```python
program.save("checkpoint.json")
```

## Key Takeaways

- **In-Context Learning**: Synalinks optimizes prompts and examples, not
  model weights. This works with any LLM API.

- **Trainable Variables**: Each Generator has instruction and example
  variables that get optimized during training.

- **compile() + fit()**: Familiar Keras-like API for configuring and
  running the training loop.

- **Optimizers**: Start with `RandomFewShot`, move to `OMEGA` for more
  sophisticated optimization.

- **Rewards**: Choose based on your task - `ExactMatch` for exact answers,
  `CosineSimilarity` for semantic similarity, `LMAsJudge` for complex criteria.

- **Save Trained State**: Use `program.save()` to preserve learned prompts
  and examples for deployment.

## API References

- [Program.compile](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [Program.fit](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [RandomFewShot](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/RandomFewShot%20optimizer/)
- [ExactMatch](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards%20API/ExactMatch%20reward/)
- [Metrics](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics%20API/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Data Models
# =============================================================================


class MathProblem(synalinks.DataModel):
    """A math problem."""

    problem: str = synalinks.Field(description="The math problem to solve")


class MathAnswer(synalinks.DataModel):
    """A math answer."""

    thinking: str = synalinks.Field(description="Step by step calculation")
    answer: str = synalinks.Field(description="The numerical answer only")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="guide_7_training",
    )

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # -------------------------------------------------------------------------
    # Prepare Training Data (as NumPy arrays)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Prepare Training Data")
    print("=" * 60)

    import numpy as np

    # Training data: separate arrays for inputs (x) and expected outputs (y)
    x_train = np.array(
        [
            MathProblem(problem="2 + 3"),
            MathProblem(problem="5 * 4"),
            MathProblem(problem="10 - 3"),
            MathProblem(problem="8 / 2"),
            MathProblem(problem="3 + 3 + 3"),
            MathProblem(problem="7 * 2"),
            MathProblem(problem="15 - 5"),
            MathProblem(problem="12 / 3"),
        ],
        dtype="object",
    )
    y_train = np.array(
        [
            MathAnswer(thinking="2 + 3 = 5", answer="5"),
            MathAnswer(thinking="5 * 4 = 20", answer="20"),
            MathAnswer(thinking="10 - 3 = 7", answer="7"),
            MathAnswer(thinking="8 / 2 = 4", answer="4"),
            MathAnswer(thinking="3 + 3 + 3 = 9", answer="9"),
            MathAnswer(thinking="7 * 2 = 14", answer="14"),
            MathAnswer(thinking="15 - 5 = 10", answer="10"),
            MathAnswer(thinking="12 / 3 = 4", answer="4"),
        ],
        dtype="object",
    )

    # Test data
    x_test = np.array(
        [
            MathProblem(problem="4 + 5"),
            MathProblem(problem="6 * 3"),
        ],
        dtype="object",
    )
    y_test = np.array(
        [
            MathAnswer(thinking="4 + 5 = 9", answer="9"),
            MathAnswer(thinking="6 * 3 = 18", answer="18"),
        ],
        dtype="object",
    )

    print(f"\nTraining examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")

    # -------------------------------------------------------------------------
    # Create and Compile Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Create and Compile Program")
    print("=" * 60)

    inputs = synalinks.Input(data_model=MathProblem)
    outputs = await synalinks.Generator(
        data_model=MathAnswer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_solver",
    )

    reward = synalinks.ExactMatch(in_mask=["answer"])

    program.compile(
        optimizer=synalinks.optimizers.RandomFewShot(nb_max_examples=3),
        reward=reward,
        metrics=[
            synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward"),
        ],
    )

    print("\nProgram compiled with:")
    print("  - Optimizer: RandomFewShot(nb_max_examples=3)")
    print("  - Reward: ExactMatch(in_mask=['answer'])")

    # -------------------------------------------------------------------------
    # Train the Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Train the Program")
    print("=" * 60)

    history = await program.fit(
        x=x_train,
        y=y_train,
        epochs=2,
        validation_split=0.2,
        verbose=1,
        callbacks=[],
    )

    print("\nTraining complete!")
    print(f"History keys: {list(history.history.keys())}")

    # -------------------------------------------------------------------------
    # Test the Trained Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Test Trained Program")
    print("=" * 60)

    test_problems = [
        "9 + 1",
        "5 * 5",
        "20 - 8",
    ]

    for problem in test_problems:
        result = await program(MathProblem(problem=problem))
        print(f"\nProblem: {problem}")
        print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Save and Load Trained Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Save and Load")
    print("=" * 60)

    program.save("trained_math_solver.json")
    print("\nSaved trained program to trained_math_solver.json")

    loaded = synalinks.Program.load("trained_math_solver.json")
    print(f"Loaded program: {loaded.name}")

    result = await loaded(MathProblem(problem="100 / 10"))
    print(f"\nLoaded program test: 100 / 10 = {result['answer']}")

    if os.path.exists("trained_math_solver.json"):
        os.remove("trained_math_solver.json")


if __name__ == "__main__":
    asyncio.run(main())
