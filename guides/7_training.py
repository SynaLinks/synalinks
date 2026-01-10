"""
# Guide 7: Training

Synalinks supports in-context learning optimization to improve your programs.
This guide covers training, optimizers, and rewards.

## Training Overview

Training in Synalinks optimizes:
- **Instructions**: The prompts given to generators
- **Examples**: Few-shot examples for better outputs

Unlike traditional ML, this is in-context learning - the LLM weights don't change.

## Basic Training Setup

```python
program.compile(
    optimizer=synalinks.RandomFewShot(k=3),
    reward=synalinks.ExactMatch(key="answer"),
)

history = await program.fit(
    x=train_data,
    epochs=5,
    verbose=1,
)
```

## Optimizers

| Optimizer | Description |
|-----------|-------------|
| `RandomFewShot` | Random sampling of examples |
| `OMEGA` | Advanced evolutionary optimizer |
| `GreedyOptimizer` | Greedy selection of best examples |
| `EvolutionaryOptimizer` | Population-based optimization |

## Rewards

| Reward | Description |
|--------|-------------|
| `ExactMatch` | Exact string match |
| `CosineSimilarity` | Semantic similarity using embeddings |
| `LMAsJudge` | LLM-based evaluation |

## Running the Example

```bash
uv run python guides/7_training.py
```
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# STEP 1: Define Data Models
# =============================================================================


class MathProblem(synalinks.DataModel):
    """A math problem."""

    problem: str = synalinks.Field(description="The math problem to solve")


class MathAnswer(synalinks.DataModel):
    """A math answer."""

    thinking: str = synalinks.Field(description="Step by step calculation")
    answer: str = synalinks.Field(description="The numerical answer only")


# =============================================================================
# STEP 2: Demonstrate Training
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
    # 2.1: Create Training Data
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Prepare Training Data")
    print("=" * 60)

    train_data = [
        (MathProblem(problem="2 + 3"), MathAnswer(thinking="2 + 3 = 5", answer="5")),
        (MathProblem(problem="5 * 4"), MathAnswer(thinking="5 * 4 = 20", answer="20")),
        (MathProblem(problem="10 - 3"), MathAnswer(thinking="10 - 3 = 7", answer="7")),
        (MathProblem(problem="8 / 2"), MathAnswer(thinking="8 / 2 = 4", answer="4")),
        (
            MathProblem(problem="3 + 3 + 3"),
            MathAnswer(thinking="3 + 3 + 3 = 9", answer="9"),
        ),
        (MathProblem(problem="7 * 2"), MathAnswer(thinking="7 * 2 = 14", answer="14")),
        (MathProblem(problem="15 - 5"), MathAnswer(thinking="15 - 5 = 10", answer="10")),
        (MathProblem(problem="12 / 3"), MathAnswer(thinking="12 / 3 = 4", answer="4")),
    ]

    test_data = [
        (MathProblem(problem="4 + 5"), MathAnswer(thinking="4 + 5 = 9", answer="9")),
        (MathProblem(problem="6 * 3"), MathAnswer(thinking="6 * 3 = 18", answer="18")),
    ]

    print(f"\nTraining examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # -------------------------------------------------------------------------
    # 2.2: Create and Compile Program
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

    # Compile with optimizer and reward
    reward = synalinks.ExactMatch(key="answer")

    program.compile(
        optimizer=synalinks.RandomFewShot(k=3),
        reward=reward,
        metrics=[
            synalinks.MeanMetricWrapper(metric=reward, name="mean_reward"),
        ],
    )

    print("\nProgram compiled with:")
    print("  - Optimizer: RandomFewShot(k=3)")
    print("  - Reward: ExactMatch(key='answer')")

    # -------------------------------------------------------------------------
    # 2.3: Train the Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Train the Program")
    print("=" * 60)

    history = await program.fit(
        x=train_data,
        epochs=2,
        validation_data=test_data,
        verbose=1,
        callbacks=[],
    )

    print("\nTraining complete!")
    print(f"History keys: {list(history.history.keys())}")

    # -------------------------------------------------------------------------
    # 2.4: Test the Trained Program
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
    # 2.5: Save and Load Trained Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Save and Load")
    print("=" * 60)

    program.save("trained_math_solver.json")
    print("\nSaved trained program to trained_math_solver.json")

    loaded = synalinks.Program.load("trained_math_solver.json")
    print(f"Loaded program: {loaded.name}")

    # Test loaded program
    result = await loaded(MathProblem(problem="100 / 10"))
    print(f"\nLoaded program test: 100 / 10 = {result['answer']}")

    # Cleanup
    import os

    if os.path.exists("trained_math_solver.json"):
        os.remove("trained_math_solver.json")

    # -------------------------------------------------------------------------
    # Key Takeaways
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Key Takeaways")
    print("=" * 60)
    print(
        """
1. COMPILE: Set optimizer, reward, and metrics
2. FIT: Train with (input, expected_output) pairs
3. RANDOMFEWSHOT: Simple and effective optimizer
4. EXACTMATCH: Reward for exact string matching
5. METRICS: Track performance during training
6. SAVE/LOAD: Preserve trained state
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
