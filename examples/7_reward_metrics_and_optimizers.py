"""
# Rewards, Metrics & Optimizers

## Understanding Rewards

`Reward`s are an essential part of reinforcement learning frameworks.
They are scalar values (between 0.0 and 1.0 for synalinks) 
that guide the process into making more efficient decisions or
predictions. During training, the goal is to maximize the reward function.
The reward gives the system an indication of how well it performed for that task.

All rewards consist of a function or program that takes two inputs:

- `y_pred`: The prediction of the program.
- `y_true`: The ground truth/target value provided by the training data.

## Understanding Metrics

`Metric`s are scalar values that are monitored during training and evaluation.
These values are used to know which program is best, in order to save it. Or to
provide additional information to compare different architectures with each others.
Unlike `Reward`s, a `Metric` is not used during training, meaning the metric value
is not backpropagated. Additionaly every reward function can be used as metric.

## Predictions Filtering

Sometimes, your program have to output a complex JSON but you want to evaluate
just part of it. This could be because your training data only include a subset
of the JSON, or because the additonal fields were added only to help the LMs.
In that case, you have to filter out or filter in your predictions and ground
truth using `out_mask` or `in_mask` list parameter.

## Understanding Optimizers

```mermaid
graph LR
    subgraph Training Loop
        A[Input] --> B[Program]
        B --> C[y_pred]
        D[y_true] --> E[Reward]
        C --> E
        E --> F[Optimizer]
        F --> |update| B
    end
    E --> G[Metrics]
```

Optimizers are systems that handle the update of the module's state in order to
make them more performant. They are in charge of backpropagating the rewards
from the training process and select or generate examples and instructions for
the LMs.

```python
program.compile(
    reward=synalinks.rewards.CosineSimilarity(
        embedding_model=embedding_model,
        in_mask=["answer"],  # Only evaluate the "answer" field
    ),
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=[
        synalinks.metrics.F1Score(in_mask=["answer"]),
    ],
)
```

### Key Takeaways

- **Rewards**: Guide the reinforcement learning process by providing feedback
    on the system's performance.
- **Metrics**: Scalar values monitored during training and evaluation to
    determine the best-performing program.
- **Optimizers**: Update the module's state to improve performance.
- **Filtering Outputs**: Use `out_mask` or `in_mask` to evaluate only relevant
    fields of complex JSON outputs.

## API References

- [Rewards](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/)
- [Metrics](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/)
- [Optimizers](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/)
- [Program Training API](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/Program%20training%20API/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


# Define the data models
class Query(synalinks.DataModel):
    query: str = synalinks.Field(
        description="The user query",
    )


class AnswerWithThinking(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step by step thinking",
    )
    answer: str = synalinks.Field(
        description="The correct answer",
    )


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="reward_metrics_optimizers",
    )

    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # Initialize the embedding model for cosine similarity reward
    embedding_model = synalinks.EmbeddingModel(
        model="openai/text-embedding-3-small",
    )

    # ==========================================================================
    # Example: Program Compilation with Reward, Metrics, and Optimizer
    # ==========================================================================
    print("Example: Program Compilation")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="chain_of_thought",
        description="Useful to answer in a step by step manner.",
    )

    # Compile the program with reward, optimizer, and metrics
    # Note: in_mask=["answer"] filters to only evaluate the "answer" field
    program.compile(
        reward=synalinks.rewards.CosineSimilarity(
            embedding_model=embedding_model,
            in_mask=["answer"],
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[
            synalinks.metrics.F1Score(in_mask=["answer"]),
        ],
    )

    print("Program compiled successfully!")
    print(f"Reward: {program.reward}")
    print(f"Optimizer: {program.optimizer}")
    print(f"Metrics: {program.metrics}")


if __name__ == "__main__":
    asyncio.run(main())
