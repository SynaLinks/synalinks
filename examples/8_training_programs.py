"""
# Training Programs

Like in machine learning, a LM application needs to be trained. In that case, we
don't update the weights of the model, but optimize the prompts by automatically
picking the best examples or generate instructions in order to help the program to
perform better on your dataset.

In production settings, this means that you can use smaller and more cost-effective
models from your preferred provider while enhancing their accuracy with Synalinks.

## Training Flow

```mermaid
graph TB
    subgraph Data
        A[x_train, y_train]
        B[x_test, y_test]
    end
    subgraph Training
        C[program.compile] --> D[program.fit]
        D --> E{val_reward improved?}
        E -->|Yes| F[Save Checkpoint]
        E -->|No| D
    end
    subgraph Evaluation
        G[program.evaluate]
    end
    A --> D
    B --> G
    F --> G
```

## Loading a Dataset

Synalinks provides built-in datasets for training and evaluation:

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
```

## Training with `fit()`

Training a program is similar to Keras. Use the `fit()` method with your data:

```python
history = await program.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[program_checkpoint_callback],
)
```

## Saving and Loading Checkpoints

Use callbacks to save the best performing program during training:

```python
program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
    filepath="checkpoint.program.json",
    monitor="val_reward",
    mode="max",
    save_best_only=True,
)

# Load the best checkpoint after training
program.load("checkpoint.program.json")
```

### Key Takeaways

- **Dataset Loading**: Use built-in datasets or create your own for training.
- **Training Loop**: The `fit()` method handles the training process with
    configurable epochs, batch size, and validation split.
- **Checkpointing**: Save the best performing model during training using
    `ProgramCheckpoint` callback.
- **Evaluation**: Use `evaluate()` to measure performance before and after training.

## Program Visualization

![gsm8k_baseline](../assets/examples/gsm8k_baseline.png)

## API References

- [Program Training API](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/Program%20training%20API/)
- [Rewards](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/)
- [Optimizers](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/)
- [Callbacks](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/)
- [Built-in Datasets (GSM8K)](https://synalinks.github.io/synalinks/Synalinks%20API/Built-in%20Datasets/GSM8K/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks

NB_EPOCHS = 20
BATCH_SIZE = 32
NB_SAMPLES = None  # Set to a number to limit the dataset size
NB_RUNS = 3

FOLDER = "examples"

checkpoint_filepath = "checkpoint.program.json"


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="gsm8k_training",
    )

    # Initialize the language model
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    # ==========================================================================
    # Load the GSM8k dataset
    # ==========================================================================
    print("Loading GSM8k dataset...")
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

    if NB_SAMPLES:
        x_train = x_train[:NB_SAMPLES]
        y_train = y_train[:NB_SAMPLES]
        x_test = x_test[:NB_SAMPLES]
        y_test = y_test[:NB_SAMPLES]

    print("Done.")

    # ==========================================================================
    # Create the program
    # ==========================================================================
    print("Creating program...")
    inputs = synalinks.Input(
        data_model=synalinks.datasets.gsm8k.get_input_data_model(),
    )
    outputs = await synalinks.Generator(
        data_model=synalinks.datasets.gsm8k.get_output_data_model(),
        language_model=language_model,
    )(inputs)
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="gsm8k_baseline",
        description="The GSM8k baseline",
    )

    synalinks.utils.plot_program(
        program,
        to_folder=FOLDER,
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    # ==========================================================================
    # Compile the program
    # ==========================================================================
    print("Compiling...")
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    print("Done.")

    # ==========================================================================
    # Baseline evaluation (before training)
    # ==========================================================================
    print(f"Perform baseline evaluation samples with {NB_RUNS} runs...")
    baseline_metric_list = []
    for i in range(NB_RUNS):
        print(f"Run {i + 1}/{NB_RUNS}")
        metrics = await program.evaluate(
            x=x_test,
            y=y_test,
            batch_size=BATCH_SIZE,
        )
        baseline_metric_list.append(metrics)
    print("Done.")

    synalinks.utils.plot_metrics_with_mean_and_std(
        baseline_metric_list,
        to_folder=FOLDER,
        title="Evaluation without training",
    )

    # ==========================================================================
    # Training
    # ==========================================================================
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )

    program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
        filepath=os.path.join(FOLDER, checkpoint_filepath),
        monitor="val_reward",
        mode="max",
        save_best_only=True,
    )

    print(f"Start training for {NB_EPOCHS} epochs...")
    history = await program.fit(
        x=x_train,
        y=y_train,
        validation_split=0.2,
        epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[program_checkpoint_callback],
    )
    print("Done.")

    synalinks.utils.plot_history(
        history,
        to_folder=FOLDER,
        to_file="gsm8k_baseline_training_history.png",
    )

    # ==========================================================================
    # Final evaluation (after training)
    # ==========================================================================
    print("Load best performing checkpoint...")
    program.load(os.path.join(FOLDER, checkpoint_filepath))
    print("Done.")

    print("Perform final evaluation...")
    trained_metric_list = []
    for i in range(NB_RUNS):
        print(f"Run {i + 1}/{NB_RUNS}")
        metrics = await program.evaluate(
            x=x_test,
            y=y_test,
            batch_size=BATCH_SIZE,
        )
        trained_metric_list.append(metrics)
    print("Done.")

    metrics_comparison = {
        "without_training": baseline_metric_list,
        "with_training": trained_metric_list,
    }

    synalinks.utils.plot_metrics_comparison_with_mean_and_std(
        metrics_comparison,
        to_folder=FOLDER,
        to_file="gsm8k_evaluation_comparison.png",
        title="Comparison w/o training (GSM8K with EM reward)",
    )


if __name__ == "__main__":
    asyncio.run(main())
