import asyncio
import os

import synalinks

NB_EPOCHS = 20
BATCH_SIZE = 32
STEPS_PER_EPOCHS = 3
NB_SAMPLES = None
NB_RUNS = 3

FOLDER = "examples/training_programs"

checkpoint_filepath = "checkpoint.program.json"

synalinks.clear_session()


async def main():
    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )
    print("Loading GSM8k dataset...")
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

    if NB_SAMPLES:
        x_train = x_train[:NB_SAMPLES]
        y_train = y_train[:NB_SAMPLES]
        x_test = x_test[:NB_SAMPLES]
        y_test = y_test[:NB_SAMPLES]

    print("Done.")

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

    print("Compiling...")
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    print("Done.")

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
        steps_per_epoch=STEPS_PER_EPOCHS,
        callbacks=[program_checkpoint_callback],
    )
    print("Done.")

    synalinks.utils.plot_history(
        history,
        to_folder=FOLDER,
        to_file="gsm8k_baseline_training_history.png",
    )

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
        show_values=True,
        to_file="gsm8k_evaluation_comparison.png",
        title="Comparison w/o training (GSM8K with EM reward)",
    )


if __name__ == "__main__":
    asyncio.run(main())
