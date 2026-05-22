"""
# Program Synthesis with PythonSynthesis + OMEGA

This example evolves a **Python program** to solve an ARC-AGI task.

Unlike a `Generator`, where the trainable state is a prompt and the LM
runs on every prediction, a `PythonSynthesis` module's trainable state
is a **Python script**. At inference time there is no LM at all — the
script runs in the Monty sandbox and transforms the input JSON into the
output JSON. The language model only appears during `fit`, where the
`OMEGA` optimizer uses it to **author and refine the script** across
generations (mutation + crossover), keeping a diverse population alive
with Dominated Novelty Search.

Pipeline:

1. Load an ARC-AGI task (`input_grid` -> `output_grid`).
2. Build a `PythonSynthesis` program seeded with an identity script.
3. Run the seed offline to see the starting point (it just echoes the
   input, so it scores 0 unless the rule is the identity).
4. (Optional) Evolve the script with `OMEGA`, then read the learned
   algorithm straight off the trainable variable.

`OMEGA` needs both a `language_model` (writes the code) and an
`embedding_model` (measures candidate novelty for DNS). ARC-AGI is hard:
the goal here is to show the wiring, not to guarantee a solve. Use a
strong code model for serious attempts.

## API References

- [PythonSynthesis](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Synthesis%20Modules/PythonSynthesis%20module/)
- [OMEGA](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/OMEGA/)
- [ExactMatch](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/ExactMatch%20reward/)
"""

# --8<-- [start:source]
import asyncio

from dotenv import load_dotenv

import synalinks

# Flip to True to run the (slow, LM-driven) OMEGA training loop. Left off
# by default so the example's forward pass runs offline in seconds.
RUN_TRAINING = False

# "007bbfb7" is the classic "fractal self-tiling" ARC-AGI task: the 3x3
# input is stamped into a 3x3 layout of itself, placing a copy wherever
# the input cell is non-zero (a 9x9 output).
TASK_NAME = "007bbfb7"

# The seed: an identity transform. OMEGA rewrites the body of `transform`.
SEED_SCRIPT = """
def transform(inputs):
    # Starting point: copy the input grid unchanged. OMEGA will rewrite
    # this body to discover the rule mapping input_grid -> output_grid.
    return {"output_grid": inputs.get("input_grid")}

result = transform(inputs)
"""


def grid_str(grid):
    """Render a small integer grid as text for printing.

    Cells may be plain ints (from the sandbox, via JSON) or `Color` enum
    members (from the typed ground-truth DataModel); coerce to int so
    both sides print identically.
    """
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


async def main():
    load_dotenv()
    synalinks.clear_session()

    # 1. Load the ARC-AGI task.
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
        task_name=TASK_NAME,
        arc_version=1,
    )
    print(f"Training examples: {len(x_train)} | Test examples: {len(x_test)}")

    # 2. Build the PythonSynthesis program (no LM involved).
    inputs = synalinks.Input(
        data_model=synalinks.datasets.arcagi.get_input_data_model(),
    )
    synthesis = synalinks.PythonSynthesis(
        data_model=synalinks.datasets.arcagi.get_output_data_model(),
        python_script=SEED_SCRIPT,
        default_return_value={"output_grid": [[0]]},  # schema-valid failure floor
    )
    outputs = await synthesis(inputs)
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"arcagi_synthesis_{TASK_NAME}",
        description=f"A Python program that solves ARC-AGI task {TASK_NAME}",
    )

    # 3. Run the seed offline to see the starting point.
    x0, y0 = x_train[0], y_train[0]
    pred = await program(x0)
    solved = pred.get("output_grid") == y0.get("output_grid")
    print("\nSeed output (identity):")
    print(grid_str(pred.get("output_grid")))
    print("\nGround truth:")
    print(grid_str(y0.get("output_grid")))
    print(f"\nSeed solves this example? {solved} (expected False)")
    print(f"Sandbox stderr: {pred.get('stderr')!r}")

    # 4. (Optional) Evolve the script with OMEGA.
    if not RUN_TRAINING:
        print(
            "\nSet RUN_TRAINING=True to evolve the script with OMEGA "
            "(needs a language model + embedding model; slow)."
        )
        return

    # Code synthesis needs a capable code model. Validated with
    # gemini/gemini-3.5-flash (solves this task in the first generation);
    # gemini/gemini-3.1-flash-lite-preview is cheaper but weaker, and
    # ollama/mistral runs locally but rarely lands a correct script.
    code_model = synalinks.LanguageModel(model="gemini/gemini-3.5-flash")
    embedding_model = synalinks.EmbeddingModel(model="ollama/all-minilm")

    reward = synalinks.rewards.ExactMatch(in_mask=["output_grid"])
    program.compile(
        reward=reward,
        optimizer=synalinks.optimizers.OMEGA(
            language_model=code_model,
            embedding_model=embedding_model,
            population_size=8,
            instructions=synalinks.datasets.arcagi.default_instructions(),
        ),
        metrics=[
            synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward"),
        ],
    )
    # ExactMatch saturates at 1.0 the moment a script solves the training
    # set, so cap epochs generously and let EarlyStopping halt the run once
    # the reward plateaus; restore_best_variables keeps the solving script.
    history = await program.fit(
        x=x_train,
        y=y_train,
        epochs=20,
        validation_split=0.2,
        batch_size=1,
        verbose=1,
        callbacks=[
            synalinks.callbacks.EarlyStopping(
                monitor="reward",
                mode="max",
                patience=2,
                restore_best_variables=True,
            ),
        ],
    )
    print(f"\nHistory keys: {list(history.history.keys())}")

    # 5. Read the evolved algorithm (interpretable state).
    candidates = sorted(
        synthesis.state.get("best_candidates") or [],
        key=lambda c: c.get("reward", 0),
        reverse=True,
    )
    if candidates:
        best = candidates[0]
        print(f"\nBest candidate reward: {best.get('reward')}")
        print("\n--- evolved python_script ---")
        print(best.get("python_script"))

    # 6. Evaluate on the held-out test input.
    test_pred = await program(x_test[0])
    test_solved = test_pred.get("output_grid") == y_test[0].get("output_grid")
    print(f"\nEvolved program solves the test input? {test_solved}")


if __name__ == "__main__":
    asyncio.run(main())
# --8<-- [end:source]
