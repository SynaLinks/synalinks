"""
# Program Synthesis

Every guide since [Guide 12](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) optimized the same kind of
trainable state: **prose**. An `instruction` string the LM reads on
every call, and a list of few-shot `examples` it imitates. The
optimizer's job was to write better English.

This guide swaps the genome. The trainable variable is now a **Python
program**, and the optimizer's job is to **write better code**. The
module is `PythonSynthesis`, and the only optimizer that can drive it
is an evolutionary one — `OMEGA`.

The mental shift is worth stating plainly up front. With a `Generator`,
the LM is *in the loop at inference time*: every prediction is an LM
call. With `PythonSynthesis`, the LM is *only in the training loop*. At
inference time there is **no LM** — just a deterministic Python script
running in a sandbox. The LM's role is to author and refine that script
during `fit`, not to run it. This is the neuro-symbolic, self-evolving
paradigm that systems like DeepMind's AlphaEvolve popularized: a
language model proposes code, a deterministic evaluator scores it, and
an evolutionary search keeps what works.

## Two Kinds of Trainable State

```mermaid
graph LR
    subgraph Prose["Generator (prose genome)"]
        A1["input"] --> B1["LM call w/ instruction + examples"]
        B1 --> C1["output"]
    end
    subgraph Code["PythonSynthesis (code genome)"]
        A2["input dict"] --> B2["run python_script in sandbox"]
        B2 --> C2["output dict"]
    end
```

Both modules expose a trainable variable and both are optimized by
scoring outputs against a reward. The difference is *what* the variable
holds and *who* runs it:

| | `Generator` | `PythonSynthesis` |
|---|---|---|
| Trainable variable | instruction + examples (text) | a Python script (code) |
| Runs the variable at inference | the LM | the Monty sandbox |
| LM calls per prediction | one | **zero** |
| LM calls during training | scoring/optimizing the prompt | authoring/refining the script |
| Output | whatever the LM emits | whatever the script assigns to `result` |

## What `PythonSynthesis` Does

A `PythonSynthesis` module is a pure data transform: it takes the input
JSON object, runs the current script, and returns the output JSON
object. The contract the script must honor:

- The input is exposed as a dict named `inputs`.
- The script must assign its answer to a variable named `result`
  before it ends.
- `result` must validate against the module's target schema (the
  `data_model` you pass in). If it does not, the run is discarded.

The script runs inside the **Monty** sandbox
(https://github.com/pydantic/monty), a restricted Python interpreter.
This is not a detail you can ignore — the optimizer must write code that
*stays inside the sandbox*, so you need to know its walls:

- Only a subset of the stdlib is importable: `sys`, `os`, `typing`,
  `asyncio`, `re`, `datetime`, `json`, `math`, `pathlib`. Notably
  **`itertools`, `collections`, `functools`, `random` and `time` are
  not** — and neither is any third-party library (`numpy`, `pandas`).
- No `class` definitions and no `match` statements. Use functions and
  `if`/`elif` chains.
- The host filesystem, environment variables and network are
  unreachable: `open()`, `os.system`, `os.environ`, `sys.argv` and
  friends are pruned or gated away.
- Execution is bounded by the module's `timeout` (default 5s) and
  Monty's memory limits.

Because LM-authored code fails *often* — a syntax slip, a forbidden
import, an infinite loop, an output of the wrong shape — the module is
built to **fail soft**. On any error it returns your
`default_return_value` (which must itself validate against the schema)
and surfaces the interpreter's `stdout`/`stderr` on the output, so the
optimizer can read the traceback and fix the bug on the next
generation. A script that loops forever is killed at `timeout` and
treated the same way.

```python
import synalinks

synthesis = synalinks.PythonSynthesis(
    data_model=OutputModel,                  # the schema `result` must match
    python_script=SEED_SCRIPT,               # the starting script (the seed)
    default_return_value={"output_grid": [[0]]},  # used when a run fails
)
```

## Why It Needs `OMEGA` (and not `RandomFewShot`)

`RandomFewShot` ([Guide 15](https://synalinks.github.io/synalinks/guides/Training/)) optimizes a `Generator` by collecting
high-reward `(input, output)` pairs and pasting them into the prompt as
demonstrations. It has no machinery to *write code* — it only curates
examples. Point it at a `PythonSynthesis` module and it has nothing to
do; the script never changes.

`OMEGA` is an **evolutionary** optimizer ([Guide 15](https://synalinks.github.io/synalinks/guides/Training/)
introduced it briefly). It treats the script as a **genome** and runs a genetic
algorithm over a population of script variants:

- **Mutation.** An LM is shown the current script, the inputs, what it
  predicted, the ground truth, and the reward — and asked to rewrite the
  script to do better. This is where the bug-fixing and rule-discovery
  happens.
- **Crossover.** Two high-performing scripts are merged by an LM into a
  new one that combines their strengths.
- **Dominated Novelty Search (DNS).** OMEGA's signature step: a
  candidate is dropped only if it is *both* lower-reward *and* similar
  to a fitter neighbor (similarity measured by embedding the candidate
  with an `embedding_model`). This keeps the population diverse, so the
  search does not collapse onto one mediocre approach — which is exactly
  the failure mode you want to avoid when synthesizing algorithms.

That last point is why `OMEGA` needs **both** a `language_model` (to
write the code) **and** an `embedding_model` (to measure novelty):

```python
optimizer = synalinks.optimizers.OMEGA(
    language_model=code_model,        # writes/refines the scripts
    embedding_model=embedding_model,  # measures candidate novelty (DNS)
    population_size=8,                # how many scripts to keep alive
)
```

## The Evolutionary Loop

```mermaid
flowchart TD
    S["Seed script"] --> POP["Population of scripts"]
    POP --> RUN["Run each in the Monty sandbox over the batch"]
    RUN --> REW["ExactMatch reward on the output"]
    REW --> MUT["OMEGA: LM mutates / crosses over the survivors"]
    MUT --> DNS["DNS: drop candidates that are dominated AND similar"]
    DNS --> POP
    REW --> STOP{"reward == 1.0?"}
    STOP -->|"yes"| KEEP["Keep the winning script"]
```

The seed is just a starting point — usually a trivial identity
function. The optimizer earns its keep by turning that seed into a real
algorithm across generations.

## The Task: ARC-AGI

ARC-AGI (the Abstraction and Reasoning Corpus) is a benchmark of small
visual puzzles. Each task gives you a handful of `(input_grid,
output_grid)` example pairs that demonstrate a hidden transformation
rule, then a fresh `input_grid` you must transform by the same rule. A
grid is a list of integer rows, each integer a color `0`–`9`.

ARC-AGI is a natural fit for program synthesis for three reasons:

1. **The rules are algorithmic.** "Tile the grid into a 3×3 of itself
   wherever a cell is non-zero", "recolor the largest shape", "reflect
   across the diagonal" — these are short programs, not facts to
   memorize. A Python script can express them exactly; a prompt can
   only gesture at them.
2. **Scoring is exact and cheap.** A predicted grid either equals the
   true grid or it does not. `ExactMatch` gives a crisp `0`/`1` reward
   with no LM-as-judge cost.
3. **It is genuinely hard — but tractable for a strong model.** This
   particular task (`007bbfb7`) is one a capable code model can crack:
   in practice `gemini/gemini-3.5-flash` solved it in the first
   generation (see the evolved script below). Treat that as the
   optimistic case, not the rule — most ARC tasks are far harder, and a
   weak or local model will mostly demonstrate the wiring rather than
   land a solution.

Synalinks ships ARC-AGI as a built-in dataset. `load_data` returns the
familiar `(x_train, y_train), (x_test, y_test)` NumPy-array shape, and
provides ready-made input/output `DataModel`s so you do not hand-write
the schema:

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
    task_name="007bbfb7",
    arc_version=1,
)
input_model = synalinks.datasets.arcagi.get_input_data_model()   # examples + input_grid
output_model = synalinks.datasets.arcagi.get_output_data_model()  # output_grid
```

## Wiring It Together

```python
inputs = synalinks.Input(data_model=input_model)   # examples + input_grid
synthesis = synalinks.PythonSynthesis(
    data_model=output_model,                        # result must match this
    python_script=SEED_SCRIPT,
    default_return_value={"output_grid": [[0]]},
)
outputs = await synthesis(inputs)
program = synalinks.Program(inputs=inputs, outputs=outputs)

reward = synalinks.rewards.ExactMatch(in_mask=["output_grid"])
program.compile(
    reward=reward,
    optimizer=synalinks.optimizers.OMEGA(
        language_model=code_model,
        embedding_model=embedding_model,
        population_size=8,
        instructions=synalinks.datasets.arcagi.default_instructions(),
    ),
    metrics=[synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward")],
)
history = await program.fit(
    x=x_train,
    y=y_train,
    epochs=20,  # a cap; EarlyStopping ends the run as soon as it solves
    validation_split=0.2,
    callbacks=[
        synalinks.callbacks.EarlyStopping(
            monitor="reward", mode="max", patience=2, restore_best_variables=True
        ),
    ],
)
```

Note the reward's `in_mask=["output_grid"]`: the module's output also
carries `stdout`/`stderr` fields, and we score *only* the grid.

The `EarlyStopping` callback matters more here than in prose tuning.
`ExactMatch` is a `0`/`1` reward, so it jumps straight to its `1.0`
ceiling the instant a script solves the training set — often in the
first generation with a capable code model. Set a generous `epochs`
cap and let `EarlyStopping(mode="max")` halt the run once the reward
stops improving; `restore_best_variables=True` keeps the solving
script. Without it you keep paying for LM mutation calls that
re-evolve an already-perfect solution.

## The Payoff: Interpretable State

After training, the learned state is not an inscrutable tensor — it is a
**Python script you can read**. Pull it off the module's variable:

```python
best = sorted(
    synthesis.state.get("best_candidates") or [],
    key=lambda c: c.get("reward", 0),
    reverse=True,
)
print(best[0]["python_script"])  # the highest-scoring evolved algorithm
```

This is the same property the [Training guide](https://synalinks.github.io/synalinks/guides/Training/) emphasized for prompts,
taken to its conclusion: the artifact of training is source code. You
can read it, audit it, copy it into a regular module, and run it forever
with zero LM calls.

Here is a script OMEGA actually evolved for this task, driving the loop
with `gemini/gemini-3.5-flash`. It reached reward `1.0` in the **first
generation** and solved all 5 training examples *and* the held-out test
input — a correct, general fractal self-tiling algorithm:

```python
def transform(inputs):
    input_grid = inputs.get("input_grid")
    n = len(input_grid)
    m = len(input_grid[0])

    # Create the output grid (n*3 x m*3) initialized to 0
    output_grid = [[0] * (m * 3) for _ in range(n * 3)]

    for r in range(n):
        for c in range(m):
            if input_grid[r][c] != 0:
                # Stamp a copy of the whole input into block (r, c)
                for br in range(n):
                    for bc in range(m):
                        output_grid[r * 3 + br][c * 3 + bc] = input_grid[br][bc]

    return {"output_grid": output_grid}

result = transform(inputs)
```

Read that against the seed (an identity that just echoes the input):
the optimizer discovered the rule — "tile the grid into a 3×3 of itself,
placing a copy wherever the input cell is non-zero" — purely from the
reward signal and the example pairs in the mutation prompt.

## Honest Expectations and Failure Modes

- **ARC is hard; the seed will score 0.** The identity seed echoes its
  input, so unless the task *is* the identity, the first reward is `0`.
  Improvement shows up across generations, and only if the code model is
  strong enough to discover the rule. A small local model will mostly
  produce sandbox errors; reach for a capable **code** model for real
  attempts.
- **`ExactMatch` is unforgiving.** One wrong cell drops the reward to
  `0`. There is no partial credit. This is correct for ARC but means the
  reward signal is sparse — large populations and more generations help.
- **The sandbox rejects a lot.** Scripts that import `numpy`, define a
  `class`, or use `itertools` will fail and fall back to the default.
  Read `stderr` on the predictions to see why — that feedback is exactly
  what OMEGA's mutation step consumes.
- **Cost scales with `population_size × generations × batch`.** Each
  candidate is an LM authoring call. Start small while you debug the
  wiring, then scale up.

## Best Practices, Distilled

1. **Make the seed runnable, not clever.** A trivial identity script
   that returns a schema-valid `result` gives the optimizer a working
   scaffold to mutate. A broken seed wastes the first generations on
   syntax repair.
2. **Pick a strong code model for the LM.** This optimizer writes
   programs; the bottleneck is the model's coding ability, not its prose.
3. **Keep `default_return_value` schema-valid and minimal.** It is the
   floor every failed run lands on; `{"output_grid": [[0]]}` is enough.
4. **Read `stderr`.** When rewards stay at `0`, the predictions'
   `stderr` tells you whether the model is hitting a sandbox wall versus
   getting the algorithm wrong.
5. **Stop early.** Pair a generous `epochs` cap with
   `EarlyStopping(monitor="reward", mode="max", restore_best_variables=True)`.
   Because `ExactMatch` saturates at `1.0`, a capable model often solves
   in the first generation — without early stopping you pay for every
   remaining epoch re-evolving a solution you already have.
6. **Save after every run.** `program.save("path.json")` writes the
   evolved script to a human-readable file — `diff` it across runs to
   watch the algorithm take shape.

## API References

- [PythonSynthesis](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Synthesis%20Modules/PythonSynthesis%20module/)
- [OMEGA](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/OMEGA/)
- [ExactMatch](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/ExactMatch%20reward/)
- [EarlyStopping](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/EarlyStopping/)
- [Program.fit](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/Program%20training%20API/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# Flip to True to run the (slow, LM-driven) OMEGA training loop. Left off
# by default so the guide's forward pass runs offline in seconds.
RUN_TRAINING = False

# The ARC-AGI task to attack. "007bbfb7" is the classic "fractal
# self-tiling" task: the 3x3 input is stamped into a 3x3 layout of itself,
# placing a copy wherever the input cell is non-zero (a 9x9 output).
TASK_NAME = "007bbfb7"

# The seed: an identity transform. OMEGA rewrites the body of `transform`
# across generations to discover the rule mapping input_grid -> output_grid.
SEED_SCRIPT = """
def transform(inputs):
    # Starting point: copy the input grid unchanged. OMEGA will rewrite
    # this body to discover the rule that maps the input grid to the
    # output grid, using the examples carried on the input.
    return {"output_grid": inputs.get("input_grid")}

result = transform(inputs)
"""


def _grid_str(grid):
    """Render a small integer grid as text for printing.

    Cells may come back as ints (from the sandbox, via JSON) or as
    `Color` enum members (from the typed ground-truth DataModel); coerce
    to int so both sides print identically.
    """
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


async def main():
    load_dotenv()
    synalinks.clear_session()

    # -------------------------------------------------------------------------
    # Step 1: Load the ARC-AGI task
    # -------------------------------------------------------------------------
    print("=" * 60)
    print(f"Step 1: Load ARC-AGI task {TASK_NAME}")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
        task_name=TASK_NAME,
        arc_version=1,
    )
    print(f"\nTraining examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")

    # -------------------------------------------------------------------------
    # Step 2: Build the PythonSynthesis program (no LM involved)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Build the PythonSynthesis program")
    print("=" * 60)

    inputs = synalinks.Input(
        data_model=synalinks.datasets.arcagi.get_input_data_model(),
    )
    synthesis = synalinks.PythonSynthesis(
        data_model=synalinks.datasets.arcagi.get_output_data_model(),
        python_script=SEED_SCRIPT,
        # The floor every failed/timed-out run lands on. Must be schema-valid.
        default_return_value={"output_grid": [[0]]},
    )
    outputs = await synthesis(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"arcagi_synthesis_{TASK_NAME}",
        description=f"A Python program that solves ARC-AGI task {TASK_NAME}",
    )
    print("\nProgram built. The trainable variable is the Python script itself.")

    # -------------------------------------------------------------------------
    # Step 3: Run the seed (offline) to see the starting point
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Run the seed script (no LM call)")
    print("=" * 60)

    x0, y0 = x_train[0], y_train[0]
    pred = await program(x0)
    solved = pred.get("output_grid") == y0.get("output_grid")

    print("\nSeed output (identity — just echoes the input grid):")
    print(_grid_str(pred.get("output_grid")))
    print("\nGround-truth output grid:")
    print(_grid_str(y0.get("output_grid")))
    print(f"\nSeed solves this example? {solved}  (expected False)")
    print(f"Sandbox stderr: {pred.get('stderr')!r}  (empty == the seed ran cleanly)")

    # -------------------------------------------------------------------------
    # Step 4 (optional): Evolve the script with OMEGA
    # -------------------------------------------------------------------------
    if not RUN_TRAINING:
        print("\n" + "=" * 60)
        print("Step 4: Training is OFF")
        print("=" * 60)
        print(
            "\nSet RUN_TRAINING=True at the top of this file to evolve the\n"
            "script with OMEGA. Training needs a language model (ideally a\n"
            "strong code model) and an embedding model, and is slow: cost\n"
            "scales with population_size x generations x batch size."
        )
        return

    print("\n" + "=" * 60)
    print("Step 4: Evolve the script with OMEGA")
    print("=" * 60)

    # The LM authors/refines the scripts; the embedding model powers DNS
    # novelty. Code synthesis needs a capable code model: a strong one
    # can discover this task's rule in a single generation. Validated with
    # `gemini/gemini-3.5-flash`; `gemini/gemini-3.1-flash-lite-preview` is
    # cheaper but weaker. `ollama/mistral` runs locally but rarely lands a
    # correct script.
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

    # `ExactMatch` saturates at 1.0 the moment a script solves every training
    # example, so set a generous epoch cap and let `EarlyStopping` halt the run
    # as soon as the reward plateaus — `restore_best_variables` keeps the best
    # (solving) script. Without this you burn LM calls re-evolving an already
    # perfect solution. We monitor `reward` (train) rather than `val_reward`
    # because the leave-one-out split leaves only a single, noisy validation
    # example.
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
    print(f"\nTraining complete! History keys: {list(history.history.keys())}")

    # -------------------------------------------------------------------------
    # Step 5: Read the evolved algorithm (interpretable state)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: The learned Python script")
    print("=" * 60)

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
    else:
        print("\nNo evolved candidates yet — the active script is:")
        print(synthesis.state.get("python_script"))

    # -------------------------------------------------------------------------
    # Step 6: Run the evolved program on the held-out test input
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Evaluate on the held-out test input")
    print("=" * 60)

    test_pred = await program(x_test[0])
    test_solved = test_pred.get("output_grid") == y_test[0].get("output_grid")
    print(f"\nEvolved program solves the test input? {test_solved}")


if __name__ == "__main__":
    asyncio.run(main())
