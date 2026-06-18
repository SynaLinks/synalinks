"""The Synalinks program *and* its training harness, in one file.

Every *lever* is a constant in this file, not a CLI flag. That is deliberate: the
autotrain loop *edits this file and commits it*, so each experiment is fully
captured by the diff and reproducible from the code alone (`git reset --hard`
undoes a run; a stray `--flag` would not).

    uv run python train.py            # iterate: fit + report VAL reward
    uv run python train.py --test     # confirm: also evaluate the held-out TEST set

The only flag is `--test`, and it is a *mode gate*, not a lever: by default the
held-out test set is never touched, so you cannot tune toward it during the loop.
Run `--test` only to confirm a result you already like on `val_reward`.

What it does, in order:
  1. load a GSM8K subset (small by default so it runs locally),
  2. build the program (see ``build_program`` below) and compile it with a
     reward + optimizer,
  3. fit() with EarlyStopping + checkpointing + CSV logging,
  4. (only with --test) evaluate() on the held-out test subset,
  5. print a RESULTS block and append one row to results.tsv.

To run an experiment, edit a lever in the CONFIG block (or the program /
optimizer below), then run. See ``AUTOTRAIN.md`` for the loop around it.
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

import synalinks

ROOT = Path(__file__).resolve().parent  # results/checkpoints live next to this file


# ---------------------------------------------------------------------------
# CONFIG — the levers. Edit in place; the autotrain loop commits this file.
# ---------------------------------------------------------------------------

MODEL = "ollama/llama3.2:latest"
EMBEDDING_MODEL = "ollama/mxbai-embed-large"  # used only by the OMEGA optimizer

EPOCHS = 4
TRAIN_SIZE = 50  # examples used for training (0 = all)
TEST_SIZE = 50  # examples used for evaluation (0 = all)

# Steer the Generator (a str, never a list). None keeps the program's default.
INSTRUCTIONS = None

# One-line note for this run, recorded in results.tsv. Set it to your hypothesis.
DESCRIPTION = "baseline"


# ---------------------------------------------------------------------------
# The program — this is what the autotrain loop optimizes. Every field
# description below is part of the prompt; rewording one is a research lever.
# ---------------------------------------------------------------------------


class MathQuestion(synalinks.DataModel):
    question: str = synalinks.Field(
        description="The math word problem to solve",
    )


class MathAnswer(synalinks.DataModel):
    answer: float = synalinks.Field(
        description="The final numerical answer to the problem",
    )


async def build_program(lm):
    """Build the program we optimize.

    A single ``ChainOfThought`` that reads a ``MathQuestion`` and produces a
    ``MathAnswer`` (the chain-of-thought adds a ``thinking`` field of its own).
    Swap it for a plain ``Generator``, add self-consistency branches, or wire
    extra modules here — this function is the "module composition" lever.
    """
    inputs = synalinks.Input(data_model=MathQuestion)
    outputs = await synalinks.ChainOfThought(
        data_model=MathAnswer,
        language_model=lm,
        instructions=INSTRUCTIONS,
    )(inputs)
    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_solver",
        description="Solves GSM8K math word problems.",
    )


# ---------------------------------------------------------------------------
# The harness.
# ---------------------------------------------------------------------------


def _head(x, y, n):
    """Return the first n (x, y) pairs, or all of them when n == 0."""
    return (x, y) if not n else (x[:n], y[:n])


def _fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


async def main():
    parser = argparse.ArgumentParser(description="Train the program on GSM8K.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="also evaluate on the held-out TEST set. Omit it during the loop so "
        "the test set never leaks into your decisions; use it only to confirm a "
        "result you already like on val_reward.",
    )
    args = parser.parse_args()

    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model=MODEL)

    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
    x_train, y_train = _head(x_train, y_train, TRAIN_SIZE)
    x_test, y_test = _head(x_test, y_test, TEST_SIZE)

    program = await build_program(lm)

    # The optimizer lever — swap this line to change strategy:
    optimizer = synalinks.optimizers.RandomFewShot(nb_max_examples=3)
    # OMEGA needs an embedding model for its diversity (DNS) metric:
    # em = synalinks.EmbeddingModel(model=EMBEDDING_MODEL)
    # optimizer = synalinks.optimizers.OMEGA(language_model=lm, embedding_model=em)

    reward = synalinks.ExactMatch(in_mask=["answer"])
    program.compile(
        optimizer=optimizer,
        reward=reward,
        metrics=[synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward")],
    )

    checkpoint = ROOT / "checkpoints" / "best_program.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = await program.fit(
        x=x_train,
        y=y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            synalinks.callbacks.EarlyStopping(
                monitor="val_reward",
                mode="max",
                patience=2,
                stop_at=1.0,
                restore_best_variables=True,
            ),
            synalinks.callbacks.ProgramCheckpoint(
                filepath=str(checkpoint),
                monitor="val_reward",
                save_best_only=True,
            ),
            synalinks.callbacks.CSVLogger(filepath=str(ROOT / "run.csv")),
        ],
    )

    # The held-out test set is only ever touched with --test, so it cannot leak
    # into the iterative loop. Without it, test_reward stays unset.
    test_reward = None
    if args.test:
        eval_metrics = await program.evaluate(x=x_test, y=y_test)
        test_reward = eval_metrics.get("reward", eval_metrics.get("mean_reward"))

    h = history.history
    train_reward = (h.get("reward") or [None])[-1]
    val_reward = (h.get("val_reward") or [None])[-1]

    optimizer_name = type(optimizer).__name__

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"optimizer    : {optimizer_name}")
    print(f"train_reward : {_fmt(train_reward)}")
    print(f"val_reward   : {_fmt(val_reward)}")
    test_display = _fmt(test_reward) if args.test else "(not run — pass --test)"
    print(f"test_reward  : {test_display}")
    print(f"best program : {checkpoint}")

    # Append one row to results.tsv with placeholders, writing the header once.
    # The autotrain loop edits this row in place (AUTOTRAIN.md step 5):
    # commit -> the short hash, status -> the verdict (keep / revert / crash).
    # Do not append a second row.
    results = ROOT / "results.tsv"
    if not results.exists():
        results.write_text(
            "commit\toptimizer\ttrain_reward\tval_reward\t"
            "test_reward\tepochs\tstatus\tdescription\n"
        )
    row = [
        "PENDING",  # -> git rev-parse --short HEAD
        optimizer_name,
        _fmt(train_reward),
        _fmt(val_reward),
        _fmt(test_reward) if args.test else "",  # only filled by --test runs
        EPOCHS,
        "PENDING",  # -> keep | revert | crash
        DESCRIPTION or "(no description)",
    ]
    with results.open("a") as f:
        f.write("\t".join(str(c) for c in row) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
