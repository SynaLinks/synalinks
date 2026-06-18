"""The Synalinks program *and* its evaluation harness, in one file.

This is **autosolve**: a self-improving system where *you, the coding
agent, are the optimizer*. There is **no training loop** here — no
``program.fit()``, no in-context optimizer, no few-shot search. Instead you read
the program's actual failures and **rewrite the program itself** (its
instructions, field descriptions, module composition, or whole new modules),
then re-run this file to measure whether the reward went up. The optimization
loop is your own reasoning; this script is just the measuring instrument.

    uv run python evaluate.py            # iterate: score the DEV set, dump failures
    uv run python evaluate.py --test     # confirm: also score the held-out TEST set

The only flag is ``--test``, and it is a *mode gate*, not a lever: by default the
held-out test set is never touched, so you cannot tune toward it while you
iterate. Run ``--test`` only to confirm a result you already like on
``dev_reward``.

What it does, in order:
  1. turn on Synalinks logging so every module call is observable (see
     ``LOG_LEVEL``) — the run prints each module's inputs/outputs to stdout,
  2. load a GSM8K subset (small by default so it runs locally — even ONE sample
     is a valid loop; see ``DEV_SIZE``),
  3. build the program (see ``build_program`` below) — the thing you edit,
  4. run it over the DEV set with ``predict()`` and score every example with the
     reward, writing per-example results to ``predictions_dev.jsonl`` (READ
     THESE — they are how you find the next thing to fix),
  5. (only with --test) score the held-out TEST subset,
  6. print a RESULTS block and append one row to ``results.tsv``.

The autosolve loop runs this as ``uv run python evaluate.py > eval.log 2>&1``,
so ``eval.log`` holds the full module-by-module trace. To understand *why* a
prediction is wrong, read it — pair it with ``DEV_SIZE = 1`` to study a single
problem in full detail.

Every *lever* is code in this file, not a CLI flag. That is deliberate: each
experiment is fully captured by the diff and reproducible from the code alone
(`git reset --hard` undoes one). See ``AUTOSOLVE.md`` for the loop around it.
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

import synalinks

ROOT = Path(__file__).resolve().parent  # results/predictions live next to this file


# ---------------------------------------------------------------------------
# CONFIG — the levers. Edit in place; the autosolve loop commits this file.
# ---------------------------------------------------------------------------

MODEL = "ollama/llama3.2:latest"

# How much Synalinks logs while the program runs. `enable_logging` attaches a
# Logger hook that prints every module's inputs/outputs as JSON during the run —
# that trace is how you, the optimizer, understand what the program actually did.
# It goes to stdout, so the loop's `> eval.log 2>&1` redirect saves it in
# eval.log. "info" = the data flowing through each module; "debug" = also the
# symbolic schemas. None = silent.
LOG_LEVEL = "info"

# Autosolve works with as few as ONE sample: set DEV_SIZE = 1 and read the
# full module-by-module trace in `eval.log` to see exactly where that one problem
# goes wrong, fix the program, re-run. Larger DEV_SIZE = a less noisy score;
# smaller = a faster, deeper look. 0 = the whole split.
DEV_SIZE = 50  # examples you iterate against (0 = all). Never the test set.
TEST_SIZE = 50  # examples used only by --test to confirm a result (0 = all).

# Steer the Generator (a str, never a list). None keeps the program's default.
INSTRUCTIONS = None

# One-line note for this run, recorded in results.tsv. Set it to your hypothesis.
DESCRIPTION = "baseline"


# ---------------------------------------------------------------------------
# The program — this is what YOU optimize. There is no optimizer to lean on:
# every improvement is an edit to the code below. Each field description is part
# of the prompt; ``build_program`` is where architecture lives.
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
    """Build the program we are hill-climbing by hand.

    A single ``ChainOfThought`` that reads a ``MathQuestion`` and produces a
    ``MathAnswer`` (the chain-of-thought adds a ``thinking`` field of its own).
    This function is the heart of autosolve: with no training loop, the way
    you raise the reward is to *change what this builds* — swap modules, add a
    verifier/repair step, branch for self-consistency and merge, or author a
    custom ``synalinks.Module``. Read ``AUTOSOLVE.md`` for the lever menu.
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
# The harness. No fit(), no optimizer — just run the program and score it.
# ---------------------------------------------------------------------------


def _head(x, y, n):
    """Return the first n (x, y) pairs, or all of them when n == 0."""
    return (x, y) if not n else (x[:n], y[:n])


def _fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def _json_of(item):
    """Best-effort JSON dict for a DataModel / JsonDataModel / None."""
    if item is None:
        return None
    getter = getattr(item, "get_json", None)
    return getter() if getter else item


async def solve(program, reward, x, y, split):
    """Run the program over (x, y), score each example, dump per-example results.

    Returns ``(mean_reward, records)``. Writes ``predictions_<split>.jsonl`` next
    to this file (git-ignored) so you can read the *actual* failures — the whole
    point of the autosolve loop is to fix what you see there, not the score.
    """
    y_pred = await program.predict(x, verbose=0)

    records = []
    for x_i, y_true_i, y_pred_i in zip(x, y, y_pred):
        # Reuse the reward so swapping its shape (a lever) also changes what
        # counts as a failure here. A failed pipeline (None) scores 0.0.
        if y_pred_i is None:
            r = 0.0
        else:
            r = float(await reward(y_true_i, y_pred_i))
        records.append(
            {
                "reward": r,
                "input": _json_of(x_i),
                "expected": _json_of(y_true_i),
                "predicted": _json_of(y_pred_i),
            }
        )

    mean_reward = (
        sum(rec["reward"] for rec in records) / len(records) if records else 0.0
    )

    out = ROOT / f"predictions_{split}.jsonl"
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    failures = [rec for rec in records if rec["reward"] < 1.0]
    print(
        f"\n{split}: {len(records) - len(failures)}/{len(records)} correct "
        f"-> {split}_reward = {_fmt(mean_reward)}  (per-example: {out.name})"
    )
    # Surface a few real failures right in the log so they are impossible to skip.
    for rec in failures[:3]:
        print(
            f"  - expected={rec['expected']}  predicted={rec['predicted']}\n"
            f"    Q: {rec['input']}"
        )

    return mean_reward, records


async def main():
    parser = argparse.ArgumentParser(description="Evaluate the program on GSM8K.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="also score the held-out TEST set. Omit it while you iterate so the "
        "test set never leaks into your decisions; use it only to confirm a "
        "result you already like on dev_reward.",
    )
    args = parser.parse_args()

    load_dotenv()
    synalinks.clear_session()

    # Turn on Synalinks logging so the program's behavior is observable: the
    # Logger hook (on by default) prints each module's inputs/outputs as it runs.
    # It streams to stdout, so the loop's `> eval.log 2>&1` redirect captures the
    # full trace in eval.log. This is autosolve's microscope — especially
    # with DEV_SIZE = 1.
    if LOG_LEVEL:
        synalinks.enable_logging(log_level=LOG_LEVEL)

    lm = synalinks.LanguageModel(model=MODEL)

    # train split -> DEV (the set you iterate on); test split -> held-out TEST.
    (x_dev, y_dev), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
    x_dev, y_dev = _head(x_dev, y_dev, DEV_SIZE)
    x_test, y_test = _head(x_test, y_test, TEST_SIZE)

    program = await build_program(lm)

    # The reward is the only judge here — there is no optimizer to compile for.
    # Swapping it (ExactMatch -> tolerant numeric -> LMAsJudge) is a lever.
    reward = synalinks.ExactMatch(in_mask=["answer"])

    dev_reward, _ = await solve(program, reward, x_dev, y_dev, "dev")

    # The held-out test set is only ever touched with --test, so it cannot leak
    # into the iterative loop. Without it, test_reward stays unset.
    test_reward = None
    if args.test:
        test_reward, _ = await solve(program, reward, x_test, y_test, "test")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"program     : {program.name}")
    print(f"dev_reward  : {_fmt(dev_reward)}")
    test_display = _fmt(test_reward) if args.test else "(not run — pass --test)"
    print(f"test_reward : {test_display}")

    # Append one row to results.tsv with placeholders, writing the header once.
    # The autosolve loop edits this row in place (AUTOSOLVE.md step 5):
    # commit -> the short hash, status -> the verdict (keep / revert / crash).
    # Do not append a second row.
    results = ROOT / "results.tsv"
    if not results.exists():
        results.write_text(
            "commit\tprogram\tdev_reward\ttest_reward\tstatus\tdescription\n"
        )
    row = [
        "PENDING",  # -> git rev-parse --short HEAD
        program.name,
        _fmt(dev_reward),
        _fmt(test_reward) if args.test else "",  # only filled by --test runs
        "PENDING",  # -> keep | revert | crash
        DESCRIPTION or "(no description)",
    ]
    with results.open("a") as f:
        f.write("\t".join(str(c) for c in row) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
