# Agent guide — autosolve

This is a **Synalinks autosolve** project: a single program + evaluation harness
in `evaluate.py`. There is **no training loop** — *you, the coding agent, are the
optimizer*. You read the program's actual failures and rewrite the program
itself (instructions, field descriptions, modules, architecture), then re-run to
measure whether the reward went up.

## Start here

- **`AUTOSOLVE.md`** is the playbook — the disciplined loop you run to raise
  `dev_reward`. Read it before changing anything and follow it.
- **`README.md`** explains the layout and how this differs from autotrain.
- **`NOTES.md`** is the running log — record every experiment's hypothesis and
  verdict there; record the numbers in `results.tsv`.

## Prime directives

1. **Observability is the whole game.** Each run writes a full module-by-module
   trace to `eval.log` and per-example results to `predictions_dev.jsonl`. Read
   them — they are how you find the next thing to fix. Set `DEV_SIZE = 1` to
   study one problem in complete detail.
2. **You improve the program by editing code**, not by tuning an optimizer
   (there isn't one). Levers are constants/functions in `evaluate.py`; each
   experiment is captured by the git diff and reverted with `git reset --hard`.
3. **Never touch the held-out test set during the loop.** Iterate on
   `dev_reward`. Run `uv run python evaluate.py --test` only to *confirm* a
   result you already like.
4. **One change per experiment**, then measure. Log the why and the verdict in
   `NOTES.md`; fill the `results.tsv` row (commit hash + keep/revert/crash).

## Commands

```bash
uv sync                                       # install
uv run python evaluate.py > eval.log 2>&1     # iterate: score dev, dump failures + trace
uv run python evaluate.py --test              # confirm: also score the held-out test set
```

Custom Agent Skills can live under `.agents/skills/`.

## Troubleshooting a framework bug

Most failures are in *your* program — fix those here. But if you trace a problem
to **Synalinks itself** (a stack trace inside the `synalinks` package, or a
missing/broken framework feature), fix it at the source and upstream it:

1. **Clone the framework** into this project (the `synalinks/` dir is git-ignored):

   ```bash
   git clone https://github.com/SynaLinks/synalinks.git
   git -C synalinks checkout -b fix/<short-description>
   ```

2. **Point this project at the local checkout** so your runs exercise the fix:

   ```bash
   uv add --editable ./synalinks
   ```

3. **Fix the bug** under `synalinks/synalinks/src/...`, following that repo's
   `CLAUDE.md`. Add or update a colocated `*_test.py` covering the bug.

4. **Verify**: run the framework's tests (`cd synalinks && ./shell/test.sh`),
   then re-run your own program to confirm the failure is gone.

5. **Open a PR** from the checkout, then restore the released dependency here:

   ```bash
   git -C synalinks commit -am "fix: <what you fixed>"
   git -C synalinks push -u origin HEAD
   (cd synalinks && gh pr create --fill)
   uv remove synalinks && uv add synalinks   # drop the local override
   ```
