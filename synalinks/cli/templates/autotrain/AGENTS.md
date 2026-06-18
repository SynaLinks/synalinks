# Agent guide — autotrain

This is a **Synalinks autotrain** project: a single program + training harness
in `train.py` that you improve by editing levers and letting `program.fit()` +
an in-context optimizer do the learning.

## Start here

- **`AUTOTRAIN.md`** is the playbook — the disciplined experiment loop you run to
  raise `val_reward`. Read it before changing anything and follow it.
- **`README.md`** explains the layout and quickstart.
- **`NOTES.md`** is the running log — record every experiment's hypothesis and
  verdict there; record the numbers in `results.tsv`.

## Prime directives

1. **Levers are constants in `train.py`, never CLI flags.** Change a lever by
   editing the file, then re-run — each experiment is captured by the git diff
   and reverted with `git reset --hard`.
2. **Never touch the held-out test set during the loop.** Iterate on
   `val_reward`. Run `uv run python train.py --test` only to *confirm* a result
   you already like.
3. **One change per experiment**, then measure. Log the why and the verdict in
   `NOTES.md`; fill the `results.tsv` row (commit hash + keep/revert/crash).
4. **Read real failures, not just scores**, before deciding the next lever.

## Commands

```bash
uv sync                          # install
uv run python train.py           # iterate: fit + report val_reward
uv run python train.py --test    # confirm: also evaluate the held-out test set
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
