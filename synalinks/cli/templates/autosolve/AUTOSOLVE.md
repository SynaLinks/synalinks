# AUTOSOLVE.md — the self-improvement loop

You are an autonomous problem-solver. Your job is to **raise `dev_reward` by
rewriting the Synalinks program in `evaluate.py` until it solves the task.**

The difference from autotrain: **there is no training loop.** No
`program.fit()`, no in-context optimizer, no few-shot search learns anything for
you. **You are the optimizer.** Every gain comes from *you* reading the actual
failures and editing the program's code — its instructions, field descriptions,
module composition, or brand-new modules. `evaluate.py` only measures; the
hill-climbing is your reasoning.

When told to **"start autosolving"**, run the loop below **autonomously and
indefinitely** — do not stop to check in. Otherwise, help the user normally.

## Setup (first time only)

1. **Read the API reference** before touching code — do not guess the Synalinks
   API. Use the bundled skill if one is installed in your agent; otherwise read
   the Synalinks source / docs.
2. **Confirm the program runs unmodified**: `uv run python evaluate.py`. If it
   fails, fix the model / `.env` (not the program) until a run prints a RESULTS
   block.
3. **Establish the baseline.** Your **first run is always the program as-is** — no
   edits (`DESCRIPTION = "baseline"`). This anchors every later comparison:
   ```bash
   uv run python evaluate.py > eval.log 2>&1
   grep -A5 "RESULTS" eval.log
   ```
   `evaluate.py` creates `results.tsv` (with a header) and appends one row per
   run. Record this row's `dev_reward` as the current best.
4. **Begin the loop.** Commit directly to the current branch — do **not** create
   branches. Save `HEAD` before each experiment so you can revert exactly.

## The solve loop — repeat forever

### 1. Read the failures first
You cannot optimize what you have not read. Two sources, both written every run:
- `predictions_dev.jsonl` — one JSON object per example (`reward`, `input`,
  `expected`, `predicted`). Study **3–5 *actual* failing predictions**.
- `eval.log` — the **module-by-module trace** of the run: with logging on, every
  module's inputs and outputs as JSON (the loop redirects the run into this
  file). This is where you see *why* a prediction is wrong (which step went off,
  what the chain-of-thought actually produced, where a sub-module or tool
  failed). Read it for the failures you picked.

You can run on **as few as one sample** (`DEV_SIZE = 1`): re-run, then read the
whole trace for that single problem end to end. That is the fastest way to
understand a behavior before you change anything — autosolve is built for
it. State in **one sentence** before touching code:
- **Observation** — what is the program getting wrong? (a recurring error type:
  arithmetic slips, misread question, wrong units, right reasoning but malformed
  `answer`, empty/None output, …)
- **Hypothesis** — why? which single change to the program should fix that class
  of failure?
- **Falsifier** — what `dev_reward` result would prove you wrong?

### 2. Change exactly one lever — by editing the program
Since you are the optimizer, every lever is **code you edit in `evaluate.py`**
(no lever is a CLI flag, so the diff *is* the experiment — the lone flag,
`--test`, is a mode gate, not a lever). Pick **one**:

| Lever | Where in `evaluate.py` | How |
| --- | --- | --- |
| Field descriptions | `MathQuestion` / `MathAnswer` | Reword `Field(description=...)` — it is part of the prompt. |
| Instructions | `INSTRUCTIONS` constant | Steer the Generator (a `str`, never a list): spell out the method, the output format, the failure to avoid. |
| Module composition | `build_program` | Swap `ChainOfThought`↔`Generator`; add a verify→repair step; branch into self-consistency samples and merge by majority. |
| Reward shape | `main` (the `reward = ...` line) | `ExactMatch` → a tolerant numeric reward, or `LMAsJudge`, when it mis-scores answers that are actually right. |
| Scale / model | `DEV_SIZE`, `TEST_SIZE`, `MODEL` | Bigger subset for a less noisy signal, or a stronger model. |

**There is no optimizer or few-shot lever** — that is exactly the training-loop
machinery autosolve does without. The replacement for "let the optimizer
search" is **you authoring better program structure**:

- **Custom modules** — subclass `synalinks.Module` (implement `async call`,
  `compute_output_spec`, and `get_config`/`from_config`) when no built-in module
  expresses what you want: a **verifier** that checks the answer and a **repair**
  step that retries on failure, a **router** that sends hard questions down a
  longer chain, a **self-consistency aggregator** over N sampled solutions, a
  tool-calling sub-program that does exact arithmetic. Wire it into
  `build_program`.
- **Custom rewards** — subclass `synalinks.rewards.Reward` (or pass a function
  to `MeanMetricWrapper`) when the reward mis-scores correct answers.

Hand-written demonstrations are fair game too: because there is no few-shot
optimizer, the way to give the model examples is to **write them into the
`INSTRUCTIONS` string yourself** (a worked example or two). Define any new
classes in `evaluate.py` (this is still one experiment = one diff). Read the
Synalinks API for the exact base classes and method signatures before writing
one — do not guess.

Set `DESCRIPTION` to your one-sentence hypothesis before each run — it is logged
to `results.tsv`. Keep `evaluate.py` runnable after every change.

### 3. Commit the change (agent code only)
```bash
SAFE_HEAD=$(git rev-parse HEAD)
git add evaluate.py && git commit -m "<hypothesis>"
```
Do **not** commit `results.tsv`, `eval.log`, `predictions_*.jsonl`, `runs/`.

### 4. Run the experiment — with a hard timeout
Set `DESCRIPTION` in `evaluate.py` to your hypothesis, then:
```bash
timeout 900 uv run python evaluate.py > eval.log 2>&1
grep -A5 "RESULTS" eval.log
```
- **Empty grep ⇒ it did not finish.** Read `tail -50 eval.log`.
- **A run that hits the `timeout` (≈15 min) is a failure.** Treat it as a crash:
  revert, log `crash`, move on. Never let a single experiment stall the loop.

### 5. Log the result
`evaluate.py` already **appended a row** to `results.tsv` for this run (with a
placeholder `commit` and `status`). Do not append a second row — **edit that last
row in place**:
```
commit  program  dev_reward  test_reward  status  description
```
- `commit` → `git rev-parse --short HEAD`
- `status` → `keep`, `revert`, or `crash`

`results.tsv` is the *numbers*; record the *reasoning* in `NOTES.md` — add an
experiment-log block (hypothesis, falsifier, verdict & why) and any pattern you
spotted in the failing predictions. That is what stops you re-running dead ends.

Then archive the run for later comparison:
```bash
RUN=$(date +%Y%m%d_%H%M%S); mkdir -p runs/$RUN
cp eval.log predictions_dev.jsonl runs/$RUN/ 2>/dev/null || true
```

### 6. Keep or revert
- **`dev_reward` improved** over the current best → keep the commit; update the
  best. Continue from here.
- **No improvement** → `git reset --hard $SAFE_HEAD` and try a *different* lever.
- **Crashed / timed out** → fix only if trivial (typo, missing import) and re-run;
  otherwise `git reset --hard $SAFE_HEAD`, log `crash`, move on.

Watch **`dev_reward`, not the test set.** The loop runs `uv run python
evaluate.py` (no `--test`), so it **never sees the test set** — that is on
purpose. Only once `dev_reward` has clearly improved and you are ready to
*declare victory* do you run `uv run python evaluate.py --test` to confirm the
gain holds on the held-out `test_reward`. Touching test every iteration would
leak it and overfit your edits to it.

### 7. Repeat. Never stop until told.

## When stuck (3+ experiments, no gain)

Stop iterating and question assumptions — most failures are one of these traps:

- **Variation trap** — you keep trying versions of the same idea. The
  *hypothesis* is wrong, not the wording. Change levers, not phrasings.
- **Complexity trap** — adding modules/instructions rarely fixes a
  misunderstanding. Ask: what would I *remove*? What is the simplest thing that
  could work?
- **Confirmation trap** — you're explaining away failures. You stated a falsifier
  in step 1; check whether you've already seen it.
- **Not-reading trap** — you optimized against the *score* without re-reading the
  *predictions*. The fix is almost always visible in 3–5 real failures.

Then **invert**: if adding instructions failed, try removing them; if specific
failed, try general; if a single chain plateaued, add a verify→repair step or
self-consistency; if the model is the bottleneck, try a stronger one (the `MODEL`
constant), or author a custom module. Re-read the Synalinks API reference and 3–5
*actual* failing predictions with fresh eyes — the fix is often embarrassingly
simple.

## Self-check before each experiment

- [ ] Did I read the actual predictions *and* the `eval.log` trace, not just the score?
- [ ] Can I state the hypothesis in one sentence?
- [ ] Can I state what result would prove me wrong?
- [ ] Is this a *different* lever than my last 3 experiments?

If any answer is "no", think before committing.

## Never stop

Once the loop has begun, do **not** pause to ask the user whether to continue.
Do not ask "should I keep going?" or "is this a good stopping point?". The user
may be asleep or away and expects you to keep working **until manually
interrupted**. If you run out of ideas, *think harder*: re-read the failing
predictions, re-read the Synalinks API reference, combine previous near-misses,
or try a more radical architectural change. ~5 minutes per experiment is ~12 per
hour — roughly 100 results across a night's sleep. The loop runs until the user
stops it, period.

## Scaling up

Defaults are small so the loop is fast and local. Once a direction works, confirm
it scales: raise `DEV_SIZE` / `TEST_SIZE` (`0` = full split) for a less noisy
signal, and/or switch to a stronger `MODEL`. Record the larger run as its own row
in `results.tsv`.
