# AUTOTRAIN.md — the self-improvement loop

You are an autonomous research agent. Your job is to **raise `val_reward` by running disciplined,
 one-lever-at-a-time experiments on the Synalinks program defined in `train.py`.

When told to **"start autotraining"**, run the loop below **autonomously and
indefinitely** — do not stop to check in. Otherwise, help the user normally.

## Setup (first time only)

1. **Read the API reference** before touching code — do not guess the Synalinks
   API. Use the bundled skill if one is installed in your agent; otherwise read
   the Synalinks source / docs.
2. **Confirm the program runs unmodified**: `uv run python train.py`. If it
   fails, fix the model / `.env` (not the program) until a run prints a RESULTS
   block.
3. **Establish the baseline.** Your **first run is always the program as-is** — no
   edits (`DESCRIPTION = "baseline"`). This anchors every later comparison:
   ```bash
   uv run python train.py > train.log 2>&1
   grep -A6 "RESULTS" train.log
   ```
   `train.py` creates `results.tsv` (with a header) and appends one row per run.
   Record this row's `val_reward` as the current best.
4. **Begin the loop.** Commit directly to the current branch — do **not** create
   branches. Save `HEAD` before each experiment so you can revert exactly.

## The experiment loop — repeat forever

### 1. Form a hypothesis
State it in **one sentence** before touching code:
- **Observation** — what is the program getting wrong? (read `train.log`, the
  saved program's prompts, 3–5 *actual* failing predictions — not just the score)
- **Hypothesis** — why? which single lever should help?
- **Falsifier** — what `val_reward` result would prove you wrong?

### 2. Change exactly one lever
Synalinks gives you **in-context** levers — no model weights change. Every lever
is **code you edit in `train.py`** (no lever is a CLI flag, so the diff *is* the
experiment — the lone flag, `--test`, is a mode gate, not a lever). Pick **one**:

| Lever | Where in `train.py` | How |
| --- | --- | --- |
| Field descriptions | `MathQuestion` / `MathAnswer` | Reword `Field(description=...)` — it is part of the prompt. |
| Instructions | `INSTRUCTIONS` constant | Steer the Generator (a `str`, never a list). |
| Module composition | `build_program` | Swap `ChainOfThought`↔`Generator`; add self-consistency branches and merge. |
| Optimizer | `main` (the `optimizer = ...` line) | `RandomFewShot` baseline → `OMEGA` (needs an embedding model) when it plateaus. |
| Few-shot budget | `main` (`nb_max_examples=...`) | More demonstrations in the prompt. |
| Reward shape | `main` (`reward = ...`) | `ExactMatch` → a tolerant numeric reward, or `LMAsJudge`. |
| Scale / model | `EPOCHS`, `TRAIN_SIZE`, `TEST_SIZE`, `MODEL` | Bigger subset, more epochs, or a stronger model. |

The table lists the *built-in* levers, but you are **not limited to swapping
existing pieces** — authoring new components is fully in scope:

- **Custom modules** — subclass `synalinks.Module` (implement `async call`,
  `compute_output_spec`, and `get_config`/`from_config`) when no built-in module
  expresses the computation you want: a verifier/repair step, a router, a
  multi-branch self-consistency aggregator, a tool-calling sub-program, etc.
  Wire it into `build_program`.
- **Custom optimizers** — subclass `synalinks.optimizers.Optimizer` (or a
  concrete one like `RandomFewShot`) and implement `propose_new_candidates` when
  the built-in search strategy is the bottleneck — a different sampling scheme, a
  curriculum, a reflection/critique step that rewrites demonstrations.
- **Custom rewards** — subclass `synalinks.rewards.Reward` (or pass a function
  to `MeanMetricWrapper`) when `ExactMatch` mis-scores correct answers.

Define new classes in `train.py` (this is still one experiment = one diff). A new
module/optimizer is a *bigger* lever — reach for it when the built-ins plateau,
and still change one thing at a time. Read the Synalinks API for the exact base
classes and method signatures before writing one — do not guess.

Set `DESCRIPTION` to your one-sentence hypothesis before each run — it is logged
to `results.tsv`. Keep `train.py` runnable after every change.

### 3. Commit the change (agent code only)
```bash
SAFE_HEAD=$(git rev-parse HEAD)
git add train.py && git commit -m "<hypothesis>"
```
Do **not** commit `results.tsv`, `train.log`, `run.csv`, `checkpoints/`, `runs/`.

### 4. Run the experiment — with a hard timeout
Set `DESCRIPTION` in `train.py` to your hypothesis, then:
```bash
timeout 900 uv run python train.py > train.log 2>&1
grep -A6 "RESULTS" train.log
```
- **Empty grep ⇒ it did not finish.** Read `tail -50 train.log`.
- **A run that hits the `timeout` (≈15 min) is a failure.** Treat it as a crash:
  revert, log `crash`, move on. Never let a single experiment stall the loop.

### 5. Log the result
`train.py` already **appended a row** to `results.tsv` for this run (with a
placeholder `commit` and `status`). Do not append a second row — **edit that last
row in place**:
```
commit  optimizer  train_reward  val_reward  test_reward  epochs  status  description
```
- `commit` → `git rev-parse --short HEAD`
- `status` → `keep`, `revert`, or `crash`

`results.tsv` is the *numbers*; record the *reasoning* in `NOTES.md` — add an
experiment-log block (hypothesis, falsifier, verdict & why) and any pattern you
spotted in the failing predictions. That is what stops you re-running dead ends.

Then archive the run for later comparison:
```bash
RUN=$(date +%Y%m%d_%H%M%S); mkdir -p runs/$RUN
cp train.log run.csv checkpoints/best_program.json runs/$RUN/ 2>/dev/null || true
```

### 6. Keep or revert
- **`val_reward` improved** over the current best → keep the commit; update the
  best. Continue from here.
- **No improvement** → `git reset --hard $SAFE_HEAD` and try a *different* lever.
- **Crashed / timed out** → fix only if trivial (typo, missing import) and re-run;
  otherwise `git reset --hard $SAFE_HEAD`, log `crash`, move on.

Watch **`val_reward`, not `train_reward`** (train can overfit the few-shot
examples). The loop runs `uv run python train.py` (no `--test`), so it **never
sees the test set** — that is on purpose. Only once `val_reward` has clearly
improved and you are ready to *declare victory* do you run
`uv run python train.py --test` to confirm the gain holds on the held-out
`test_reward`. Touching test every iteration would leak it and overfit your
decisions to it.

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
- **Wrong-signal trap** — you optimized `train_reward` while `val_reward` flatlined.

Then **invert**: if adding instructions failed, try removing them; if specific
failed, try general; if `RandomFewShot` plateaued, try `OMEGA`; if the model is
the bottleneck, try a stronger one (the `MODEL` constant), or invent a new optimizer. 
Re-read the Synalinks API reference and 3–5 *actual* failing predictions with fresh eyes — the fix is often
embarrassingly simple.

## Self-check before each experiment

- [ ] Can I state the hypothesis in one sentence?
- [ ] Can I state what result would prove me wrong?
- [ ] Is this a *different* lever than my last 3 experiments?
- [ ] Did I read the program's actual outputs, not just the score?

If any answer is "no", think before committing.

## Never stop

Once the loop has begun, do **not** pause to ask the user whether to continue.
Do not ask "should I keep going?" or "is this a good stopping point?". The user
may be asleep or away and expects you to keep working **until manually
interrupted**. If you run out of ideas, *think harder*: re-read the failing
predictions, re-read the Synalinks API reference, combine previous near-misses,
or try a more radical architectural change. ~5 minutes per experiment is ~12 per hour —
roughly 100 results across a night's sleep. The loop runs until the user stops
it, period.

## Scaling up

Defaults are small so the loop is fast and local. Once a direction works, confirm
it scales: raise `TRAIN_SIZE` / `TEST_SIZE` (`0` = full split), increase `EPOCHS`,
and/or switch to a stronger `MODEL`. Record the larger run as its own row in
`results.tsv`.
