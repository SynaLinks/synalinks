# Synalinks Autosolve

A minimal starting point for **self-improving**
**[Synalinks](https://github.com/SynaLinks/synalinks)** programs *with a coding
agent* (Claude Code, Cursor, Codex, …) — where **the agent itself is the
optimizer**.

This is the sibling of the *autotrain* template, with one key difference:
**there is no training loop.** Autotrain edits levers and lets
`program.fit()` + an in-context optimizer (RandomFewShot / OMEGA) do the
learning. **Autosolve** has no `fit()`, no optimizer, no few-shot search —
instead the coding agent reads the program's *actual failures* and **rewrites the
program itself** (instructions, modules, architecture), measuring each edit with
`evaluate.py`. The optimization loop is the agent's reasoning.

The template is deliberately tiny:

- `evaluate.py` — the **program and its evaluation harness in one file**: a small
  runnable example (a GSM8K math word-problem solver), wired to the built-in
  **GSM8K** benchmark with a reward. It turns on Synalinks logging (so every
  module call is observable), runs the program, scores every example, and dumps
  per-example results so you can read the failures. It works on **as few as one
  sample** — read the full module-by-module trace in `eval.log` to see exactly
  what happened;
- `AUTOSOLVE.md` — the **self-improvement playbook** the agent follows to raise
  `dev_reward`, one disciplined experiment at a time;
- `NOTES.md` — an empty scratchpad for your (or the agent's) running notes.

## Quickstart

```bash
# 1. Install (this is a uv project)
uv sync

# 2. Serve a model with vLLM (the default), on its OpenAI endpoint (:8000):
vllm serve Qwen/Qwen3-4B
#    (no GPU? set MODEL=ollama/mistral:latest in evaluate.py and run `ollama serve`)
#    (hosted model? `cp .env.template .env`, add a key, set MODEL in evaluate.py)
#    (optional: set MLFLOW_TRACKING_URI in .env to trace runs to MLflow)

# 3. Solve GSM8K (small subset by default, runs locally)
uv run python evaluate.py          # iterate: scores dev_reward, dumps failures
uv run python evaluate.py --test   # confirm a result on the held-out test set
```

Every knob (model, log level, subset sizes, instructions, reward) is a constant
or a small function at the top of `evaluate.py` — no *lever* is a CLI flag, so you
edit the file and re-run and each experiment is captured by the diff. The one
flag, `--test`, is a mode gate: the held-out test set is only scored when you
pass it, so it can't leak into the loop.

Because the agent (not an optimizer) does the learning here, **observability is
the whole game**: every run turns on Synalinks logging, and the loop redirects it
into `eval.log` — a module-by-module trace of what the program did. Set
`DEV_SIZE = 1` to iterate on a single problem and read its complete trace — the
fastest way to understand a behavior before changing the program.

## How it differs from autotrain

| | autotrain | **autosolve** |
| --- | --- | --- |
| Entry point | `train.py` | `evaluate.py` |
| Learning | `program.fit()` + an optimizer (RandomFewShot / OMEGA) | **none — the coding agent is the optimizer** |
| What you change | levers; the optimizer searches prompts/examples | the **program code** itself (prompts, modules, architecture) |
| Few-shot examples | the optimizer selects them | you hand-write them into `INSTRUCTIONS` |
| Watched metric | `val_reward` | `dev_reward` |
| Held-out check | `--test` → `test_reward` | `--test` → `test_reward` |

Reach for **autosolve** when you want the agent to *engineer* a program that
solves the task (and you can read exactly how it works in the code), and for
**autotrain** when you want an automated in-context optimizer to tune one.

## Layout

Flat by design — one program+harness file, one playbook, one notes file.

```
evaluate.py       # the program (data models, build_program)
                  #   + the harness (predict/score, predictions_*.jsonl, results.tsv)
prepare_data.py   # where your data comes from (built-in benchmark, your files,
                  #   or hand-built) — empty stub with examples; evaluate.py uses GSM8K
AUTOSOLVE.md     # the self-improvement loop the agent runs
NOTES.md          # scratchpad — qualitative companion to results.tsv
pyproject.toml    # uv project metadata
results.tsv / predictions_*.jsonl / eval.log / runs/   # artifacts (git-ignored)
```

## Working with your agent

Ask your agent to:

- *"Add a verify→repair step to the math solver and compare it to the baseline."*
- *"Write a self-consistency variant that samples 3 solutions and takes the majority."*
- *"Start autosolving and improve `dev_reward` on GSM8K."* — it will follow
  [`AUTOSOLVE.md`](./AUTOSOLVE.md).

## License

Apache 2.0, like Synalinks.
