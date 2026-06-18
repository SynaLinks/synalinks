# Synalinks Starting Template

A minimal starting point for building and **self-improving**
**[Synalinks](https://github.com/SynaLinks/synalinks)** programs *with a coding
agent* (Claude Code, Cursor, Codex, …).

Clone it, open your agent, and start building. The template is deliberately tiny:

- `train.py` — the **program and its training harness in one file**: a small
  runnable example (a GSM8K math word-problem solver), wired to the built-in
  **GSM8K** benchmark with a reward, optimizer, checkpointing, and CSV logging;
- `AUTOTRAIN.md` — the **self-improvement playbook** the agent follows to
  raise `val_reward` against that benchmark, one disciplined experiment at a time;
- `NOTES.md` — an empty scratchpad for your (or the agent's) running notes.

## Quickstart

```bash
# 1. Install (this is a uv project)
uv sync

# 2. Point at a model. Default is local Ollama — no API key needed:
ollama serve
ollama pull llama3.2:latest
#    (or `cp .env.template .env`, add a hosted key, and set MODEL in train.py)

# 3. Train against GSM8K (small subset by default, runs locally)
uv run python train.py          # iterate: reports val_reward
uv run python train.py --test   # confirm a result on the held-out test set
```

Every knob (model, epochs, subset sizes, instructions, optimizer) is a constant
or a small function at the top of `train.py` — no *lever* is a CLI flag, so you
edit the file and re-run and each experiment is captured by the diff. The one
flag, `--test`, is a mode gate: the held-out test set is only evaluated when you
pass it, so it can't leak into the loop.

## Layout

Flat by design — one program+harness file, one playbook, one notes file.

```
train.py          # the program (data models, build_program)
                  #   + the harness (fit/evaluate, results.tsv)
prepare_data.py   # where your data comes from (built-in benchmark, your files,
                  #   or hand-built) — empty stub with examples; train.py uses GSM8K
AUTOTRAIN.md   # the self-improvement loop the agent runs
NOTES.md          # research scratchpad — qualitative companion to results.tsv
pyproject.toml    # uv project metadata
results.tsv / run.csv / checkpoints/ / runs/   # artifacts (git-ignored)
```

## Working with your agent

Ask your agent to:

- *"Add a self-consistency variant of the math solver and compare it to the baseline."*
- *"Turn this into a RAG over my documents."*
- *"Start the autotrain loop and improve `val_reward` on GSM8K."* — it will
  follow [`AUTOTRAIN.md`](./AUTOTRAIN.md).

## License

Apache 2.0, like Synalinks.
