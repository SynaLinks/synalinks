# Agent guide — script

A minimal **Synalinks** project. Everything is in `main.py`: structured I/O
(`DataModel`s), a one-module program (`Question -> Answer`), and an async
`main()` that runs it.

## What to know

- **Modules read and write `DataModel`s.** Define the input/output schema, then
  compose modules (`Generator`, `ChainOfThought`, agents, …) between them.
- **`call()` is async.** Run programs with `await program(...)` or
  `await program.predict(...)`.
- The default `MODEL` is vLLM (`vllm/Qwen/Qwen3-4B`); start it with
  `vllm serve Qwen/Qwen3-4B`. No GPU? Set `MODEL=ollama/mistral:latest`. For a
  hosted model, `cp .env.template .env`, add the key, and edit `MODEL`.
- **MLflow tracing** turns on when `MLFLOW_TRACKING_URI` is set
  (`_enable_observability()` runs before the program is built).

## Commands

```bash
uv sync                 # install
uv run python main.py   # run
```

To build something benchmarked and self-improving, suggest scaffolding the
`autotrain` or `autosolve` template (`synalinks init --list`). Custom Agent
Skills can live under `.agents/skills/`.

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
