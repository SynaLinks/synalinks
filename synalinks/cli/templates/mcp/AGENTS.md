# Agent guide — mcp

A Synalinks `FunctionCallingAgent` exposed as an **MCP server** with **FastMCP**.
Everything is in `main.py`: the agent, the MCP tool (`solve`), and the server.

## What to know

- **The MCP tool is `solve(query: str) -> NumericalAnswer`.** Its name,
  signature, and docstring become the schema + prompt the calling LM sees.
  Prefer scalar argument types (`str`, `int`, …) so the LM sees flat top-level
  parameters; the body wraps `query` into the program's `Query` model.
- **Two layers of "tool".** `calculate` is the *inner* tool the agent calls
  (returns a dict with a `log`, never raises). `solve` is the *outer* MCP tool —
  it **raises** on failure so the client LM gets a structured error.
- **Build once, serve forever.** `build_and_save_program` writes `program.json`;
  the lifespan only *loads* it (and fails fast if missing). Building does **not**
  call the LM — even for an agent — so it runs offline (no model needed until a
  tool call).
- **Reach the program via `Context.lifespan_context["program"]`** inside the
  tool; the lifespan yields it once at startup.
- **Mask the trajectory:** `result.out_mask(mask=["messages"])` before
  validating against `NumericalAnswer`.
- **Model = vLLM by default** (`MODEL=vllm/Qwen/Qwen3-4B`), base from
  `HOSTED_VLLM_API_BASE` (must include `/v1`). No GPU? `MODEL=ollama/mistral:latest`.
- **MLflow tracing** turns on when `MLFLOW_TRACKING_URI` is set
  (`_enable_observability()` runs in the lifespan before the program loads).
- **Transport:** stdio by default (what clients launch). For remote, use
  `mcp.run(transport="http", host=..., port=...)`.

## Commands

```bash
uv sync                       # install
uv run python main.py build   # build the agent artifact (offline; program.json)
vllm serve Qwen/Qwen3-4B      # local model server (needed at request time)
uv run python main.py         # serve over stdio
```

Swap `calculate` for your own tools, or replace the agent with any Synalinks
program — the MCP layer is unchanged. Custom Agent Skills can live under
`.agents/skills/`.

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
   then re-run your own server to confirm the failure is gone.

5. **Open a PR** from the checkout, then restore the released dependency here:

   ```bash
   git -C synalinks commit -am "fix: <what you fixed>"
   git -C synalinks push -u origin HEAD
   (cd synalinks && gh pr create --fill)
   uv remove synalinks && uv add synalinks   # drop the local override
   ```
