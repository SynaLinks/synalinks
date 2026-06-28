# Agent guide — api

An **OpenAI chat-completions-compatible** endpoint backed by a Synalinks
`FunctionCallingAgent`, served with **FastAPI**. Everything is in `main.py`:
the tool, the agent, the OpenAI-compatible schemas, and the app.

## What to know

- **The endpoint is `POST /v1/chat/completions`.** It accepts the OpenAI
  request shape (`{model, messages: [{role, content}]}`) and returns the OpenAI
  response shape, so OpenAI clients work by setting `base_url` to `.../v1`.
- **Messages → agent.** The request messages become a `synalinks.ChatMessages`,
  fed to a `FunctionCallingAgent`. The agent's trajectory comes back in the
  result's `messages`; the **last** message is the assistant's answer.
- **Build once, serve forever.** `build_and_save_program` writes `program.json`;
  the lifespan only *loads* it (fail fast if missing). Building does **not** call
  the LM — even for an agent — so it runs offline (CI, Docker image-build). Never
  build inside a request handler.
- **DataModels are Pydantic models**, so FastAPI uses the request/response
  schemas directly and publishes them at `/docs`.
- **Status codes are honest:** a guard rejection (`result is None`) is `422`.
  `stream=true` is unsupported and returns `400`.
- **Model = vLLM by default** (`MODEL=vllm/Qwen/Qwen3-4B`); the base URL comes
  from `HOSTED_VLLM_API_BASE`. docker-compose runs vLLM (GPU) on its own service.
  No GPU? Set `MODEL=ollama/mistral:latest`. Override via the `MODEL` env var.
- **MLflow tracing** is enabled at startup whenever `MLFLOW_TRACKING_URI` is set
  (`_enable_observability()` runs before the agent is loaded). Unset = no tracing.

## Commands

```bash
uv sync                       # install
uv run python main.py build   # build the agent artifact (offline; program.json)
uv run python main.py         # serve (or: fastapi dev main.py)
docker compose up --build     # API + vLLM + MLflow
```

Swap `calculate` for your own tools, or replace the `FunctionCallingAgent` with
an RLM / RAG / DeepAgent — the OpenAI-compatible layer is unchanged. Custom
Agent Skills can live under `.agents/skills/`.

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
