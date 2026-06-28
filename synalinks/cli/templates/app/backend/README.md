# Backend — Synalinks + FastAPI

The API service for the app: a Synalinks `Generator` program (question → answer)
served with FastAPI, following the build-once / serve-forever pattern.

```bash
uv sync
uv run python main.py build   # build the program artifact (program.json)
uv run python main.py         # serve on :8000  (or: fastapi dev main.py)
```

Endpoints: `POST /answer` ({question} → {answer}), `GET /healthz`, docs at
`/docs`.

## Layout

```
main.py        # build + serve the program (DataModels, program, FastAPI app)
data_models/   # Synalinks DataModels — structured I/O schemas
programs/      # Synalinks Program definitions (the pipeline you grow)
routes/        # FastAPI routers — thin HTTP endpoints
services/      # business logic / integrations
auth/          # authentication
Dockerfile     # installs deps, builds the artifact, serves
.env.template  # copy to .env for MODEL / endpoints / MLflow URI
```

`main.py` starts as a single self-contained file; each folder above is where you
move things (and why) as the backend grows — see each folder's `README.md`.

## Model & observability

- **Model = vLLM by default** (`MODEL=vllm/Qwen/Qwen3-4B`); the endpoint comes
  from `HOSTED_VLLM_API_BASE` (must include `/v1`). `MODEL` is baked into the
  artifact at build time — rebuild to change it. No GPU? Set
  `MODEL=ollama/mistral:latest` and point `HOSTED_VLLM_API_BASE` at Ollama.
- **MLflow tracing** turns on when `MLFLOW_TRACKING_URI` is set
  (`_enable_observability()` runs at startup, before the program is loaded).

This service is normally run via the app's top-level `docker compose up` (which
also starts vLLM and MLflow); see the app `README.md`.
