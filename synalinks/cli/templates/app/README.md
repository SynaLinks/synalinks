# Synalinks App

A full-stack starter built around a **[Synalinks](https://github.com/SynaLinks/synalinks)**
+ FastAPI **backend**, a local **vLLM** model server, and an **MLflow** tracking
server, wired together with Docker Compose. The **frontend is left to you** —
`frontend/` is a placeholder with guidance for plugging in whatever you like
(see `frontend/README.md`).

## Quickstart (Docker)

`docker compose up` runs the backend, vLLM, and MLflow. vLLM needs an NVIDIA GPU
+ the NVIDIA Container Toolkit; the first start downloads the model.

```bash
docker compose up --build
```

Then:

- **Backend docs:** <http://localhost:8000/docs>  (try `POST /answer`)
- **MLflow:** <http://localhost:5000>
- **vLLM:** <http://localhost:8001/v1>

No GPU? Set `MODEL=ollama/mistral:latest` and point `HOSTED_VLLM_API_BASE` at an
Ollama service instead (see `backend/.env.template`).

## Layout

```
backend/            # Synalinks + FastAPI service (question -> answer)
  main.py           #   build + serve the program
  Dockerfile
  auth/             #   add authentication here (see auth/README.md)
  services/         #   business logic / integrations (see services/README.md)
frontend/           # placeholder — bring your own frontend (see its README)
docker-compose.yaml # vLLM + MLflow + backend
```

## Observability

Tracing is on whenever `MLFLOW_TRACKING_URI` is set (docker-compose sets it for
you). The backend calls `synalinks.enable_observability(...)` at startup, before
the program is loaded, so module calls are traced to MLflow. See the
[Observability guide](https://synalinks.github.io/synalinks/guides/Observability/).

## Developing the backend alone

```bash
cd backend
uv sync
uv run python main.py build
uv run python main.py            # http://localhost:8000  (or: fastapi dev main.py)
```

## Next steps

- **Build a frontend** in `frontend/` and add a service for it to
  `docker-compose.yaml` (see `frontend/README.md`).
- Swap the backend's one-module `Generator` for a `ChainOfThought`, an agent, or
  a RAG pipeline — the wire format (`POST /answer`) stays the same.
- For an OpenAI-compatible chat endpoint backed by an agent, scaffold the `api`
  template instead (`synalinks init --list`).

## License

Apache 2.0, like Synalinks.
