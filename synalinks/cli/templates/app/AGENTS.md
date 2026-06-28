# Agent guide — app

A full-stack starter: a Synalinks + FastAPI **backend** (`backend/main.py`), a
local **vLLM** model server, and an **MLflow** tracking server, orchestrated by
`docker-compose.yaml`. The **frontend is a placeholder** (`frontend/` holds only
a README) — the user brings their own.

## What to know

- **Backend** = a Synalinks `Generator` program (question → answer) served with
  FastAPI. Build-once / serve-forever: `build_and_save_program` writes
  `program.json`; the server lifespan only *loads* it. Endpoints: `POST /answer`,
  `GET /healthz`, docs at `/docs`. (Details in `backend/README.md`.)
- **Frontend** = not implemented. `frontend/README.md` documents the backend's
  wire format and two ways to wire a UI in (nginx reverse-proxy, or direct +
  CORS). If asked to build one, add it under `frontend/` and add a matching
  service to `docker-compose.yaml`.
- **Models = vLLM by default** (`MODEL=vllm/Qwen/Qwen3-4B`); the base URL comes
  from `HOSTED_VLLM_API_BASE`. `MODEL` is baked into the backend artifact at
  build time — set it in `backend/.env` or as the compose build arg, then
  rebuild. No GPU? Use `MODEL=ollama/mistral:latest`.
- **MLflow tracing** is enabled at startup whenever `MLFLOW_TRACKING_URI` is set
  (`_enable_observability()` runs before the program is loaded).
- **Compose wiring.** `backend` (8000) → `vllm` (8001→8000) + `mlflow` (5000).

## Commands

```bash
docker compose up --build      # backend :8000 (docs /docs), vLLM :8001, MLflow :5000

# Backend alone (run vLLM yourself on a non-8000 port first):
cd backend && uv sync && uv run python main.py build && uv run python main.py
```

For an OpenAI-compatible chat endpoint backed by an agent, see the `api`
template. Custom Agent Skills can live under `.agents/skills/`.

## Troubleshooting a framework bug

Most failures are in *your* program — fix those here. But if you trace a problem
to **Synalinks itself** (a stack trace inside the `synalinks` package, or a
missing/broken framework feature), fix it at the source and upstream it:

1. **Clone the framework** into `backend/` (the `synalinks/` dir is git-ignored):

   ```bash
   cd backend
   git clone https://github.com/SynaLinks/synalinks.git
   git -C synalinks checkout -b fix/<short-description>
   ```

2. **Point the backend at the local checkout** so your runs exercise the fix:

   ```bash
   uv add --editable ./synalinks
   ```

3. **Fix the bug** under `synalinks/synalinks/src/...`, following that repo's
   `CLAUDE.md`. Add or update a colocated `*_test.py` covering the bug.

4. **Verify**: run the framework's tests (`cd synalinks && ./shell/test.sh`),
   then re-run the backend to confirm the failure is gone.

5. **Open a PR** from the checkout, then restore the released dependency:

   ```bash
   git -C synalinks commit -am "fix: <what you fixed>"
   git -C synalinks push -u origin HEAD
   (cd synalinks && gh pr create --fill)
   uv remove synalinks && uv add synalinks   # drop the local override
   ```
