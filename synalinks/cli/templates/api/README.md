# Synalinks API

An **OpenAI chat-completions-compatible** endpoint backed by a
**[Synalinks](https://github.com/SynaLinks/synalinks)** `FunctionCallingAgent`,
served with [FastAPI](https://fastapi.tiangolo.com/). Point any OpenAI client at
it — the agent can call tools (here, a calculator) before answering.

## Quickstart (local)

```bash
# 1. Install (this is a uv project)
uv sync

# 2. Serve a model with vLLM (the default). Run it on a non-8000 port so it
#    doesn't clash with this API, and point the app at it:
vllm serve Qwen/Qwen3-4B --port 8001
cp .env.template .env            # then set: HOSTED_VLLM_API_BASE=http://localhost:8001/v1
#    (no GPU? set MODEL=ollama/mistral:latest and run `ollama serve` instead)

# 3. (optional) MLflow tracing — start a server and set the URI:
#    mlflow server --host 127.0.0.1 --port 5000
#    then in .env: MLFLOW_TRACKING_URI=http://localhost:5000

# 4. Build the agent once (offline — no model needed), then serve it
uv run python main.py build
uv run python main.py            # or: fastapi dev main.py
```

Then call it like the OpenAI API:

```bash
curl http://localhost:8000/healthz

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "synalinks-agent",
       "messages": [{"role": "user", "content": "What is 12 * 7?"}]}'
```

…or with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="synalinks-agent",
    messages=[{"role": "user", "content": "What is 12 * 7?"}],
)
print(resp.choices[0].message.content)
```

Interactive docs live at <http://localhost:8000/docs>.

## Quickstart (Docker)

`docker compose up` runs the API together with a local **vLLM** model server and
an **MLflow** tracking server. The agent is baked into the image at build time
(offline). vLLM needs an NVIDIA GPU + the NVIDIA Container Toolkit; the first
start downloads the model, and the API waits for vLLM to be healthy so requests
succeed.

```bash
docker compose up --build
```

Once up:

- **API docs:** <http://localhost:8000/docs>
- **MLflow:** <http://localhost:5000>
- **vLLM:** <http://localhost:8001/v1>

No GPU? Set `MODEL=ollama/mistral:latest` and point `HOSTED_VLLM_API_BASE` at an
Ollama service instead (see `.env.template`).

## Layout

```
main.py             # the agent + the OpenAI-compatible FastAPI app
pyproject.toml      # uv project metadata
Dockerfile          # installs deps, builds the agent (offline), serves
docker-compose.yaml # API + vLLM (model server) + MLflow (tracing)
.env.template       # copy to .env and set MODEL / endpoints / MLflow URI
auth/               # add authentication here (see auth/README.md)
```

## Observability

Tracing is on whenever `MLFLOW_TRACKING_URI` is set (docker-compose sets it for
you). `main.py` calls `synalinks.enable_observability(...)` at startup, before
the agent is loaded, so every module call is traced to MLflow. Leave the var
unset to disable. See the
[Observability guide](https://synalinks.github.io/synalinks/guides/Observability/).

## Notes

- **Streaming** (`stream=true`) is not implemented in this starter; the server
  returns `400` for it. Add it with a `StreamingResponse` when you need it.
- **Build once, serve forever.** `build` constructs and saves the agent; the
  server only loads it. Building doesn't call the LM (it just composes modules
  and records schemas), so it runs offline — no model needed until request time.
- Swap `calculate` for your own tools, or change the agent for an RLM / RAG /
  DeepAgent — the OpenAI-compatible layer stays the same.

For production posture (process manager, timeouts, auth, observability), read
the [deployment guide](https://synalinks.github.io/synalinks/guides/FastAPI%20Deployment/).

## License

Apache 2.0, like Synalinks.
