# Project Templates

`synalinks init` scaffolds a ready-to-run project from a bundled template. The
templates ship inside the wheel, so it works fully offline.

```shell
# Interactive — pick a template from a menu:
uvx synalinks init

# Or non-interactively:
uvx synalinks init my-agent --template api
```

If Synalinks is already installed in your environment, use `synalinks init`
directly (the `uvx` prefix just runs it in a throwaway environment).

## The `init` command

```shell
synalinks init [PROJECT_NAME] [OPTIONS]
```

| Option | Description |
| --- | --- |
| `PROJECT_NAME` | Project (and package) name. Prompted if omitted. |
| `-n, --name` | Same as the positional name; takes precedence. |
| `-t, --template` | Template to scaffold (see below). Prompted if omitted. |
| `-d, --description` | Description written into the generated `pyproject.toml`. |
| `-l, --list` | List the available templates and exit. |
| `-f, --force` | Scaffold into an existing non-empty directory. |

Run `synalinks init --list` to see the templates available in your version.

## The templates

| Template | What you get |
| --- | --- |
| **`script`** | A minimal single-file project — one `Generator` program you run and grow from. The best starting point. |
| **`api`** | An **OpenAI chat-completions-compatible** REST endpoint (`POST /v1/chat/completions`) backed by a `FunctionCallingAgent`, served with FastAPI. Point any OpenAI client at it. |
| **`app`** | A full-stack starter: a Synalinks + FastAPI **backend**, a local **vLLM** model server, and an **MLflow** tracking server wired together with Docker Compose. The `frontend/` is a placeholder for the UI of your choice. |
| **`mcp`** | A Synalinks agent exposed as an **MCP server** (Model Context Protocol) with FastMCP, so clients like Claude Desktop or Cursor can call it as a tool. |
| **`autotrain`** | A program + training harness (`train.py`) you improve by editing levers and letting `program.fit()` + an in-context optimizer learn. |
| **`autosolve`** | A program + evaluation harness (`evaluate.py`) where *your coding agent is the optimizer*: read the failures, rewrite the program, re-measure. No training loop. |

After scaffolding, the printed next steps are:

```shell
cd <project>
uv sync
npx skills add -y SynaLinks/synalinks-skills --skill synalinks
```

Then start your coding agent (Claude Code, Cursor, Copilot, …) in the project
folder — every template ships an `AGENTS.md` (with a `CLAUDE.md` symlink) that
teaches the agent how the project is laid out.

## Shared conventions

Every template follows the same conventions, so moving between them is easy:

- **Model via the `MODEL` setting.** The default is **vLLM**
  (`vllm/Qwen/Qwen3-4B`), an OpenAI-compatible local server. The base URL comes
  from `HOSTED_VLLM_API_BASE` and **must include `/v1`**
  (e.g. `http://localhost:8000/v1`). No GPU? Set `MODEL=ollama/mistral:latest`
  and run `ollama serve`. Any [LiteLLM provider](Synalinks API/Language Models API.md)
  works by changing the string and adding the matching key in `.env`.
- **Build once, serve forever.** Server templates split *building* the program
  (a `build` step that composes modules and `.save()`s a JSON artifact) from
  *serving* it (the server only `.load()`s the artifact). Building does **not**
  call the LM — even for an agent — so it runs fully offline (CI, Docker
  image-build); a model is only needed at request time.
- **Observability is one env var away.** Set `MLFLOW_TRACKING_URI` and every
  module call is traced to [MLflow](guides/Observability.md). Leave it unset and
  tracing is a no-op — no MLflow server required.
- **`.env` for secrets and endpoints.** Each template ships a `.env.template`;
  copy it to `.env` and fill in only what your model needs.
- **Bring your own coding agent.** `AGENTS.md` documents the project for coding
  agents, and the [`synalinks-skills`](https://github.com/SynaLinks/synalinks-skills)
  teach framework conventions.

## Running the server templates

`api` and `app` ship a `Dockerfile` and a `docker-compose.yaml` that run the app
alongside a local vLLM (GPU) and an MLflow server:

```shell
docker compose up --build
```

vLLM needs an NVIDIA GPU + the NVIDIA Container Toolkit. Without one, set
`MODEL=ollama/mistral:latest` and point `HOSTED_VLLM_API_BASE` at an Ollama
service (see each template's `.env.template`). The `mcp` template runs over
stdio (what MCP clients launch) and has no compose file.

Each template's own `README.md` has the exact quickstart, ports, and endpoints.

## Related

- [FastAPI Deployment](guides/FastAPI Deployment.md) — the pattern behind the
  `api` and `app` templates.
- [FastMCP Deployment](guides/FastMCP Deployment.md) — the pattern behind the
  `mcp` template.
- [Training](guides/Training.md) — the `.compile()` / `.fit()` loop behind
  `autotrain`.
- [Observability](guides/Observability.md) — the MLflow tracing every template
  wires in.
