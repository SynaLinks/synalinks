# Synalinks MCP server

A **[Synalinks](https://github.com/SynaLinks/synalinks)** agent exposed as an
**MCP** (Model Context Protocol) server with [FastMCP](https://gofastmcp.com/),
so language-model clients (Claude Desktop, Cursor, …) can call it as a tool. It
follows the build-once / serve-forever pattern from the
[FastMCP Deployment guide](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/).

## Quickstart

```bash
# 1. Install (this is a uv project)
uv sync

# 2. Serve a model with vLLM (the default), on its OpenAI endpoint (:8000):
vllm serve Qwen/Qwen3-4B
#    (no GPU? set MODEL=ollama/mistral:latest in main.py and run `ollama serve`)

# 3. (optional) trace tool calls to MLflow:
#    mlflow server --host 127.0.0.1 --port 5000
#    then set MLFLOW_TRACKING_URI=http://localhost:5000 in .env

# 4. Build the agent once, then serve it over stdio
uv run python main.py build
uv run python main.py
```

## Connecting an MCP client

MCP clients launch the server themselves over **stdio**. Add an entry like this
to the client's MCP config (the exact file/format depends on the client):

```json
{
  "mcpServers": {
    "synalinks-agent": {
      "command": "uv",
      "args": ["run", "python", "main.py"],
      "cwd": "/absolute/path/to/this/project"
    }
  }
}
```

The client then sees one tool, `solve`, that answers arithmetic word problems.

### Remote (HTTP) instead of stdio

For a client on another machine, serve over HTTP — in `main.py` swap
`mcp.run()` for:

```python
mcp.run(transport="http", host="127.0.0.1", port=8001)
```

## Layout

```
main.py             # the agent, the MCP tool (`solve`), and the server
pyproject.toml      # uv project metadata
.env.template       # copy to .env and set MODEL / endpoints / MLflow URI
```

## Notes

- **Build at a separate step.** `build` constructs and saves the agent; the
  server only *loads* it. Building doesn't call the LM (it just composes modules
  and records schemas), so it runs offline — no model needed until a tool call.
- **Tool errors are part of the protocol** — `solve` raises (it does not swallow
  exceptions) so the calling LM gets a structured error and can react. In
  production, consider `FastMCP(..., mask_error_details=True)`.
- Swap `calculate` for your own tools, or replace the `FunctionCallingAgent`
  with any Synalinks program — the MCP layer stays the same.

For production posture and transports beyond stdio, read the
[deployment guide](https://synalinks.github.io/synalinks/guides/FastMCP%20Deployment/).

## License

Apache 2.0, like Synalinks.
