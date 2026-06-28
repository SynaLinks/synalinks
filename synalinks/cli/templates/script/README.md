# Synalinks Script

A minimal **[Synalinks](https://github.com/SynaLinks/synalinks)** project — a
single `main.py` with one program you can run and grow from.

## Quickstart

```bash
# 1. Install (this is a uv project)
uv sync

# 2. Serve a model with vLLM (the default), on its OpenAI endpoint (:8000):
vllm serve Qwen/Qwen3-4B
#    (no GPU? set MODEL=ollama/mistral:latest in main.py and run `ollama serve`)
#    (hosted model? `cp .env.template .env`, add a key, and set MODEL in main.py)

# 3. (optional) trace runs to MLflow:
#    mlflow server --host 127.0.0.1 --port 5000
#    then set MLFLOW_TRACKING_URI=http://localhost:5000 in .env

# 4. Run it
uv run python main.py
```

## Layout

```
main.py          # a one-module program (Question -> Answer) and how to run it
pyproject.toml   # uv project metadata
.env.template    # copy to .env for the vLLM endpoint, MLflow URI, or hosted keys
```

## Next steps

Edit `main.py`: change the data models, swap `Generator` for `ChainOfThought`,
chain more modules, or add tools to build an agent. If you want a benchmarked,
self-improving setup, scaffold the `autotrain` or `autosolve` template instead
(`synalinks init --list`).

## License

Apache 2.0, like Synalinks.
