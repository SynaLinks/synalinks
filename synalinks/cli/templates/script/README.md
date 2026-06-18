# Synalinks Script

A minimal **[Synalinks](https://github.com/SynaLinks/synalinks)** project — a
single `main.py` with one program you can run and grow from.

## Quickstart

```bash
# 1. Install (this is a uv project)
uv sync

# 2. Point at a model. Default is local Ollama — no API key needed:
ollama serve
ollama pull llama3.2:latest
#    (or `cp .env.template .env`, add a hosted key, and set MODEL in main.py)

# 3. Run it
uv run python main.py
```

## Layout

```
main.py          # a one-module program (Question -> Answer) and how to run it
pyproject.toml   # uv project metadata
.env.template    # copy to .env and fill in a key if you use a hosted model
```

## Next steps

Edit `main.py`: change the data models, swap `Generator` for `ChainOfThought`,
chain more modules, or add tools to build an agent. If you want a benchmarked,
self-improving setup, scaffold the `autotrain` or `autosolve` template instead
(`synalinks init --list`).

## License

Apache 2.0, like Synalinks.
