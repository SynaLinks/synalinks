# services/

Business logic and integrations live here, separate from the HTTP layer. Keep
`main.py`'s route handlers thin: parse the request, call a service, shape the
response. Everything else — databases, external APIs, and orchestrating the
Synalinks program — belongs in a service module.

## What goes here

- Building / loading and running Synalinks programs (the `Generator`, agents,
  RAG pipelines).
- Knowledge bases and retrievers, caches, third-party API clients.
- Pure domain logic you want to unit-test without spinning up the server.

## Sketch

```python
# services/qa.py
from pathlib import Path

import synalinks

_PROGRAM: synalinks.Program | None = None


def load_program(path: Path) -> None:
    global _PROGRAM
    _PROGRAM = synalinks.Program.load(str(path))


async def answer_question(question) -> dict:
    result = await _PROGRAM(question)
    return result.get_json() if result is not None else {}
```

```python
# main.py
from services import qa

# load_program(...) in the lifespan; call qa.answer_question(...) in the handler.
```

Keeping services framework-agnostic (no FastAPI imports here) makes them easy to
test and reuse — e.g. from a CLI, a worker, or a notebook.
