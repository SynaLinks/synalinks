# programs/

The Synalinks `Program` definitions live here — the functions that compose
modules (`Generator`, `ChainOfThought`, agents, RAG pipelines) into the pipeline
your app serves. This is the part you'll grow and optimize.

## What goes here

- `build_*` functions that wire `Input` → modules → `Program`.
- The choice of modules and how they're composed (the architecture).

## Sketch

```python
# programs/qa.py
import synalinks

from data_models.qa import Answer, Question


async def build_qa(lm) -> synalinks.Program:
    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
    return synalinks.Program(
        inputs=inputs, outputs=outputs, name="qa", description="Answers a question."
    )
```

Keep build-once / serve-forever in mind: a `build` step constructs and `.save()`s
the program, and the server only `.load()`s the artifact. Building does **not**
call the LM (not even for agents) — it just composes modules and records their
schemas — so you can build offline, e.g. in CI or at Docker image-build time.
The bundled `main.py` builds inline; move it here as the project grows.
