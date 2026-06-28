# data_models/

The Synalinks `DataModel`s that define your structured I/O live here — the typed
schemas modules read and write, and that FastAPI reuses as request/response
bodies (they're Pydantic models, so the schema is written once and shows up at
`/docs` for free).

## What goes here

- Input/output schemas (e.g. `Question`, `Answer`) and any nested models.
- Field descriptions — these are part of the prompt the LM sees, so write them
  well.

## Sketch

```python
# data_models/qa.py
import synalinks


class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="The question to answer")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer to the question")
```

Import them from `programs/` (to build the pipeline) and `routes/` (as request /
response types). Starting out, the bundled `main.py` defines these inline — move
them here as the project grows.
