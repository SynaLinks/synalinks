# routes/

The FastAPI **routers** live here — the HTTP endpoints, grouped into
`APIRouter`s and included from `main.py`. Keep handlers thin: parse the request,
call a `services/` function, return a `data_models/` type.

## What goes here

- One `APIRouter` per resource/feature (e.g. `qa.py`).
- Request/response typing via your `data_models/`; auth via `auth/`
  dependencies; the actual work delegated to `services/`.

## Sketch

```python
# routes/qa.py
from fastapi import APIRouter

from data_models.qa import Answer, Question
from services import qa

router = APIRouter()


@router.post("/answer")
async def answer(question: Question) -> Answer:
    return Answer.model_validate(await qa.answer_question(question))
```

```python
# main.py
from routes.qa import router as qa_router

app.include_router(qa_router)
```

The bundled `main.py` defines the `/answer` route inline — split routes out here
as the API grows.
