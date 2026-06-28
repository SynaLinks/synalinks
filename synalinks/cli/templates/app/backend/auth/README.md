# auth/

Authentication and authorization for the backend live here. It's empty by
design — Synalinks doesn't opine on auth, and FastAPI already ships the building
blocks.

## What goes here

- API-key / bearer-token verification, OAuth2 / OIDC flows, JWT decoding.
- A FastAPI **dependency** that validates the caller and is attached to
  protected routes (e.g. `/answer`).

## Sketch

```python
# auth/api_key.py
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="Authorization", auto_error=False)


async def require_api_key(value: str = Security(_header)) -> None:
    # Compare against an env/secret store — never hard-code keys.
    if value != f"Bearer {EXPECTED_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

```python
# main.py
from auth.api_key import require_api_key


@app.post("/answer", dependencies=[Depends(require_api_key)])
async def answer(...):
    ...
```

See FastAPI's [security guide](https://fastapi.tiangolo.com/tutorial/security/)
for OAuth2, scopes, and JWTs. Keep secrets in the environment (`.env`) for local dev, and in secret managers in prod.
Never in the source.
