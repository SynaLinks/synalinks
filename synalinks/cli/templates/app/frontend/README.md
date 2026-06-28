# Frontend (bring your own)

This template ships the **backend only** — pick whatever frontend you like and
drop it in here (React, Vue, Svelte, SvelteKit, plain HTML, …).

## Talking to the backend

The backend (see `../backend`) exposes:

- `POST /answer` — body `{"question": "..."}`, returns `{"answer": "..."}`
- `GET /healthz` — liveness probe
- `GET /docs` — interactive OpenAPI docs

Example call from the browser:

```js
const res = await fetch("/api/answer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "What is the capital of France?" }),
});
const { answer } = await res.json();
```

## Wiring it up

Two common options:

1. **Reverse-proxy (recommended)** — serve the frontend and proxy `/api/` to the
   backend so the browser only ever talks to one origin (no CORS). For a static
   build, an `nginx:alpine` image with a `location /api/ { proxy_pass
   http://backend:8000/; }` block does it.
2. **Direct + CORS** — call `http://localhost:8000` directly and enable CORS on
   the backend with FastAPI's `CORSMiddleware`.

Then add a `frontend` service to the top-level `docker-compose.yaml` (with
`depends_on: [backend]`) and publish its port.
