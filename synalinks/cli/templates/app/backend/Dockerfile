# Self-contained image for the app backend (Synalinks + FastAPI).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY . .

# Install dependencies into the image's virtual environment.
RUN uv sync --no-dev

# MODEL is baked into the program artifact, so it must be set at build time.
ARG MODEL=vllm/Qwen/Qwen3-4B
ENV MODEL=$MODEL

# Build the program artifact at image-build time ("build once"): a plain
# Generator program builds offline, so no model/credentials are needed here.
RUN uv run python main.py build

EXPOSE 8000

# Serve the prepared artifact.
CMD ["uv", "run", "python", "main.py"]
