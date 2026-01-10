# Building a REST API

The optimal approach to developing web-apps or micro-services using Synalinks involves building REST APIs and deploying them. You can deploy these APIs locally to test your system or on a cloud provider of your choice to scale to millions of users.

For this purpose, you will need to use FastAPI, a Python library that makes it easy and straightforward to create REST APIs. If you use the default backend, the DataModel will be compatible with FastAPI as they both use Pydantic.

In this tutorial we are going to make a backend that runs locally to test our system.

## Project structure

Your project structure should look like this:

```shell
demo/
├── backend/
│   ├── app/
│   │   └── main.py
│   ├── programs/
│   │   └── checkpoint.program.json
│   ├── requirements.txt
│   ├── Dockerfile
├── frontend/
│   └── ... (your frontend code)
├── scripts/
│   └── train.py (refer to the code examples to learn how to train programs)
├── docker-compose.yml
├── .env.backend
└── README.md
```

## Your `.env.backend` file

This file contains your API keys and configuration:

```shell title=".env.backend"
OPENAI_API_KEY=your-openai-api-key
# Add other provider keys as needed
```

## Your `requirements.txt` file

Import additionally any necessary dependency:

```txt title="requirements.txt"
fastapi[standard]
uvicorn
python-dotenv
synalinks
```

## Creating your endpoint using FastAPI and Synalinks

Now you can create your endpoint using FastAPI.

```python title="main.py"
import argparse
import logging
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

import synalinks

# Load the environment variables
load_dotenv()

# Enable Synalinks built-in observability (uses MLflow)
synalinks.enable_observability(
    tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    experiment_name=os.environ.get("EXPERIMENT_NAME", "production"),
)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set up FastAPI
app = FastAPI()

# The dictionary mapping the name of your custom modules to their class
custom_modules = {}

# Load your program
program = synalinks.Program.load(
    "programs/checkpoint.program.json",
    custom_modules=custom_modules,
)


@app.post("/v1/chat_completion")
async def chat_completion(messages: synalinks.ChatMessages):
    logger.info(messages.prettify_json())
    try:
        result = await program(messages)
        if result:
            logger.info(result.prettify_json())
            return result.get_json()
        else:
            return None
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
```

## Creating the Dockerfile

Here is the Dockerfile to use according to FastAPI documentation.

```Dockerfile title="Dockerfile"
FROM python:3.13

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./programs /code/programs

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
```

## The docker compose file

And finally your docker compose file.

```yml title="docker-compose.yml"
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow-data:/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root mlflow-artifacts:/
      --serve-artifacts
      --artifacts-destination /mlflow/artifacts
    restart: unless-stopped
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - .env.backend
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - EXPERIMENT_NAME=production
    depends_on:
      - mlflow
    restart: unless-stopped
```

## Launching your backend

Launch your backend using `docker compose`:

```shell
cd demo
docker compose up
```

Open your browser to `http://0.0.0.0:8000/docs` and test your API with the FastAPI UI.

You can view traces and metrics at `http://0.0.0.0:5000` (MLflow UI).
