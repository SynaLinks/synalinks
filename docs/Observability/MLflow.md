# Observability with MLflow

Synalinks provides built-in observability through MLflow, enabling you to trace and monitor your LM programs in production.

## Overview

The observability system automatically creates spans for each module call, capturing:

- **Inputs and outputs** of each module
- **Duration** of each call
- **Cost** information (when available from the language model)
- **Success/failure status**
- **Parent-child relationships** between nested module calls

## Quick Start

### Enable Observability

> **Important**: You must call `enable_observability()` **BEFORE** creating any modules.
> Hooks are registered when modules are instantiated, so enabling observability after
> module creation will not trace those modules.

```python
import synalinks

# Enable FIRST, before creating any modules
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",
    experiment_name="my_experiment"
)

# Now create your modules - they will be automatically traced
inputs = synalinks.Input(data_model=Question)
outputs = await synalinks.Generator(...)(inputs)
```

Once enabled, all module calls in your program will be automatically traced.

### Example Usage

```python
import asyncio
import synalinks

# Enable observability before creating your program
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",
    experiment_name="question_answering"
)


class Answer(synalinks.DataModel):
    answer: str


async def main():
    # Create a simple question-answering program
    inputs = synalinks.Input(data_model=synalinks.String)
    outputs = synalinks.Generator(
        data_model=Answer,
        description="Answer the user's question",
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
        description="A simple QA program",
    )

    # Run the program - traces will be sent to MLflow
    result = await program(synalinks.String(text="What is the capital of France?"))
    if result:
      print(result.prettify_json())

if __name__ == "__main__":
    asyncio.run(main())
```

## Running MLflow with Docker

### Using Docker

Run MLflow tracking server locally with artifact proxying enabled:

```bash
docker run -d \
    --name mlflow \
    -p 5000:5000 \
    -v mlflow-data:/mlflow \
    ghcr.io/mlflow/mlflow:latest \
    mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root mlflow-artifacts:/ \
    --serve-artifacts \
    --artifacts-destination /mlflow/artifacts
```

**Important flags:**

- `--serve-artifacts`: Enables the MLflow server to proxy artifact uploads from clients
- `--default-artifact-root mlflow-artifacts:/`: Tells clients to use the server as an artifact proxy
- `--artifacts-destination /mlflow/artifacts`: Where the server stores artifacts on disk

Then configure Synalinks to use it:

```python
import synalinks

synalinks.enable_observability(
    tracking_uri="http://localhost:5000",
    experiment_name="synalinks_traces"
)
```

### Using Docker Compose

For a more complete setup with persistent storage, create a `docker-compose.yml`:

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow
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
```

Start the services:

```bash
docker compose up -d
```

Access the MLflow UI at [http://localhost:5000](http://localhost:5000).

### Production Setup with PostgreSQL

For production deployments, use PostgreSQL as the backend store:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: mlflow-postgres
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "5000:5000"
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --default-artifact-root mlflow-artifacts:/
      --serve-artifacts
      --artifacts-destination /mlflow/artifacts
    restart: unless-stopped

volumes:
  postgres-data:
  mlflow-artifacts:
```

## Understanding Traces

When you run a Synalinks program with observability enabled, MLflow captures detailed traces.

### Span Types

Synalinks automatically categorizes spans based on module type for better visualization in MLflow:

| Module | Span Type |
|--------|-----------|
| `Generator`, `ChainOfThought`, `SelfCritique` | `LLM` |
| `FunctionCallingAgent` | `AGENT` |
| `EmbedKnowledge`, `RetrieveKnowledge`, `UpdateKnowledge` | `RETRIEVER` |
| `Tool` | `TOOL` |
| Other modules | `CHAIN` |

### Span Attributes

Each span includes these attributes:

| Attribute | Description |
|-----------|-------------|
| `synalinks.call_id` | Unique identifier for this call |
| `synalinks.parent_call_id` | ID of the parent call (for nested modules) |
| `synalinks.module` | Module class name (e.g., `Generator`) |
| `synalinks.module_name` | Custom name given to the module |
| `synalinks.module_description` | Module description |
| `synalinks.is_symbolic` | Whether the call was symbolic (graph building) |
| `synalinks.duration` | Call duration in seconds |
| `synalinks.success` | Whether the call succeeded |
| `synalinks.cost` | LLM API cost (when available) |

### Exception Events

When a module call fails, the span automatically records an exception event with:
- `exception.type`: The exception class name
- `exception.message`: The exception message

### Viewing Traces

1. Open MLflow UI at `http://localhost:5000`
2. Navigate to your experiment
3. Click on a run to see detailed traces
4. Use the trace view to explore the call hierarchy

## Configuration Options

### Environment Variables

You can also configure MLflow using environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Then in your code:

```python
import synalinks

# Will use MLFLOW_TRACKING_URI from environment
synalinks.enable_observability(experiment_name="my_experiment")
```

### Direct Monitor Hook

For fine-grained control, you can create a Monitor hook directly:

```python
import synalinks

monitor = synalinks.hooks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="custom_experiment"
)

# Add to a specific module
generator = synalinks.Generator(
    data_model=Answer,
    hooks=[monitor]
)
```

## Training Metrics and Artifacts

The `Monitor` callback logs training metrics and program artifacts to MLflow during `fit()`.

### Basic Usage

```python
import synalinks

# Create the monitor callback
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_experiment",
    run_name="my_training_run",
    log_program_plot=True,  # Save program visualization as artifact
)

# Use during training
program.fit(
    x=train_inputs,
    y=train_labels,
    epochs=10,
    callbacks=[monitor]
)
```

### Program Plot Artifact

When `log_program_plot=True` (the default), the Monitor callback automatically saves
a visualization of your program architecture as an MLflow artifact at the start of training.

The plot is saved under `program_plots/` in the artifacts folder and includes:

- Module names and types
- Input/output schemas
- Trainable status of each module

You can view the program plot in the MLflow UI under the "Artifacts" tab of your run.

### Program Model Artifact

When `log_program_model=True` (the default), the Monitor callback saves the program's
trainable state at the end of training. This includes:

- **`model/state_tree.json`**: Contains all trainable variables (few-shot examples, optimized prompts, etc.)
- **`model/model_info.json`**: Metadata about the program (name, description, number of trainable variables)

This is useful for:

- Checkpointing learned parameters during optimization
- Comparing different training runs
- Restoring program state for inference

### Callback Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment_name` | Program name | MLflow experiment name |
| `run_name` | Auto-generated | MLflow run name |
| `tracking_uri` | Local `./mlruns` | MLflow tracking server URI |
| `log_batch_metrics` | `False` | Log metrics at batch level |
| `log_epoch_metrics` | `True` | Log metrics at epoch level |
| `log_program_plot` | `True` | Save program visualization as artifact |
| `log_program_model` | `True` | Save program trainable state as artifact |
| `tags` | `{}` | Additional tags for the run |

### Example with Full Configuration

```python
import synalinks

monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="gsm8k_optimization",
    run_name="chain_of_thought_v1",
    log_batch_metrics=True,
    log_epoch_metrics=True,
    log_program_plot=True,
    tags={
        "model": "gpt-4o-mini",
        "optimizer": "RandomFewShot",
        "dataset": "gsm8k"
    }
)

program.fit(
    x=train_questions,
    y=train_answers,
    epochs=5,
    callbacks=[monitor]
)
```

## Combining Tracing with Training

When using both `enable_observability()` and the `Monitor` callback for training, traces
are created in different experiments depending on the context:

1. **During program building** (symbolic calls): Traces go to the experiment specified
   in `enable_observability()`

2. **During training** (`fit()`): Traces are associated with the training run and go to
   the experiment specified in the `Monitor` callback

### Full Example

```python
import synalinks

# Enable tracing for all module calls
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",
    experiment_name="synalinks_traces"  # Traces during setup go here
)

# Create your program (symbolic traces created here)
inputs = synalinks.Input(data_model=Question)
outputs = await synalinks.Generator(
    data_model=Answer,
    language_model=language_model,
)(inputs)

program = synalinks.Program(inputs=inputs, outputs=outputs, name="my_program")

# Create Monitor callback for training
monitor = synalinks.callbacks.Monitor(
    tracking_uri="http://localhost:5000",
    experiment_name="training_runs",  # Training metrics + traces go here
    run_name="experiment_v1",
)

# Train - traces during fit() are associated with the training run
program.compile(reward=reward, optimizer=optimizer)
await program.fit(x=train_x, y=train_y, epochs=5, callbacks=[monitor])
```

After training, you'll have:
- **synalinks_traces experiment**: Setup traces (symbolic module calls)
- **training_runs experiment**: Training run with metrics, artifacts, and execution traces

## Best Practices

1. **Enable observability early** in your script, before creating any modules
2. **Use meaningful experiment names** to organize your traces by project or feature
3. **Use persistent storage** (PostgreSQL) for production deployments
4. **Set up retention policies** to manage storage for long-running applications

## Troubleshooting

### No traces being created

If you don't see any traces in MLflow:

1. **Check call order**: Ensure `enable_observability()` is called **before** creating any modules
   ```python
   # Wrong - modules created before enabling observability
   inputs = synalinks.Input(data_model=Question)
   synalinks.enable_observability()  # Too late!

   # Correct - enable first
   synalinks.enable_observability()
   inputs = synalinks.Input(data_model=Question)  # Now traces will be created
   ```

2. **Verify observability is enabled**: Check with `synalinks.is_observability_enabled()`

3. **Check the correct experiment**: During training, traces go to the training experiment,
   not the observability experiment

### MLflow not receiving traces

1. Verify the MLflow server is running: `curl http://localhost:5000/health`
2. Check the tracking URI is correct
3. Ensure `mlflow` package is installed: `pip install mlflow`

### Artifacts not showing in MLflow UI

If artifacts are uploaded but don't appear in the MLflow UI:

1. **Check server configuration**: Ensure the MLflow server is started with `--serve-artifacts` flag
2. **Verify artifact root**: The server must use `--default-artifact-root mlflow-artifacts:/` for remote clients
3. **Check permissions**: The server needs write access to `--artifacts-destination` path

**Correct server configuration:**
```bash
mlflow server \
    --serve-artifacts \
    --default-artifact-root mlflow-artifacts:/ \
    --artifacts-destination /mlflow/artifacts
```

**Common mistake** - Missing `--serve-artifacts` causes clients to try writing directly to the server's local filesystem, resulting in permission errors like:
```
PermissionError: [Errno 13] Permission denied: '/mlflow'
```

### Missing cost information

Cost tracking requires the language model to return usage information. Ensure your LLM provider supports this feature.
