# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synalinks is a neuro-symbolic Language Model (LM) framework inspired by Keras. It provides a declarative API for building, training, and deploying LM-based applications including RAGs, autonomous agents, and self-evolving reasoning systems
by composing modules like layers in deep learning.

Its phylosophy: Keras but with JSON data models instead of tensors, JSON schemas instead of tensor shapes. Only focused on gradient free methods with in-context learning via LM update or out-context using knowledge bases as extendible memory systems at runtime via tool calls. A lot of the focus is with evals using reward systems and metrics to evaluate the success of a program the equivalent of a deep learning model.

## Development Commands

```bash
# Install dependencies
./shell/install.sh

# Run tests with coverage
./shell/test.sh

# Run a single test file
uv run pytest synalinks/src/path/to/test_file.py -v

# Run a specific test
uv run pytest synalinks/src/path/to/test_file.py::test_function_name -v

# Lint check
./shell/lint.sh

# Format code
./shell/format.sh

# Build documentation
./shell/doc.sh
```

## Methodology for long running tasks (batch issue fixing, multi-file reviews, etc.)

At the start of every session, read `task_tracker.json` at the repo root if it exists: it records where a previous session stopped (crash, context reset, interruption). Resume from the first task with `"is_finished": false` in the first goal entry.

1. Gather all the information you need first: browse the relevant issues, files and external documentation quickly.

2. Add a goal entry to `task_tracker.json` at the repo root listing every task, in exactly this format (valid JSON, same keys and nesting; do not rename, add or drop fields):

```json
[
  {
    "goal": "one line describing the overall objective",
    "tasks": [
      {
        "key": "task-name-1",
        "title": "one line summary (also the issue/PR title)",
        "labels": ["bug"],
        "description": "full description: what to do, how to verify it",
        "is_finished": false,
        "files": ["synalinks/src/path/to/file.py"],
        "url": "https://github.com/SynaLinks/synalinks/issues/123"
      }
    ]
  }
]
```

   Update the file as soon as a task is done (`"is_finished": true`, fill `url` with the issue/PR link), never in a batch at the end, so a crash loses at most one task. When every task of a goal is finished, remove that goal entry from the tracker.

3. Work on one task at a time. Use tests to verify your assumptions and keep the code minimal (removing unnecessary code is better than adding). Use `/ponytail` all the time.

4. Be as autonomous as possible. If something is undecided or needs human approval, open a GitHub issue tagged as such and move on to the next task.

5. Think about the scope of the framework and the bigger picture when making architectural decisions: the software takes inspiration from ML frameworks like Keras and JAX, repurposed for neuro-symbolic AI. Be serious in your work, no laziness accepted.

`task_tracker.json` is gitignored and must never be committed.

## Code is the memory

Never rely on your own internal memory (auto-memory files, recollections from past sessions): it is often stale. The code is law: it is the memory. Before acting on anything remembered (a function, flag, file, behavior or design decision), verify it against the current code, tests and git history, and trust the code when they disagree. The only persistent state for ongoing work is `task_tracker.json`.

## Architecture

### Core Abstractions

The framework follows a Keras-like pattern with these core abstractions:

- **Module** (`synalinks/src/modules/module.py`): Base class for all composable units, similar to Keras Layers. Modules have:
  - `__init__()`: Define attributes and create variables
  - `build()`: Create state that depends on input shapes
  - `call()`: The forward pass logic (async)
  - `get_config()`/`from_config()`: Serialization support

- **Program** (`synalinks/src/programs/program.py`): Groups modules into trainable/deployable objects (like Keras Models). Inherits from both `Trainer` and `Module`. Supports:
  - Functional API: Chain module calls from `Input` to outputs
  - Subclassing: Override `call()` method
  - Sequential: Stack of single-input/single-output modules

- **DataModel**: Pydantic-based structured data with JSON schema support. All module I/O uses DataModels or its pydantic independent JsonDataModels.

### Key Components

- **Generator** (`synalinks/src/modules/core/generator.py`): Core module for LM inference with structured outputs
- **FunctionCallingAgent** (`synalinks/src/modules/agents/function_calling_agent.py`): Autonomous agent with parallel tool calling
- **ChainOfThought** (`synalinks/src/modules/ttc/chain_of_thought.py`): Generator with thinking field for step-by-step reasoning

### Training System

- **Trainer** (`synalinks/src/trainers/trainer.py`): Provides `compile()` and `fit()` methods
- **Optimizers** (`synalinks/src/optimizers/`): In-context RL optimizers for prompt/example optimization
- **Rewards** (`synalinks/src/rewards/`): Reward functions like `ExactMatch`, `CosineSimilarity`, `LMAsJudge`
- **Metrics** (`synalinks/src/metrics/`): Training metrics tracking like `F1Score`

### Backend

Uses Pydantic as the data backend (`synalinks/src/backend/pydantic/`). The backend provides:
- `Variable`: Trainable state containers
- `SymbolicDataModel`/`JsonDataModel`: Data model representations
- Name scoping and state management

### Serialization

All objects are JSON-serializable via `synalinks.saving`. Custom objects need `@synalinks.saving.register_synalinks_serializable()` decorator.

## Code Conventions

- Use `uvx ruff` for linting with config in `pyproject.toml`
- Use `black` for formatting with 90 char line length
- Tests are colocated with source files using `*_test.py` suffix
- All module `call()` and `build()` methods are async
- OOP for modules, utility functions in the `utils` folder

## API Structure

Public API is exported via `synalinks/api/` directory. The `api_gen.py` script generates `synalinks/__init__.py` from exports decorated with `@synalinks_export()`.
