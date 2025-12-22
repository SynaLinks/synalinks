# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synalinks is a neuro-symbolic Language Model (LM) framework inspired by Keras. It provides a declarative API for building, training, and deploying LM-based applications including RAGs, autonomous agents, and self-evolving reasoning systems.

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

- **DataModel**: Pydantic-based structured data with JSON schema support. All module I/O uses DataModels.

### Key Components

- **Generator** (`synalinks/src/modules/core/generator.py`): Core module for LM inference with structured outputs
- **FunctionCallingAgent** (`synalinks/src/modules/agents/function_calling_agent.py`): Autonomous agent with parallel tool calling
- **ChainOfThought** (`synalinks/src/modules/ttc/chain_of_thought.py`): Generator with thinking field for step-by-step reasoning

### Training System

- **Trainer** (`synalinks/src/trainers/trainer.py`): Provides `compile()` and `fit()` methods
- **Optimizers** (`synalinks/src/optimizers/`): In-context RL optimizers for prompt/example optimization
- **Rewards** (`synalinks/src/rewards/`): Reward functions like `ExactMatch`, `CosineSimilarity`
- **Metrics** (`synalinks/src/metrics/`): Training metrics tracking

### Backend

Uses Pydantic as the data backend (`synalinks/src/backend/pydantic/`). The backend provides:
- `Variable`: Trainable state containers
- `SymbolicDataModel`/`JsonDataModel`: Data model representations
- Name scoping and state management

### Serialization

All objects are JSON-serializable via `synalinks.saving`. Custom objects need `@synalinks.saving.register_synalinks_serializable()` decorator.

## Code Conventions

- Use `ruff` for linting with config in `pyproject.toml`
- Use `black` for formatting with 90 char line length
- Tests are colocated with source files using `*_test.py` suffix
- All module `call()` methods are async
- Graph databases (Neo4J/MemGraph) required for some tests

## API Structure

Public API is exported via `synalinks/api/` directory. The `api_gen.py` script generates `synalinks/__init__.py` from exports decorated with `@synalinks_export()`.
