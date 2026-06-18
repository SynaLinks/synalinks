"""
# Observability

Imagine your program is a black box. A question goes in, an answer
comes out. When the answer is wrong, where do you look? Up to now we
have built that black box. In this guide we install windows in it,
so you can see what happened inside *without* taking it apart.

The technical word for "ability to see inside the system" is
**observability**. The term comes from control theory, where a
system is called *observable* when its full internal state can be
reconstructed from a finite sequence of outputs. In software we are
less ambitious: we just want enough emitted data to be able to tell
what each component did, in what order, with what arguments, and to
what effect.

This matters more for LM programs than for ordinary code. A single
forward pass through a Synalinks program involves a *non-deterministic*
call to a remote model, parsing the model's text back into a
structured object, possibly invoking tools, possibly branching on
what the model returned. When the final answer is wrong, the bug
could live in any of those steps. A `print` at the end tells you the
program failed — but it does not tell you which link broke.

## Two Tools: Logging and Tracing

There are two related but distinct ideas. Keep them separate in your
head.

A **log record** is what `print()` would be if it were better
behaved: a time-stamped line of text emitted from one point in the
code. Logs are *flat* — no parent/child relationships, no nesting —
easy to grep through, and familiar to anyone who has ever debugged
with print statements.

A **trace** is a recording of a call: it records who called whom,
with what arguments, and how long each call took, like a flight
recorder for your program. More concretely, a trace is a *tree* of
**spans**. A span is a single unit of work and records its start
time, end time, inputs, outputs, status, and which span (if any) is
its parent. Linking each child span to its parent gives a rooted
tree: the **trace**.

Logs answer "what happened?". Traces answer "what happened, in what
order, under which caller, and how long did each step take?". A
non-trivial program benefits from both.

```mermaid
graph TD
    Root["Span: Program (traced_qa)"] --> S1["Span: Generator"]
    S1 --> S2["Span: LanguageModel"]
    S2 --> M["LLM HTTP call"]
```

Each box is a span; each arrow is a "this span called that one" edge.
Because the root span only ends after all of its children have ended, its
duration is at least as long as any descendant's. This nested-call shape is
exactly what Synalinks records.

## What Synalinks Emits

Synalinks provides two cooperating mechanisms. You can use either,
or both.

1. **`synalinks.enable_logging(log_level=...)`** installs a logger.
   Every time a `Module` is called, the logger prints a small block
   of text (a "stanza") containing:

   - a fresh `call_id` (a unique identifier for *this* call),
   - the `parent_call_id` (the id of the caller, or `None` at the
     top),
   - the module's class, name, and description,
   - the input or output data, serialized as JSON.

   This is **structured logging**: each stanza is readable by a
   human, but the `call_id` / `parent_call_id` pair also lets you
   reconstruct the full span tree afterwards by linking children to
   their parents.

2. **`synalinks.enable_observability(tracking_uri=...,
   experiment_name=...)`** sends the same span data to an **MLflow**
   server. MLflow is a popular open-source service for storing ML
   experiments, metrics, and traces. It stores, indexes, and shows
   the trace as an interactive tree in a web UI, and it also
   captures training metrics and saved artifacts.

The two mechanisms are independent. Logging needs no server; MLflow
gives you a nicer interface but needs one running. The runnable
example below uses logging only, so you can run it offline.

## Running with MLflow

The runnable section below leaves `enable_observability` commented out so
the guide runs without any external services. To send spans to an MLflow
UI, start one in another terminal and uncomment the call:

```bash
mlflow ui --port 5000
```

## The Anatomy of One Trace

The logger emits stanzas in the order the calls happen. Read them as
entries in a span tree: a stanza with `Parent call ID: None` is the root;
every other stanza names its parent's id.

For a single question through this guide's program, the order is:

1. Root span opens: the `Functional` program receives the input `Query`.
2. Child span opens: the `Generator` module receives the same `Query`.
3. Grandchild span opens: the `LanguageModel` module receives a
   `ChatMessages` payload (a system prompt describing the required JSON
   keys, plus a user message containing the input).
4. Child span closes: the `Generator` returns the parsed `Answer`.
5. Root span closes: the program returns that same `Answer`.

In the current implementation the `LanguageModel` span's close line and the
`Generator` span's close line appear interleaved in the output, but the
`call_id` / `parent_call_id` fields make the actual tree unambiguous.

```mermaid
sequenceDiagram
    participant U as User code
    participant P as Program span
    participant G as Generator span
    participant L as LanguageModel span
    U->>P: program(Query(...))
    P->>G: forward Query
    G->>L: ChatMessages(system, user)
    L-->>G: parsed Answer
    G-->>P: Answer
    P-->>U: Answer
```

## Inspection Without Running: `program.summary()`

Before you watch a program run, it is useful to see what it is made of.
`program.summary()` prints a table of the modules in the program, the shape
of each module's output, and how many trainable variables each module owns.
Below is what the runnable example prints for our two-module program (an
`InputModule` followed by a `Generator`):

```
Program: traced_qa
description: 'A `Functional` program is a `Program` defined as a directed graph
of modules.'
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Module (type)                   ┃ Output Schema          ┃    Variable # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_module (InputModule)      │ Query (object)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ generator (Generator)           │ Answer (object)        │             1 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

(The `Output Schema` cell normally shows the full JSON Schema; we have
abbreviated it here for readability.)

`Variable #` counts **trainable in-context variables** — JSON
objects obeying a `Trainable` schema that the optimizers in
`synalinks.optimizers` can rewrite during training, much like
weights in a neural network. The most common cases hold a system
instruction or a list of few-shot examples, but in general the
JSON object can carry any structured state its schema permits
([Guide 12](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)).

## Inspection Without Running: `plot_program()`

`synalinks.utils.plot_program(program, ...)` writes a PNG image of the
module graph to disk. ("DAG" stands for directed acyclic graph: arrows
have a direction and never form a cycle, which is the shape of a normal
feed-forward program.) This is useful for documenting non-trivial programs
and for double-checking that the functional API wired up the modules the
way you intended.

## What Goes Wrong Without Observability?

It is worth naming the kinds of bugs that tracing directly helps
with — so you recognize them in the wild and know to reach for the
right tool:

- **Silent prompt drift.** A refactor accidentally changes a system
  prompt; offline tests still pass; the production model's answers
  quietly get worse. Without per-call prompt logging, you only learn
  about the regression when users complain.
- **Schema rejections.** The model returns malformed JSON; the
  parser raises; you see the parser's error but not the raw model
  output that caused it. Traces capture both sides.
- **Latency budget violations.** Your pipeline of *N* modules feels
  slow. Without per-span timings you cannot tell which module is the
  slow one without adding hand-rolled instrumentation.
- **Token accounting.** Hosted LM providers charge by tokens in and
  tokens out per call. You cannot total the cost without a record of
  each call.
- **Branch attribution.** A program with `if`-style control flow
  took the wrong branch on some input. Tracing records *which*
  branch fired and on what predicate.

Once spans are persisted, every one of these moves from "open-ended
detective work" to "a query against a database."

## Logging Levels

`enable_logging(log_level=...)` accepts the standard Python logging levels
(`"debug"`, `"info"`, `"warning"`, `"error"`). Synalinks currently treats
`"debug"` and `"info"` the same way: emit every span. `"warning"` and
`"error"` emit nothing during normal operation. A reasonable rule of thumb:
use `"debug"` while developing the program, and switch to `"warning"` in
production once MLflow (or another sink) is wired up to receive the spans
instead.

## A Complete Worked Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")

async def main():
    load_dotenv()
    synalinks.clear_session()

    # Enable per-module logging (always safe).
    synalinks.enable_logging(log_level="info")

    # Optional: forward spans to an MLflow tracking server.
    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="observability_demo",
    # )

    lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="traced_qa",
    )

    program.summary()
    result = await program(Query(query="What is Python used for?"))
    print(result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Take-Home Summary

- **Observability** means being able to reconstruct what a program
  did from the data it emitted while running. For LM programs this
  takes the form of **span trees**: nested records of every module
  call and every call out to a provider.
- **Synalinks gives you two ways to capture spans.**
  `enable_logging` prints structured per-call text to stdout.
  `enable_observability` forwards spans to an MLflow server for an
  interactive UI.
- **Static** inspection (`program.summary`, `plot_program`)
  describes the shape of the program *before* running it.
  **Runtime** tracing records what *actually* happened during
  execution. The two complement each other.
- **Tracing** turns a class of hard debugging problems — prompt
  drift, schema errors, latency, cost, branch attribution — from
  one-off detective work into routine database queries.

## API References

- [enable_observability](https://synalinks.github.io/synalinks/Synalinks%20API/Observability%20API/)
- [enable_logging](https://synalinks.github.io/synalinks/Synalinks%20API/Observability%20API/)
- [plot_program](https://synalinks.github.io/synalinks/Synalinks%20API/Utils%20API/)
- [Program.summary](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User question."""

    query: str = synalinks.Field(description="User question")


class Answer(synalinks.DataModel):
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # -------------------------------------------------------------------------
    # Enable Observability
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Enable Observability")
    print("=" * 60)

    # Optional: forward spans to an MLflow tracking server.
    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_10_observability",
    # )

    print("\nMLflow tracing is left disabled in this guide; uncomment the")
    print("enable_observability(...) call above to forward spans to MLflow.")
    print("Per-module logging (below) prints span data to stdout regardless.")

    # -------------------------------------------------------------------------
    # Enable Logging
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Enable Logging")
    print("=" * 60)

    synalinks.enable_logging(log_level="info")
    print("\nLogging enabled at INFO level")

    # -------------------------------------------------------------------------
    # Create and Run Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Create and Run Program (spans will be logged)")
    print("=" * 60)

    lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="traced_qa",
    )

    result = await program(Query(query="What is Python used for?"))

    print(f"\nAnswer: {result['answer'][:100]}...")

    # -------------------------------------------------------------------------
    # Program Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Program Summary")
    print("=" * 60)

    program.summary()

    # -------------------------------------------------------------------------
    # Inspect Trainable Variables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Inspect Trainable Variables")
    print("=" * 60)

    print("\nTrainable variables:")
    for var in program.trainable_variables:
        print(f"  - {var.name}")

    # -------------------------------------------------------------------------
    # Visualize Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Visualize Program")
    print("=" * 60)

    synalinks.utils.plot_program(
        program,
        to_folder="guides",
        show_module_names=True,
        show_trainable=True,
        show_schemas=False,
    )

    print("\nProgram visualization saved to guides/traced_qa.png")

    # -------------------------------------------------------------------------
    # Multiple Traced Calls
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 7: Multiple Traced Calls")
    print("=" * 60)

    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is deep learning?",
    ]

    print("\nRunning multiple queries (each emits its own span tree):")
    for q in queries:
        result = await program(Query(query=q))
        print(f"  Q: {q[:30]}... -> A: {result['answer'][:40]}...")

    # -------------------------------------------------------------------------
    # Tracing Structure
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Span Tree Shape")
    print("=" * 60)

    print(
        """
Each program invocation produces a tree of spans of this shape:

Program span (Functional: traced_qa)
└── Generator span
    └── LanguageModel span
        └── (one HTTP call to the configured provider)

Each span is logged twice: once on entry (with the input data model)
and once on exit (with the output data model). The call_id and
parent_call_id fields are sufficient to reconstruct the tree.
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
