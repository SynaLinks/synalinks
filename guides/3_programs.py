"""
# Programs

So far you have seen `Generator` calls one at a time. Real applications
chain several of them together — answer the question, then summarize the
answer, then translate the summary. In this guide we meet `Program`, the
object that bundles many modules into one thing you can call, save,
load, and (later on) train.

The mental picture to start with is a **flowchart**: each box is a
`Module`, the arrows show how data flows between boxes, and the whole
flowchart is itself a `Module` you can drop inside an even bigger
flowchart. The framework keeps this flowchart as inspectable data — not
just as a hidden chain of Python function calls — and that distinction
is what makes saving and training possible.

A **Module**, recall, is the smallest reusable building block — one
input, one output, maybe some internal state. A **Program** is simply a
`Module` built out of other `Module`s, wired together.

If you like a slightly more formal description: a `Program` is a pair
`(G, θ)`.

- `G = (V, E)` is a **directed acyclic graph** ("DAG" — a flowchart
  where you can't loop back to a box you've already visited). The
  vertices `V` are modules; the edges `E` carry typed placeholders
  (`SymbolicDataModel`s — they describe the *shape* of the data that
  will flow, not the data itself).
- `θ` (the Greek letter "theta") is the bag of **trainable
  variables** attached to the modules. In a neural network these
  would be floating-point weights. Here each variable is a
  **JSON object** with a fixed schema — an *interpretable* data
  structure the optimizer is allowed to rewrite. The most common
  cases are a `Generator`'s system instruction (a JSON variable
  whose main field is a string of natural-language guidance) and
  its few-shot examples (a JSON variable whose main field is a
  list of input/output pairs), but in general a trainable variable
  can hold any structured state — see [Guide 11](Trainable%20Variables.md).

In one sentence: a `Program` is a flowchart of modules plus the knobs
the framework is allowed to tune.

Why wrap a flowchart this way instead of just writing a regular Python
function that calls `generator1`, then `generator2`, and so on? Three
reasons:

1. **Trainability.** The optimizers in `synalinks.optimizers` reach into
   `θ` and update those knobs in place. Only variables exposed on graph
   vertices are visible to the optimizer; variables hidden inside an
   ordinary Python function are invisible to it.
2. **Serializability.** `program.save(...)` writes the flowchart and
   the current values of every knob to a JSON file, and
   `Program.load(...)` reads them back. Your trained program survives
   process restarts.
3. **Introspectability.** `program.summary()` lists every box in the
   order it will run, the shape of its output, and how many tunable
   knobs it owns. It is your main debugging tool.

`Program` inherits from two classes you will see in API docs:

- `Trainer` — provides `compile()` and `fit()`, the training loop.
- `Module` — provides `__call__`, `build`, `call`, `get_config`, and
  the variable-tracking machinery.

A `Program` is itself a `Module`, so you can drop one inside another
larger program as a single box. That nesting is how big systems are
assembled.

## Why a Graph Is Better Than a Function

A `Program` is **declarative**: you describe the flowchart once, as
data, and the runtime walks it every time you call the program. This is
not how a normal Python script works. A normal script is
**procedural** — the framework only sees a sequence of opaque function
calls, and cannot reason about their structure or improve them.

```mermaid
graph LR
    A["Function"] --> B["API call 1"]
    B --> C["Parse"]
    C --> D["API call 2"]
    D --> E["Return"]
```

A `Program` exposes the same chain as data the framework can read and act
on:

```mermaid
graph LR
    A["Input DataModel"] --> B["Module 1"]
    B --> C["Module 2"]
    C --> D["Output DataModel"]
    T["Optimizer"] -.-> B
    T -.-> C
    S["save / load"] -.-> B
    S -.-> C
```

The solid arrows are the **forward pass** — the flow of data when you
run the program. The dashed arrows are *not* part of execution; they
represent extra things the framework can do *to* each module (train it,
save it) precisely because it knows the module is there. A plain Python
function hides its insides from the framework, so no dashed arrows are
possible.

## Four Ways to Build a Program

There are four ways to build a `Program`. They differ in (a) how much
code you write and (b) how much of the structure the framework can see
ahead of time. We will look at each in turn.

```mermaid
graph TD
    A["Program construction"] --> B["Functional API"]
    A --> C["Subclassing API"]
    A --> D["Sequential API"]
    A --> E["Mixing strategy"]
    B --> F["Explicit DAG; full introspection"]
    C --> G["Imperative call(); opaque body"]
    D --> H["Linear chain; sugar over Functional"]
    E --> I["Deferred build(); reusable component"]
```

The **Functional API** is the default; reach for it first. The other
three exist for situations where a single fixed flowchart is either
*impossible* (Subclassing, when the next step depends on runtime data)
or *overkill* (Sequential, Mixing).

### Strategy 1: Functional API

Think of building a Functional program as wiring up Lego bricks: you
start from an input piece, snap modules onto it one at a time, and at
the end you tell `Program` which piece is the entrance and which is the
exit. The framework records the wiring as it happens.

Concretely, you create an `Input` placeholder, call modules on
**symbolic** values (placeholders that say "a value of this shape will
arrive here at runtime"), and pass the resulting endpoints to
`Program(inputs=..., outputs=...)`.

Because every wire is a `SymbolicDataModel` — a typed placeholder
rather than real data — the framework can do three useful things
*before* the program ever runs:

- Sort the modules into the right execution order (a **topological
  sort** — picking an order in which every box runs only after its
  inputs are ready).
- Check that the output type of one module matches the input type of
  the next.
- Catch type mismatches up front, instead of mid-run after several LM
  calls have already happened.

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    \"\"\"User question.\"\"\"
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    \"\"\"Answer with reasoning.\"\"\"
    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    # 1. Symbolic entry point.
    inputs = synalinks.Input(data_model=Query)

    # 2. Calling a module on a symbolic value adds a vertex to the graph.
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    # 3. Freeze the graph as a Program.
    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
    )

    # 4. Calling with a concrete DataModel runs the forward pass.
    result = await program(Query(query="What is 2+2?"))
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

A very common trap: the line `await Generator(...)(inputs)` returns a
`SymbolicDataModel` — still a placeholder, *not* real data. If you try
to read `outputs.answer` at this point you get back the placeholder's
metadata, not an actual answer. Real data only appears once you call
`program(concrete_input)`. (This is the same construction-vs-execution
distinction you saw in [Guide 1](Getting%20Started.md): defining and running are two separate
steps.)

Because the wiring is just Python expressions on symbolic values, you
can express parallel branches, merges (where two branches join back
into one), and content-dependent routing using normal Python syntax.

### Strategy 2: Subclassing API

Sometimes the flowchart is not fixed in advance — which module runs
next depends on the actual data. You might want to keep calling an LM
until its answer passes a check (a `while` loop), branch on what the
user asked (an `if`), or retry on failure. No single static graph can
capture that. The **Subclassing API** is the escape hatch: you write
the forward pass as ordinary async Python and trade some framework
visibility for full programming flexibility.

To use it, you inherit from `synalinks.Program` and override `call`:

```python
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer")

class QAProgram(synalinks.Program):
    \"\"\"QA program built by subclassing.\"\"\"

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model
        # Create sub-modules in __init__ so they are tracked as attributes
        # and their variables are picked up by the optimizer.
        self.generator = synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
        )

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        return await self.generator(inputs, training=training)
```

Three rules to remember when subclassing:

- **Create sub-modules as attributes in `__init__`** (or in `build`).
  The act of assigning a module to `self.something` is what tells the
  framework "this is mine, track its variables." A module created as a
  *local variable* inside `call` is invisible to the optimizer, and its
  knobs will never get trained.
- `call` receives **concrete** `JsonDataModel` values — real data —
  not symbolic placeholders.
- The framework cannot peek inside your `call`, so `program.summary()`
  shows a subclassed program as a single opaque box rather than
  enumerating its inner modules.

### Strategy 3: Sequential API

If your flowchart is a straight line — one input, one output, no
branches — writing out the Functional API by hand is just repetitive.
`Sequential` is a shorthand for the case where step *i+1* takes
whatever step *i* produced. Mathematically, you are computing
`f_n(... f_2(f_1(x)) ...)`:

```mermaid
graph LR
    A["Input"] --> B["Module 1"]
    B --> C["Module 2"]
    C --> D["Module 3"]
    D --> E["Output"]
```

```python
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

class Thinking(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Step by step thinking")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer")

lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

program = synalinks.Sequential(
    name="sequential_qa",
    description="A sequential question-answering pipeline",
)
program.add(synalinks.Input(data_model=Query))
program.add(synalinks.Generator(data_model=Thinking, language_model=lm))
program.add(synalinks.Generator(data_model=Answer, language_model=lm))
```

The output type of each stage must be a valid input type for the
next. The moment the chain branches, merges, or has more than one
input or output, switch to the Functional API.

### Strategy 4: Mixing strategy (deferred build)

Suppose you want to write a *reusable component* — say, a generic
"Chain of Thought" wrapper — whose internal flowchart depends on the
type of input it receives. You cannot build the flowchart in
`__init__` because at that point you do not yet know the input shape.

The fix is called **deferred build**: store the hyperparameters in
`__init__`, and override `build(inputs)`. The framework calls `build`
exactly once, with a symbolic placeholder, the first time the
component is used. Inside `build` you assemble the Functional graph
using that placeholder, then call `super().__init__(inputs=...,
outputs=...)` a second time to install the graph onto the component
itself.

```mermaid
graph TD
    A["__init__ stores hyperparameters"] --> B["First call with SymbolicDataModel"]
    B --> C["build(inputs) creates Functional DAG"]
    C --> D["super().__init__ installs DAG"]
    D --> E["Component now behaves like a Functional Program"]
```

```python
import synalinks

class ChainOfThought(synalinks.Program):
    \"\"\"Reusable chain-of-thought component.\"\"\"

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model

    async def build(self, inputs: synalinks.SymbolicDataModel) -> None:
        outputs = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=self.language_model,
        )(inputs)

        # Re-initialize *this* program with the freshly built graph.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
        )
```

The trap to watch for: calling `super().__init__` a **second** time
inside `build` looks like a mistake, but it is intentional. The first
call (in `__init__`) registers the object as a bare `Program` with no
graph yet; the second call upgrades it to a fully wired Functional
`Program` now that the input type is known. Skip the second call and
your component has no graph and refuses to run.

## What You Get for Free

Once you have a `Program`, the framework offers three conveniences you
do not have to write yourself.

### Saving and loading

A `Program` saves to a single JSON file. The file contains the
flowchart structure, the configuration of every module, and the
current values of every trainable variable. Loading reconstructs an
equivalent program with all training progress intact — no separate
checkpoint format, no per-module surgery.

```python
program.save("my_program.json")
loaded = synalinks.Program.load("my_program.json")
```

### `program.summary()`

Call `program.summary()` and you get a table: every module in the
program, listed in execution order, with its name, its type, the JSON
schema of its output, and the number of trainable variables it owns.
This is your main debugging tool for checking that a Functional
program is wired the way you intended. Note: `summary()` cannot see
inside a subclassed `call`, so a subclassed program appears as a
single opaque box.

### Batch inference

`program.predict(xs)` runs the program on every element of the list
`xs` concurrently — in parallel — and returns the list of results.
Use it when you have many independent inputs and want the total wall
time to be roughly that of a single call instead of the sum. If the
inputs are *not* independent — say the output of one feeds the next
— call them one at a time instead.

```python
results = await program.predict([query1, query2, query3])
```

## Expected output

Running this file end-to-end (with `ollama/llama3.2:latest` and the prompts
below) produces:

```
============================================================
Strategy 1: Functional API
============================================================

Functional API Result: 4

============================================================
Strategy 2: Subclassing API
============================================================

Subclassing API Result: 6

============================================================
Strategy 3: Sequential API
============================================================

Sequential API Result: 8

============================================================
Strategy 4: Mixing Strategy
============================================================

Mixing Strategy Result: 10

============================================================
Program Features
============================================================

Program Summary:
Program: functional_qa
description: 'A `Functional` program is a `Program` defined as a directed graph
of modules.'
(table: input_module / generator, variable counts 0 and 1)

Saved program to functional_qa.json
Loaded program: functional_qa
```

The numeric answers are stable across runs because the prompts are simple
arithmetic. The `thinking` strings vary from run to run (they come from a
non-deterministic language model) and are therefore not shown.

## Which Strategy When

A quick decision guide:

- **The flowchart is fixed up front, with branches allowed.** Use the
  **Functional API**. It is the only strategy where `summary()` and
  most graph-level optimizations can see the full structure.
- **The forward pass needs Python control flow** — `if`, `while`,
  recursion, retry-on-failure. Use **Subclassing**, and accept that
  the framework sees your `call` as a black box.
- **The flowchart is a strict linear chain, no branches.** Use
  **Sequential** for less boilerplate. Switch to Functional the moment
  you add a second output or a side branch.
- **You are writing a reusable component whose internal graph depends
  on the input type.** Use the **Mixing strategy** so the graph is
  built lazily, once the input type is known, and then frozen.

## Take-Home Summary

- A **`Program`** is the pair `(G, θ)` — a DAG of modules plus
  the trainable JSON variables attached to them.
- **Construction is not execution.** Building a Program draws
  the flowchart; calling it runs the flowchart. Confusing the
  two is the single most common Functional-API confusion.
- Four ways to build: **Functional** (the default — explicit
  DAG), **Subclassing** (when the forward pass needs Python
  control flow), **Sequential** (linear chains, sugar over
  Functional), and the **Mixing strategy** (deferred-build
  reusable components).
- A `Program` is itself a `Module`, so it can be nested inside
  another `Program` as a single box.
- Three things you get for free: **`save` / `load`**,
  **`summary()`**, and **`predict()`** (parallel batch
  inference).

## API References

- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [Sequential](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Sequential%20class/)
- [Input](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Input%20module/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User question."""

    query: str = synalinks.Field(description="User question")


class ThinkingOutput(synalinks.DataModel):
    """Intermediate thinking output."""

    thinking: str = synalinks.Field(description="Step by step thinking")


class Answer(synalinks.DataModel):
    """Final answer."""

    answer: str = synalinks.Field(description="The final answer")


class AnswerWithThinking(synalinks.DataModel):
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# Subclassing Example
# =============================================================================


class QAProgram(synalinks.Program):
    """A QA program using subclassing."""

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model
        self.generator = synalinks.Generator(
            data_model=Answer,
            language_model=language_model,
        )

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        return await self.generator(inputs, training=training)


# =============================================================================
# Mixing Strategy Example
# =============================================================================


class ChainOfThought(synalinks.Program):
    """Reusable chain-of-thought component."""

    def __init__(self, language_model, **kwargs):
        super().__init__(**kwargs)
        self.language_model = language_model

    async def build(self, inputs: synalinks.SymbolicDataModel) -> None:
        outputs = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=self.language_model,
        )(inputs)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
        )


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_3_programs",
    # )

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    # -------------------------------------------------------------------------
    # Functional API
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Strategy 1: Functional API")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
    )(inputs)

    functional_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="functional_qa",
    )

    result = await functional_program(Query(query="What is 2+2?"))
    print(f"\nFunctional API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # Subclassing API
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 2: Subclassing API")
    print("=" * 60)

    subclass_program = QAProgram(language_model=lm, name="subclass_qa")
    result = await subclass_program(Query(query="What is 3+3?"))
    print(f"\nSubclassing API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # Sequential API
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 3: Sequential API")
    print("=" * 60)

    sequential_program = synalinks.Sequential(
        name="sequential_qa",
        description="A sequential question-answering pipeline",
    )
    sequential_program.add(synalinks.Input(data_model=Query))
    sequential_program.add(
        synalinks.Generator(data_model=ThinkingOutput, language_model=lm)
    )
    sequential_program.add(synalinks.Generator(data_model=Answer, language_model=lm))

    result = await sequential_program(Query(query="What is 4+4?"))
    print(f"\nSequential API Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # Mixing Strategy
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Strategy 4: Mixing Strategy")
    print("=" * 60)

    cot = ChainOfThought(language_model=lm, name="cot_component")

    inputs = synalinks.Input(data_model=Query)
    outputs = await cot(inputs)

    mixing_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="mixing_qa",
    )

    result = await mixing_program(Query(query="What is 5+5?"))
    print(f"\nMixing Strategy Result: {result['answer']}")

    # -------------------------------------------------------------------------
    # Program Features
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Program Features")
    print("=" * 60)

    print("\nProgram Summary:")
    functional_program.summary()

    functional_program.save("functional_qa.json")
    print("\nSaved program to functional_qa.json")

    loaded = synalinks.Program.load("functional_qa.json")
    print(f"Loaded program: {loaded.name}")

    if os.path.exists("functional_qa.json"):
        os.remove("functional_qa.json")


if __name__ == "__main__":
    asyncio.run(main())
