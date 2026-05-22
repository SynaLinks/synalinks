# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Custom Modules with Trainable Variables

In a few guides we will meet **training** — the process that runs
your `Program` on labeled examples, scores its predictions with a
reward, and rewrites the program's internal knobs to do better
next time. Before we can talk about that loop sensibly ([Guide 14](Training.md)),
we need to understand what *kind of state* on a `Module` the
optimizer is even allowed to touch. That is the topic of this
guide. By the end you will have written a `Module` of your own
that owns trainable state, and you will know exactly what the
optimizer expects to see once training begins.

A **trainable variable** is a piece of state your module owns — like
a normal Python attribute — except the framework knows about it.
Because the framework knows about it, it can save it to disk, load it
back later, and let an **optimizer** (the component that improves
your program during training) change it between batches of data.

In classical deep learning, the equivalent object is a **weight
tensor** — a grid of numbers that gradient descent nudges every
step. In Synalinks, the analogous object holds structured JSON (a
dictionary with a fixed shape) instead of numbers, and the optimizer
is an *in-context search procedure* — a loop that proposes new JSON
values and keeps the ones that score well — rather than stochastic
gradient descent.

The **interface** you write as a developer is deliberately
Keras-shaped: you declare state inside a module, the module exposes
it through a list called `trainable_variables`, the trainer reads
and writes that state during training, and saving/loading round-trips
it automatically. If you have ever written a Keras `Layer` with a
`self.kernel = self.add_weight(...)` line in `build`, this is the
same shape — only the *contents* of the weight are different.

This guide builds the picture bottom-up. We introduce the `Variable`
container (the box that holds the state), state precisely when a
variable is **trainable** (allowed to be changed by the optimizer),
walk through what the optimizer actually sees, and finish with a
small runnable program whose output is reproduced verbatim below.

## 1. Parameters: classical ML versus Synalinks

In machine learning, a **parameter** (often called a **weight**) is
a piece of state the system learns from data. In a tensor framework
like PyTorch or TensorFlow, a parameter is a contiguous block of
floating-point numbers with a fixed shape; training rewrites those
numbers in place using **gradients** (calculus telling each number
which way to move).

In Synalinks, a parameter is instead a **JSON value with a fixed
schema** — a fixed list of field names, types, and default values.
Training rewrites that JSON in place using a search loop driven by a
language model and a **scalar reward** (a single number scoring how
well the current value works).

```mermaid
graph LR
    subgraph Tensor["Classical ML"]
        A1["Tensor weight"] --> A2["Gradient"]
        A2 --> A3["SGD update"]
        A3 --> A1
    end
    subgraph Synalinks["Synalinks"]
        B1["JSON variable"] --> B2["Reward"]
        B2 --> B3["Candidate proposer"]
        B3 --> B1
    end
```

Despite the different contents, both systems share the same shape:
a **container** holds mutable state, a **signal** scores the current
value, and an **update rule** produces a new value from the old one.
The two boxes that change are *what flows through the container*
(numbers vs. JSON) and *what produces the update* (gradient descent
vs. an LM-driven proposer).

## 2. The `Variable` Container

The framework's container for state is the class `synalinks.Variable`
(source in `synalinks/src/backend/common/variables.py`). Think of it
as a smart dictionary that the framework can find, save, and let
optimizers touch.

A `Variable` holds three pieces of information:

- a **current JSON value**, accessed via `get_json()` (returns the
  whole dict) or `get(field)` (returns one field),
- a **JSON schema** describing what fields and types are allowed,
  accessed via `get_schema()`,
- a **`trainable` flag** — a boolean saying whether optimizers are
  allowed to rewrite it.

There are three ways to mutate a `Variable`, listed from most surgical to
most sweeping:

```python
self.state.get("instructions")           # read one field
self.state.update({"history": [...]})    # merge keys in place
self.state.assign(new_json_dict)         # full replacement
```

It is worth contrasting `Variable` with `JsonDataModel`, since they
look similar but play opposite roles. A `JsonDataModel` is the
**value flowing between modules during a single call** — the input
to your module, the output of your module — and is *immutable* for
the duration of that call; you cannot change it in place. A
`Variable` is the **state carried by a module across calls** and is
*mutable* by design. Mix these up and you will either try to mutate
something that refuses to change, or accidentally let one call leak
state into the next.

## 3. What Makes a Variable *Trainable*?

Intuition first: a variable is *trainable* if it carries the extra
bookkeeping fields the optimizer needs in order to safely change it.
If those fields are not there, the framework refuses to mark the
variable as trainable, because there would be nowhere to record the
optimizer's internal state.

Formally, the rule lives in a single line inside `Module.add_variable` (the
method you call to register a new variable on a module):

```python
trainable = trainable and is_trainable(data_model)
```

In plain English: a variable counts as trainable only if you *asked* for it
to be trainable *and* its DataModel actually supports training.
`is_trainable(data_model)` returns True if and only if the DataModel's
schema *contains* the `synalinks.Trainable` schema, which in practice means
the DataModel inherits from `synalinks.Trainable`. This is not stylistic;
the optimizer relies on a fixed set of bookkeeping fields being present on
every trainable variable:

| field                  | role                                                      |
|------------------------|-----------------------------------------------------------|
| `examples`             | few-shot examples currently in use                        |
| `current_predictions`  | predictions made during the *current* batch               |
| `predictions`          | predictions accumulated across the epoch                  |
| `seed_candidates`      | warm-start candidates (user-provided)                     |
| `candidates`           | candidates proposed by the optimizer this epoch           |
| `best_candidates`      | best-so-far candidates (size capped by `population_size`) |
| `history`              | sequence of selected best candidates over time            |
| `nb_visit`             | times this variable was used in the current batch         |
| `cumulative_reward`    | sum of rewards collected on `nb_visit` predictions        |

Those nine fields come for free when you inherit from `synalinks.Trainable`.
On top of them, your subclass adds the field(s) you actually want to learn —
the equivalent of a "weight" in a neural network:

```python
import synalinks

class Persona(synalinks.Trainable):
    \"\"\"How the assistant should speak.\"\"\"
    persona: str = synalinks.Field(
        description="A short description of the assistant's tone and style.",
        default="A friendly, concise assistant.",
    )
```

Two rules the framework relies on, both of which catch people out:

- **Every field of a `Trainable` subclass must have a default
  value.** When you build a program, the framework needs to
  construct a *placeholder* instance of your DataModel *before* any
  real data has flowed through. If a field has no default, that
  construction crashes during symbolic setup (when you call
  `synalinks.Input(...)`) — not later at runtime, where it would be
  harder to diagnose. A *default* is simply the value the field
  takes if no one supplies one.
- **`add_variable` silently downgrades a request to
  `trainable=False`** when the DataModel does not subclass
  `Trainable`. There is no error; the variable just quietly becomes
  non-trainable. That is correct behavior for genuinely
  non-trainable state (counters, caches), but it is the single most
  common cause of "why is my variable missing from
  `program.trainable_variables`?"

## 4. Declaring the Variable: `Module.add_variable`

Variables are not declared *just anywhere* — you create them inside
the module's constructor (or its `build` method, see below) so the
framework can keep an accurate list of them. That list is frozen the
first time the module runs, so the optimizer sees the *same* set of
variables on every batch.

Concretely: state must be created in `__init__()` (the constructor), or in
`build(self, input_schema)` if the variable's shape depends on the input
schema (what kind of data will be passed in). After the first call to the
module, the module's variable *tracker* — the internal list that remembers
every `Variable` you registered — locks; any later `add_variable` call
raises an error. This mirrors Keras's "build once" semantics.

```python
class Personalize(synalinks.Module):
    def __init__(self, persona=None, name=None, description=None, trainable=True):
        super().__init__(name=name, description=description, trainable=trainable)
        self.persona = persona

        # Let the DataModel own its default. Only pass `persona` through
        # when the caller actually provided one.
        seed = Persona(persona=persona) if persona is not None else Persona()

        # Register the variable. The module's __setattr__ runs every
        # assignment through a Tracker that recognises Variable instances
        # and routes them into `trainable_variables`.
        self.state = self.add_variable(
            initializer=seed.get_json(),
            data_model=Persona,
            name="persona_" + self.name,
        )
```

Three details deserve attention:

- `initializer` is the starting value of the variable. It accepts either a
  JSON dict or an `Initializer` callable (a function-like object that
  produces an initial value), such as
  `synalinks.initializers.Empty(data_model=Persona)`. The dict form is
  what you will use in almost every case.
- `name` is the human-readable identifier optimizers log and what
  `get_variable(name=...)` looks up. Suffixing the name with `self.name`
  keeps variables unique inside programs that instantiate the same module
  several times.
- The assignment `self.state = ...` is structural, not cosmetic. `Module`
  intercepts every attribute assignment via Python's `__setattr__`
  mechanism, recognises any value that is a `Variable`, and routes it into
  the module's `trainable_variables` list. That is how the framework
  discovers your variables automatically — there is no separate
  "register this variable" call.

## 5. Reading and Writing State Inside `call()`

`call(self, inputs, training=False)` is the method that runs your
module's logic for one example. It is an `async` method (defined
with `async def`, awaited by the framework), it takes a
`JsonDataModel` as input, and returns a `JsonDataModel`. The
`training` flag tells the module whether it is currently being
trained — when `True`, the module owes the optimizer some extra
bookkeeping (we will see what in a moment).

Inside the body, the variable is just structured state you read from:

```python
async def call(self, inputs, training=False):
    persona = self.state.get("persona")
    greeting = f"[{persona}] hello {inputs.get('name')}"
    return Greeting(greeting=greeting)
```

When `training=True`, the contract widens. You must append a record to
the `current_predictions` field so the optimizer can later figure out which
reward to attribute to this variable. Without that record, the optimizer
has no evidence that the variable was used and will skip it. The pattern
is identical to what `Generator.call` does internally:

```python
if training:
    predictions = self.state.get("current_predictions")
    predictions.append(
        {
            "inputs": inputs.get_json(),
            "outputs": result.get_json(),
            "reward": None,  # filled in by the optimizer
        }
    )
```

After the batch's rewards are computed (one reward per training example),
the optimizer's method `assign_reward_to_predictions` walks every trainable
variable, pairs each `None`-reward record with its scalar reward,
increments `nb_visit` (the count of how many times this variable was used),
and adds to `cumulative_reward` (the running total). Skip the append step,
and your variable looks unvisited; the optimizer will see no evidence and
propose nothing useful.

## 6. Non-trainable Variables: same API, narrower contract

Sometimes a module needs to remember things across calls but you do
**not** want an optimizer touching them — for example, a counter of
how many times the module ran, a cache of previously seen inputs, or
a conversation memory. The API is the same `add_variable` call, just
with `trainable=False` (or with a DataModel that does not subclass
`Trainable`):

```python
class CallStats(synalinks.DataModel):
    count: int = synalinks.Field(default=0, description="Calls so far.")

self.stats = self.add_variable(
    initializer=CallStats().get_json(),
    data_model=CallStats,
    trainable=False,
    name="stats_" + self.name,
)
```

Such a variable will appear in `module.non_trainable_variables` and in
`module.variables` (the union of trainable and non-trainable), but *not* in
`module.trainable_variables`. You can read and write it freely from
`call()`; the optimizer will neither propose new values for it nor revert
it between batches.

## 7. `build()` versus `__init__()`

You have a choice of *where* to create a variable. Two rules cover
it:

- If the variable's schema is **fixed and fully known** at module
  construction time, create it in `__init__` (the constructor).
- If the schema **depends on the module's input schema** — for
  example, a variable whose fields mirror `inputs.get_schema()`,
  whose shape depends on what data will eventually flow in — create
  it inside `build(self, input_schema)`. The framework calls
  `__call__` on your module the first time it runs; `__call__`
  invokes `build` automatically on that first call and then locks
  the variable tracker (the internal list of variables) so no
  further variables can be added.

The locked tracker is a **load-bearing invariant** — meaning lots of
other machinery depends on it. It guarantees that the set of
trainable variables a `Program` reports is the same set the
optimizer sees on every batch. Adding state mid-training would
silently invalidate the optimizer's bookkeeping.

## 8. The Optimizer's View of Your Module

Each batch — a small group of training examples processed together —
the trainer runs a fixed sequence of steps. The contract you owe the
optimizer is narrow: declare a `Trainable` DataModel, expose it
through `add_variable`, read it in `call`, and append to
`current_predictions` when `training=True`. Everything else —
proposing new values, scoring them, keeping the best — is the
optimizer's responsibility.

```mermaid
sequenceDiagram
    participant T as Trainer
    participant O as Optimizer
    participant V as Variable
    participant M as Module.call

    T->>O: on_train_begin
    O->>V: seed candidates / best_candidates
    loop per batch
        T->>O: on_batch_begin
        O->>V: pick candidate, write into primary fields
        T->>M: predict_on_batch(training=True)
        M->>V: append to current_predictions
        T->>O: assign_reward_to_predictions
        O->>V: update nb_visit, cumulative_reward
        O->>V: move current_predictions to predictions
        T->>O: propose_new_candidates
        T->>O: on_batch_end
        O->>V: write best candidate into primary fields
    end
```

Different optimizer subclasses differ only in **how** they propose
new candidates: a random-few-shot optimizer resamples examples from
a pool, an LM-based optimizer asks a language model to mutate the
JSON, a genetic optimizer crosses two "parent" candidates to make a
child. The interface on **your** side is the same regardless.

## 9. Serialization: `get_config` and `from_config`

Saving a program to disk has two halves. First, the framework needs
to know **how to rebuild your module** — what class, what
constructor arguments. Second, it needs to know **what values its
variables currently hold**.

`synalinks.Program.save(...)` stores both. The "how to rebuild" half
comes from `get_config()`, which returns a JSON-serializable dict;
the rebuild itself is performed by `from_config(cls, config)`. The
variable values are saved and reloaded by the saving system
automatically — you do not write that code yourself.

```python
def get_config(self):
    return {
        "persona": self.persona,
        "name": self.name,
        "description": self.description,
        "trainable": self.trainable,
    }

@classmethod
def from_config(cls, config):
    return cls(**config)
```

If your module holds another Synalinks object as an attribute (a
`LanguageModel`, an `EmbeddingModel`, a sub-`Module`), you cannot just put
it directly into the dict — serialise it with
`synalinks.saving.serialize_synalinks_object(obj)` in `get_config` and
restore it with `synalinks.saving.deserialize_synalinks_object(...)` in
`from_config`. See `examples/9_custom_modules.py` for the canonical
example.

Carrying an *initial* value in `get_config` (such as `"persona"`
above) is fine and not duplication: it gives `from_config` a usable
seed value *before* the saved variable state is loaded on top. The
loader overwrites the seed with the persisted value, so the variable
ends up holding what it held when you saved.

## 10. Putting It All Together (runnable)

The `main()` function below builds the `Personalize` module from this
guide, wraps it in a tiny program, and demonstrates four things:

- discovery: the variable shows up in `program.trainable_variables`,
- effect: the persona field controls the deterministic output,
- update: a manual `update()` call stands in for one optimizer step
  (changing the persona by hand, the way an optimizer would),
- persistence: the non-trainable counter keeps incrementing across calls.

The example does not call any language model — it is fully deterministic.
Running `program.fit(...)` in earnest requires three more ingredients: a
reward (e.g. `synalinks.ExactMatch`, which scores 1.0 if the output matches
the expected answer and 0.0 otherwise), an optimizer (e.g.
`synalinks.optimizers.OMEGA` for free-form mutation or
`synalinks.optimizers.RandomFewShot` for example selection), and a
dataset. See `examples/9a_trainable_variables.py` for a richer version
with a `Generator` wrapped around the same trainable variable.

Expected output (deterministic, no LM involved):

```
============================================================
Trainable variables collected by the program
============================================================
  - persona_personalize: fields = ['examples', 'current_predictions', 'predictions', 'seed_candidates', 'candidates', 'best_candidates', 'history', 'nb_visit', 'cumulative_reward', 'persona']

Non-trainable variables (runtime state, not optimized):
  - stats_personalize: fields = ['count']

============================================================
Running the program with the initial persona
============================================================
  output: [A pirate who shouts.] hello Ada

============================================================
Simulating an optimizer step: assign() a new persona
============================================================
  new persona: A precise, formal academic.
  output: [A precise, formal academic.] hello Ada

============================================================
Non-trainable state: the call counter
============================================================
  stats_personalize: count = 2
```

Two things to read off the output. First, the trainable variable's field
list contains all nine `Trainable` bookkeeping fields *and* the
user-defined `persona` field — the optimizer's machinery is genuinely
there even though we never invoke an optimizer. Second, the call counter
ends at `2`, not `1`: the program is called once before the manual update
and once after, and the non-trainable counter survives both calls because
variables persist across calls by design.

## Take-Home Summary

- **State lives in `Variable`s**, created via
  `self.add_variable(...)` in `__init__` or `build`. Variables are
  JSON-valued, schema-typed (their structure is fixed), and mutable
  across calls. `JsonDataModel`s, by contrast, are the values
  passed between modules and are immutable for a single call.
- **Trainable if and only if the DataModel subclasses
  `synalinks.Trainable`.** No subclass, no trainability —
  `add_variable` silently downgrades to `trainable=False`. Every
  field of a `Trainable` subclass needs a default value.
- **`call()` reads state**; under `training=True` it also appends a
  record to `current_predictions` so the optimizer can match
  rewards to the predictions that produced them.
- **Non-trainable variables share the same API**, with
  `trainable=False` or a non-`Trainable` data model. Use them for
  runtime counters, caches, and per-conversation state.
- **Optimizers see only `program.trainable_variables`.** Their
  contract is fixed: `seed_candidates` / `candidates` /
  `best_candidates` go in, a new value comes out. Your module never
  needs to know which optimizer is being used.
- **`get_config` / `from_config` rebuild the module's *shape*.**
  The variable *values* are saved and loaded separately by the
  saving system, so you do not have to handle them yourself.

## API References

- [Module (Base Class)](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Base%20Module%20class/)
- [Variable](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Variable%20class/)
- [Trainable](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
- [Optimizers](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/)
- [Generator (canonical trainable module)](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
"""

import asyncio

import synalinks


# =============================================================================
# Data Models
# =============================================================================


class Persona(synalinks.Trainable):
    """How the assistant should speak.

    Subclassing `Trainable` is what makes this DataModel eligible for
    optimization. Every field must have a default value.
    """

    persona: str = synalinks.Field(
        description="A short description of the assistant's tone and style.",
        default="A friendly, concise assistant.",
    )


class CallStats(synalinks.DataModel):
    """Plain DataModel — *not* trainable, just persistent state."""

    count: int = synalinks.Field(
        description="Number of times the module has been called.",
        default=0,
    )


class Name(synalinks.DataModel):
    """Module input."""

    name: str = synalinks.Field(description="The user's name.")


class Greeting(synalinks.DataModel):
    """Module output."""

    greeting: str = synalinks.Field(description="A greeting addressed to the user.")


# =============================================================================
# Custom Module
# =============================================================================


class Personalize(synalinks.Module):
    """Greets the user in a style controlled by a trainable persona variable.

    The `call()` method is deterministic and does not use a language model,
    so this guide can run offline. The *shape* of the module — trainable
    state, non-trainable state, build-time registration, serialization — is
    exactly what you would write for an LM-backed module.
    """

    def __init__(
        self,
        persona=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(name=name, description=description, trainable=trainable)
        self.persona = persona

        # Trainable: subclass of `synalinks.Trainable`, picked up by optimizers.
        # When `persona` is None, the DataModel's own default applies.
        seed = Persona(persona=persona) if persona is not None else Persona()
        self.state = self.add_variable(
            initializer=seed.get_json(),
            data_model=Persona,
            name="persona_" + self.name,
        )

        # Non-trainable: plain DataModel; lives on the module but is invisible
        # to the optimizer.
        self.stats = self.add_variable(
            initializer=CallStats().get_json(),
            data_model=CallStats,
            trainable=False,
            name="stats_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        # Read trainable state.
        persona = self.state.get("persona")

        # Update non-trainable state (runtime bookkeeping).
        self.stats.update({"count": self.stats.get("count") + 1})

        result = Greeting(greeting=f"[{persona}] hello {inputs.get('name')}")

        # When training, the optimizer expects predictions on the variable
        # so it can assign rewards to them. This is the same pattern
        # `Generator` follows internally.
        if training:
            predictions = self.state.get("current_predictions")
            predictions.append(
                {
                    "inputs": inputs.get_json(),
                    "outputs": result.get_json(),
                    "reward": None,
                }
            )

        return result

    async def compute_output_spec(self, inputs, training=False):
        return synalinks.SymbolicDataModel(
            schema=Greeting.get_schema(),
            name="greeting_" + self.name,
        )

    def get_config(self):
        return {
            "persona": self.persona,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =============================================================================
# Demonstration
# =============================================================================


async def main():
    synalinks.clear_session()

    inputs = synalinks.Input(data_model=Name)
    outputs = await Personalize(persona="A pirate who shouts.")(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="personalize_demo",
        description="A toy program demonstrating a trainable variable.",
    )

    # -------------------------------------------------------------------------
    # 1) The variable is discovered as a trainable_variable.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Trainable variables collected by the program")
    print("=" * 60)
    for v in program.trainable_variables:
        print(f"  - {v.name}: fields = {list(v.keys())}")

    print("\nNon-trainable variables (runtime state, not optimized):")
    for v in program.non_trainable_variables:
        print(f"  - {v.name}: fields = {list(v.keys())}")

    # -------------------------------------------------------------------------
    # 2) Run the program. The persona controls the output.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running the program with the initial persona")
    print("=" * 60)
    result = await program(Name(name="Ada"))
    print(f"  output: {result['greeting']}")

    # -------------------------------------------------------------------------
    # 3) Simulate an optimizer step by assigning a new persona.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Simulating an optimizer step: assign() a new persona")
    print("=" * 60)
    persona_variable = program.trainable_variables[0]
    persona_variable.update({"persona": "A precise, formal academic."})
    print(f"  new persona: {persona_variable.get('persona')}")

    result = await program(Name(name="Ada"))
    print(f"  output: {result['greeting']}")

    # -------------------------------------------------------------------------
    # 4) The non-trainable counter has incremented across calls.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Non-trainable state: the call counter")
    print("=" * 60)
    stats_variable = program.non_trainable_variables[0]
    print(f"  {stats_variable.name}: count = {stats_variable.get('count')}")


if __name__ == "__main__":
    asyncio.run(main())
