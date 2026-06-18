"""
# Control Flow

A `Program` is a **static graph**: you wire every module together once,
at construction time, and the framework runs the same flowchart on every
input ([Guide 3](https://synalinks.github.io/synalinks/guides/Programs/)). That sounds like it leaves no room to *react* to the
data — to skip a step, take a different path, or do several things at
once. This guide is about exactly that room. It shows how to express
branching, parallelism, and recombination **without leaving the
declarative graph**, so the result stays inspectable, serializable, and
trainable.

You have already met the pieces. [Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/) introduced the operator
algebra on data models (`+`, `&`, `|`, `^`, `~`). [Guide 4](https://synalinks.github.io/synalinks/guides/Modules/) introduced the
control-flow modules (`Decision`, `Branch`) and the merging modules.
This guide is the layer above both: it assembles those pieces into the
handful of **patterns** you will reach for again and again.

## `None` Is the Control Signal

Here is the single idea the whole guide rests on. In an ordinary Python
program, control flow is about *which lines execute*. In a Synalinks
graph, every wired module is always *present* — but a module on an
inactive path produces `None` instead of a value. So control flow
becomes a question about **values, not lines**:

- a module that should not contribute on this input yields `None`, and
- the **operators** decide what happens when a value might be `None`.

That is why the operators from Guide 2 are not just a convenience for
gluing schemas together — they are the *language of control flow*. `|`
means "whichever path actually ran." `&` means "only if this prerequisite
is present." `^` means "exactly one of these, never both." Reading them
as boolean logic over "is there a value here?" is the right mental model.

```mermaid
graph LR
    A["Input"] --> B["Branch"]
    B -->|"selected"| C["value"]
    B -->|"skipped"| D["None"]
    C --> E["Merge (| )"]
    D --> E
    E --> F["Output"]
```

## Two Kinds of Control Flow

Synalinks gives you two distinct ways to make a program behave
differently on different inputs, and it is worth being deliberate about
which one you pick.

- **Declarative** (this guide). You express the branching *as part of
  the graph*, using `Decision`/`Branch` to route and the operators to
  recombine. The structure stays visible: `program.summary()` shows it,
  the optimizer can train through it, and `program.save()` serializes
  it. The cost is that the *shape* of the flow is fixed — a `Branch`
  always has the same, finite set of arms.
- **Imperative** (the Subclassing API from [Guide 3](https://synalinks.github.io/synalinks/guides/Programs/)). You write the
  forward pass as ordinary `async` Python and use real `if`/`while`/
  recursion. This handles flows a static graph cannot — "keep calling
  the LM until the answer passes a check," for instance — but the
  framework sees your `call` as an opaque box.

The rule of thumb: **reach for declarative control flow first.** Drop to
imperative Python only when the flow is genuinely unbounded or
data-dependent in a way a fixed set of branches cannot capture. Most
routing, fan-out, and fallback logic is declarative.

## Pattern 1: Fan-Out (Parallel Branches)

The simplest non-linear shape is *fan-out*: send the same input to
several modules at once. In Synalinks this is automatic — **whenever two
modules read the same input, they run concurrently.** There is no
special "parallel" construct; the framework infers it from the graph.

```mermaid
graph LR
    A["Input"] --> B["Generator: pros"]
    A --> C["Generator: cons"]
    B --> D["Merge (+)"]
    C --> D
    D --> E["Output"]
```

```python
inputs = synalinks.Input(data_model=Question)

# Both read `inputs` -> they execute in parallel.
pros = await synalinks.Generator(data_model=Pros, ...)(inputs)
cons = await synalinks.Generator(data_model=Cons, ...)(inputs)

# Fan back in: + (Concat) unions the fields of both into one model.
outputs = pros + cons
program = synalinks.Program(inputs=inputs, outputs=outputs)
```

You have two choices for what to do with the parallel results:

- **Keep them separate.** Pass a *list* as the program's outputs
  (`outputs=[pros, cons]`) and the program returns a list of results.
- **Merge them.** Combine the branches with `+` (Concat) into a single
  data model whose fields are the union of both. If two branches share a
  field name, Concat keeps both by suffixing the duplicate (`answer`,
  `answer_1`) so no information is lost.

Use fan-out for ensembles (several answers, pick or vote), multi-faceted
analysis (sentiment *and* topic *and* urgency in one pass), or simply to
shave wall-clock time off independent steps.

## Pattern 2: Routing (Decision and Branch)

The other fundamental shape is *routing*: pick **one** path based on
what the input looks like. The primitive is `Decision` — single-label
classification over a closed set of labels — and `Branch` is `Decision`
wired directly to a list of modules.

A `Branch` returns a **tuple with one slot per label**. At runtime the
classifier picks a label, the module in that slot runs, and every other
slot comes back `None`. So the tuple is "one value, the rest `None`."

```mermaid
graph LR
    A["Input"] --> B["Branch: easy / hard"]
    B -->|"easy"| C["Generator: Answer"]
    B -->|"hard"| D["Generator: Answer+Thinking"]
    C --> E["Collapse (| )"]
    D --> E
    E --> F["Output"]
```

```python
(easy, hard) = await synalinks.Branch(
    question="How hard is this query to answer?",
    labels=["easy", "hard"],
    branches=[
        synalinks.Generator(data_model=Answer, ...),          # for "easy"
        synalinks.Generator(data_model=AnswerWithThinking, ...),  # for "hard"
    ],
    language_model=language_model,
)(inputs)

# Exactly one of (easy, hard) is non-None. Collapse to the live one:
outputs = easy | hard
```

That last line is the canonical routing idiom: **`Branch` produces a
tuple, and `|` collapses it back to the single output that actually
ran.** `|` (Or) returns its first non-`None` operand, so it always hands
you whichever branch fired.

## The Operators as Control Flow

Guide 2 listed the five operators; here is what each one is *for* once
you start thinking of `None` as a signal. The table below is the same
algebra, read as control flow.

| Operator | Reach for it when…                                            |
|----------|---------------------------------------------------------------|
| `\\|` Or  | collapsing a `Branch`, or falling back: `primary \\| backup`.  |
| `&` And  | gating: attach context only if a path is live; `None` if not. |
| `+` Concat | joining parallel results that are *both* expected to exist. |
| `^` Xor  | mutual exclusion: a guard that fires only if exactly one side. |
| `~` Not  | cancelling a path outright (turn any value into `None`).       |

The behavior under missing inputs is what distinguishes them, so keep
this truth table close:

| A      | B      | A `+` B | A `&` B | A `\\|` B | A `^` B |
|--------|--------|---------|---------|-----------|---------|
| value  | value  | merged  | merged  | A         | `None`  |
| value  | `None` | A       | `None`  | A         | A       |
| `None` | value  | B       | `None`  | B         | B       |
| `None` | `None` | `None`  | `None`  | `None`    | `None`  |

Two patterns built straight out of this table:

- **Fallback chain.** `primary | secondary | tertiary` walks left to
  right and yields the first path that produced a value. A cheap model
  with an expensive backup is just `cheap | expensive`.
- **Safe gating.** `inputs & branch_output` attaches the original input
  to a branch's result *only when that branch ran* — if the branch was
  skipped (`None`), the `&` short-circuits to `None` and nothing
  downstream crashes trying to read a missing field. This is the
  difference between `&` and `+`: `+` would raise on the `None`.

When you need to keep only part of a model before merging (to hide a
scratch `thinking` field, say), use the masking helpers from Guide 2 —
`in_mask`/`out_mask` — in the same pipeline.

## Putting It All Together

The example below is one runnable program that uses each pattern. It
fans out for a quick pros/cons pass, routes a query by difficulty and
collapses the branch with `|`, and then demonstrates the raw operator
table on concrete data models so you can see `None` flow through the
algebra with no language model involved.

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="The question to consider")

class Pros(synalinks.DataModel):
    pros: str = synalinks.Field(description="The strongest argument in favor")

class Cons(synalinks.DataModel):
    cons: str = synalinks.Field(description="The strongest argument against")

async def main():
    load_dotenv()
    synalinks.clear_session()
    lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    # Fan-out: both generators read `inputs`, run in parallel, merge with +.
    inputs = synalinks.Input(data_model=Question)
    pros = await synalinks.Generator(data_model=Pros, language_model=lm)(inputs)
    cons = await synalinks.Generator(data_model=Cons, language_model=lm)(inputs)
    program = synalinks.Program(inputs=inputs, outputs=pros + cons)

    result = await program(Question(question="Should small teams adopt microservices?"))
    print(result.get_json())

if __name__ == "__main__":
    asyncio.run(main())
```

## Take-Home Summary

- **Control flow in a graph is about values, not lines.** An inactive
  path yields `None`; the operators decide what `None` means downstream.
- **Fan-out is automatic.** Two modules reading the same input run in
  parallel. Keep the results as a list, or merge them with `+`.
- **Route with `Branch`, collapse with `|`.** A `Branch` returns a tuple
  with one live slot and the rest `None`; `a | b | ...` hands you the
  live one. The same `|` builds fallback chains.
- **`&` is the safe join.** `inputs & maybe_none` attaches context only
  when the path ran, short-circuiting to `None` otherwise — where `+`
  would raise.
- **Prefer declarative control flow** (this guide) so the structure
  stays visible and trainable; drop to the Subclassing API only for
  genuinely unbounded, data-dependent flows.

## API References

- [Decision](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Decision%20module/)
- [Branch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Branch%20module/)
- [Merging Modules (And, Or, Xor, Concat)](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Merging%20Modules/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [JSON Ops](https://synalinks.github.io/synalinks/Synalinks%20API/Ops%20API/JSON%20Ops/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Question(synalinks.DataModel):
    """A question to reason about."""

    question: str = synalinks.Field(description="The question to consider")


class Pros(synalinks.DataModel):
    """The case in favor."""

    pros: str = synalinks.Field(description="The strongest argument in favor")


class Cons(synalinks.DataModel):
    """The case against."""

    cons: str = synalinks.Field(description="The strongest argument against")


class Query(synalinks.DataModel):
    """A user query to route."""

    query: str = synalinks.Field(description="The user query")


class Answer(synalinks.DataModel):
    """A short, direct answer."""

    answer: str = synalinks.Field(description="The correct answer")


class AnswerWithThinking(synalinks.DataModel):
    """An answer with step-by-step reasoning."""

    thinking: str = synalinks.Field(description="Your step by step thinking")
    answer: str = synalinks.Field(description="The correct answer")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()
    synalinks.enable_logging()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_5_control_flow",
    # )

    lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    # -------------------------------------------------------------------------
    # Pattern 1: Fan-out (parallel branches) merged with + (Concat)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Pattern 1: Fan-out + Concat (+)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Question)

    # Both generators read the SAME input -> they run in parallel.
    pros = await synalinks.Generator(
        data_model=Pros,
        language_model=lm,
        name="for",
    )(inputs)
    cons = await synalinks.Generator(
        data_model=Cons,
        language_model=lm,
        name="against",
    )(inputs)

    # Fan back in: + unions the fields of both branches into one model.
    fanout_program = synalinks.Program(
        inputs=inputs,
        outputs=pros + cons,
        name="fan_out",
    )
    fanout_program.summary()

    result = await fanout_program(
        Question(question="Should small teams adopt microservices?")
    )
    print(f"\nMerged fields: {list(result.get_json().keys())}")
    print(f"  pros: {result['pros'][:70]}...")
    print(f"  cons: {result['cons'][:70]}...")

    # -------------------------------------------------------------------------
    # Pattern 2: Routing with Branch, collapsed with | (Or)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pattern 2: Branch routing + Or (|)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Branch returns a tuple with one live slot; the rest are None.
    easy, hard = await synalinks.Branch(
        question="How hard is this query to answer?",
        labels=["easy", "hard"],
        branches=[
            synalinks.Generator(data_model=Answer, language_model=lm),
            synalinks.Generator(data_model=AnswerWithThinking, language_model=lm),
        ],
        language_model=lm,
    )(inputs)

    # | collapses the tuple to whichever branch actually ran.
    routing_program = synalinks.Program(
        inputs=inputs,
        outputs=easy | hard,
        name="routing",
    )
    routing_program.summary()

    result = await routing_program(Query(query="What is 2 + 2?"))
    print(f"\nEasy query -> {result['answer']}")

    result = await routing_program(
        Query(query="Explain why the sky is blue, from first principles.")
    )
    print(f"Hard query -> {result['answer'][:70]}...")

    # -------------------------------------------------------------------------
    # Pattern 3: The operator algebra over None (no LM needed)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pattern 3: Operators as control flow")
    print("=" * 60)

    a = Answer(answer="A")
    b = Cons(cons="B")

    # + (Concat): union of fields when both are present.
    print(f"\n(a + b) fields: {list((a + b).get_json().keys())}")
    # | (Or): first non-None operand -> fallback / branch collapse.
    print(f"(a | None) -> {(a | None).get_json()}")
    print(f"(None | b) -> {(None | b).get_json()}")
    # & (And): safe join, None if either side is missing.
    print(f"(a & None) -> {a & None}")
    # ^ (Xor): value only if exactly one side is present.
    print(f"(a ^ None) is not None -> {(a ^ None) is not None}")
    print(f"(a ^ b) is None      -> {(a ^ b) is None}")
    # ~ (Not): cancel a path.
    print(f"(~a) -> {~a}")


if __name__ == "__main__":
    asyncio.run(main())
