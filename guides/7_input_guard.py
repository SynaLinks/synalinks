"""
# Input Guards

An **input guard** is a gate at the door of your program. Before the
LM ever sees the request, the guard either lets it through unchanged
or says "no" — in which case the LM call is skipped entirely. That
saves money *and* blocks unwanted inputs in one step.

A precise way to describe an input guard is as a pure function
`g : X -> Y ∪ {None}` ("∪" means "or" — the output is either a
warning value of type `Y` or the special value `None`). The guard
classifies inputs *before* they reach an expensive downstream module
(typically an LM call). The two outcomes are labeled with a
deliberately *flipped* convention:

- **`g(x) = None`** means **admit** (let the input through). The
  input passes policy and should proceed to downstream computation.
- **`g(x) = w`** for any non-`None` warning value means **reject**
  (block the input). The input violates a stated rule and downstream
  computation must be skipped.

Why this flipped convention? Because it lets us compose guards using
two simple operators (`^`, `|`) provided by the framework, instead of
writing `if`/`else` statements. The whole program stays a **static
dataflow graph** — a fixed picture of how data flows between modules
— which is easy to reason about, save to disk, and trace at runtime.

## Why Place the Guard *Before* the LM?

Two reasons: cost, and security.

**Cost.** Let `c_g` be the cost of running the guard and `c_m` be
the cost of one LM call. In practice `c_g ≪ c_m` ("much smaller
than"), often by a factor of a thousand or more. So the expected
(average) cost per request is:

    E[cost] = c_g + (1 - p_block) * c_m

Put in English: you always pay for the guard, plus the cost of the
main LM call only when the guard *didn't* block. Here `p_block` is
the fraction of inputs the guard rejects, somewhere between 0 and 1.
Even a modest `p_block` saves real money over time.

**Security.** Rejected inputs **never reach the model's context**,
so they cannot influence its output. This closes off a whole class
of attacks called **prompt injection** — where an attacker hides
malicious instructions inside the user's input, trying to trick the
LM into doing something it shouldn't.

```mermaid
graph LR
    A[input x] --> G{guard g}
    G -->|"g(x) = None"| M[LLM call]
    M --> R1[answer]
    G -->|"g(x) = w"| R2[warning w]
```

## Two Operators, One Pattern

The trick is two operators borrowed from Python's bitwise syntax,
but **redefined for data models**. Think of `None` as "nothing
here." The operators `^` (XOR) and `|` (OR) treat `None` in opposite
ways, and putting them together lets us build an "if/else" without
ever writing one. The tables below describe everything you need.

### XOR as a guarded bypass

The `^` operator returns one operand when *exactly one* of the two
is non-`None`, and returns `None` when **both** are non-`None`. We
exploit that last case on purpose: when the guard fires (produces a
warning), `^` cancels out the input and "extinguishes" it.

| `warning` | `inputs` | `warning ^ inputs`     |
|-----------|----------|------------------------|
| `None`    | `x`      | `x`   (admit: pass through) |
| `w`       | `x`      | `None`  (suppress: drop)    |

```mermaid
graph LR
    W[warning] --> X["^"]
    I[inputs] --> X
    X --> O["guarded_inputs"]
```

### OR as a fallback selector

The `|` operator returns the first non-`None` operand. We use it to
pick what to show the user: the guard's warning, or the LM's answer.

| `warning` | `answer` | `warning \\| answer`   |
|-----------|----------|------------------------|
| `None`    | `a`      | `a`   (use the answer) |
| `w`       | `None`   | `w`   (use the warning)|
| `w`       | `a`      | merged; `w` wins ties  |

Row three is a corner case worth understanding: if both sides are
non-`None`, the operator merges their fields, with the *left*
operand taking priority. In a correctly wired guard graph this case
is **unreachable** — because whenever `warning` is non-`None`,
`guarded_inputs` becomes `None`, which forces `answer` to `None` as
well. So in practice only rows one and two ever fire.

## The Composed Graph

The entire pattern is just four edges:

```mermaid
graph LR
    I[inputs] --> G[InputGuard]
    G --> W[warning]
    W --> X["warning ^ inputs"]
    I --> X
    X --> GEN[Generator]
    GEN --> A[answer]
    W --> OR["warning | answer"]
    A --> OR
    OR --> OUT[output]
```

Let us walk through the two possible execution paths step by step:

**Reject trace** (the guard fires). `g(x) = w` (a warning). Then
`w ^ x = None`, so the generator receives `None` and short-circuits
to `None` (it does nothing and returns nothing). Finally
`w | None = w`. **The LM is never called.**

**Admit trace** (the guard lets the input through). `g(x) = None`.
Then `None ^ x = x`, so the generator runs on the original input
and returns some answer `a`. Finally `None | a = a`.

The crucial **invariant** (property guaranteed to always hold): *the
generator is never invoked on a rejected input.* You can verify
this just by looking at the graph; you do not need to read the
generator's source code.

## Failure Modes to Audit

Any guard is a **binary classifier** — it answers a yes/no question
— and, like any classifier, it can make two kinds of mistakes:

- **False negatives (bypasses).** The guard *should* have blocked
  the input but admitted it. A malicious input slips through.
  Substring matching, as in the example below, is easily defeated
  by **obfuscation** — writing a banned word in a disguised form
  like `h@ck`, `h4ck`, base64, or Unicode escapes. Treat substring
  filters as a cheap *first layer*, not a real security boundary.
- **False positives (over-blocks).** The guard blocks an input that
  was fine — for example because the substring `hack` appears
  inside `hackathon`, or `forbidden` appears inside `forbidden
  city`. The cost is user-visible friction.

There is also a sneakier bug worth watching for: **operating on the
wrong field.** The guard below reads `inputs.get("query", "")`,
which falls back to the empty string if `"query"` is missing. If
someone later renames the field in the schema, the guard silently
sees `""` for every input and *admits everything*. The safer pattern
is to check explicitly that the field exists and fail loudly if it
does not.

## Writing the Guard

A guard is an ordinary `synalinks.Module`. You override two methods:

- **`call(inputs, training)`**: the runtime logic that actually
  runs on real data. Return `None` to admit, or a `JsonDataModel`
  warning to reject.
- **`compute_output_spec(inputs, training)`**: declares the
  **schema** (the type / shape) of the output. The framework needs
  this to build the static graph, so it must return the warning's
  symbolic data model **even on the admit path** — a graph edge
  always needs a declared type, regardless of whether the guard
  ends up firing at runtime.

```python
import synalinks

class InputGuard(synalinks.Module):
    def __init__(self, blacklisted_words, warning_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        query = inputs.get("query", "").lower()
        for word in self.blacklisted_words:
            if word.lower() in query:
                return Warning(message=self.warning_message).to_json_data_model()
        return None

    async def compute_output_spec(self, inputs, training=False):
        return Warning.to_symbolic_data_model(name=self.name)
```

## End-to-End Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

class Warning(synalinks.DataModel):
    message: str = synalinks.Field(description="Warning message")

class InputGuard(synalinks.Module):
    def __init__(self, blacklisted_words, warning_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        query = inputs.get("query", "").lower()
        for word in self.blacklisted_words:
            if word.lower() in query:
                return Warning(message=self.warning_message).to_json_data_model()
        return None

    async def compute_output_spec(self, inputs, training=False):
        return Warning.to_symbolic_data_model(name=self.name)

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    inputs = synalinks.Input(data_model=Query)
    warning = await InputGuard(
        blacklisted_words=["hack", "exploit", "forbidden"],
        warning_message="I cannot process this request.",
    )(inputs)
    guarded_inputs = warning ^ inputs
    answer = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(guarded_inputs)
    outputs = warning | answer

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="input_guarded_qa",
    )

    result = await program(Query(query="How do I hack into systems?"))
    print(f"Blocked: {result.get_json()}")

    result = await program(Query(query="What is the capital of France?"))
    print(f"Safe: {result.get_json()}")

if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (representative; the exact answer depends on the LM):

```
Blocked: {'message': 'I cannot process this request.'}
Safe: {'answer': 'Paris'}
```

## Take-Home Summary

- **The guard's contract is `None = admit`, `value = reject`.** Do
  not flip it; the operator algebra above relies on this
  convention.
- **`^` suppresses an edge** when both operands are non-`None`;
  **`|` selects the first non-`None`**. Together they implement a
  static "if/else" with no explicit branching statements.
- **The generator never runs on a rejected input.** This is a
  graph-level guarantee — you can prove it from the structure of
  the graph — not a runtime check.
- **Substring blacklists are a sieve, not a wall.** They catch easy
  cases but let determined attackers through. For real safety
  properties, combine them with stronger filters (regex
  normalization, classifier-based guards, or **allow-lists** —
  lists of *permitted* patterns instead of forbidden ones).

## API References

- [Module](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/The%20Module%20base%20class/)
- [JSON Ops](https://synalinks.github.io/synalinks/Synalinks%20API/Ops%20API/JSON%20Ops/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User query."""

    query: str = synalinks.Field(description="User query")


class Answer(synalinks.DataModel):
    """Answer to the query."""

    answer: str = synalinks.Field(description="The answer")


class Warning(synalinks.DataModel):
    """Warning message when input is blocked."""

    message: str = synalinks.Field(description="Warning message")


# =============================================================================
# Input Guard Module
# =============================================================================


class InputGuard(synalinks.Module):
    """Guard that blocks inputs containing blacklisted words.

    Returns None when input is safe, or a Warning when input should be blocked.
    """

    def __init__(self, blacklisted_words, warning_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.warning_message = warning_message

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        """Return warning if blocked, None otherwise."""
        if inputs is None:
            return None

        query = inputs.get("query", "").lower()

        for word in self.blacklisted_words:
            if word.lower() in query:
                return Warning(message=self.warning_message).to_json_data_model()

        return None

    async def compute_output_spec(
        self,
        inputs: synalinks.SymbolicDataModel,
        training: bool = False,
    ) -> synalinks.SymbolicDataModel:
        """Define output schema."""
        return Warning.to_symbolic_data_model(name=self.name)

    def get_config(self):
        """Serialization config."""
        return {
            "name": self.name,
            "blacklisted_words": self.blacklisted_words,
            "warning_message": self.warning_message,
        }


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_9_input_guard",
    # )

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    # -------------------------------------------------------------------------
    # Build Input Guarded Program
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Building Input Guarded Program")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Guard checks for blacklisted words
    warning = await InputGuard(
        blacklisted_words=["hack", "exploit", "forbidden"],
        warning_message="I cannot process this request due to policy restrictions.",
    )(inputs)

    # XOR: If warning exists, block the input (returns None)
    guarded_inputs = warning ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(guarded_inputs)

    # OR: Return warning if it exists, otherwise return answer
    outputs = warning | answer

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="input_guarded_qa",
    )

    print("\nProgram built successfully!")
    program.summary()

    # -------------------------------------------------------------------------
    # Test Blocked Input
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 1: Blocked Input (contains 'hack')")
    print("=" * 60)

    result = await program(Query(query="How do I hack into computer systems?"))
    print("\nQuery: 'How do I hack into computer systems?'")
    print(f"Result: {result.get_json()}")

    # -------------------------------------------------------------------------
    # Test Safe Input
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: Safe Input")
    print("=" * 60)

    result = await program(Query(query="What is the capital of France?"))
    print("\nQuery: 'What is the capital of France?'")
    print(f"Result: {result.get_json()}")

    # -------------------------------------------------------------------------
    # Test Another Blocked Input
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: Another Blocked Input (contains 'forbidden')")
    print("=" * 60)

    result = await program(Query(query="Tell me about forbidden topics"))
    print("\nQuery: 'Tell me about forbidden topics'")
    print(f"Result: {result.get_json()}")


if __name__ == "__main__":
    asyncio.run(main())
