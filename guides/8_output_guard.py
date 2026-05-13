"""
# Output Guards

[Guide 7](Input%20Guard.md) put a guard at the *entrance* of the program. This guide
puts one at the *exit*. An **output guard** is a sanity check on
what the LM has just produced: if the answer looks fine, the guard
waves it through; if it looks unsafe, the guard swaps in a canned
safe reply. Picture a forum moderator who deletes a rule-breaking
post and replaces it with an explanatory notice.

Stated precisely, an output guard is a deterministic predicate
applied to a language model's structured output that decides whether
to forward the output unchanged or substitute a safe replacement.
Let `g : A -> A | None` be the guard, where `A` is the answer schema
and `A | None` means "either a value of type `A`, or nothing." The
contract is:

- **`g(a) = None`** when `a` is **admissible** (safe — nothing to
  do).
- **`g(a) = a'`** when `a` is **inadmissible** (unsafe), where `a'`
  is a safe substitute that obeys the *same* schema `A`.

Put more plainly: the guard returns `None` to mean "no objection,"
and returns a replacement value to mean "block this and use my reply
instead."

When you compose this guard with the generator `G : X -> A`, the
overall program computes:

    F(x) = g(G(x))  if  g(G(x)) is not None,  else  G(x).

In words: run the generator on the input `x` to get a candidate
answer, then run the guard on that answer; if the guard objected
(returned a replacement), use the replacement, otherwise use the
original. Just like [Guide 7](Input%20Guard.md), we get this `if`/`else` *out of the
operator algebra alone*, without ever writing an `if` in the program
graph. ("**Branch-free composition**" just means we express the same
logic with operators on values, so the graph stays a straight
pipeline.)

## Why Filter Outputs as Well as Inputs?

Input guards reject malformed or disallowed prompts *before* the
model runs. Their characteristic failure mode is a **False Accept**:
a bad prompt slips through and reaches the model.

Output guards reject the model's response *after the fact*. Their
characteristic failure mode is a **False Forward**: a bad answer is
forwarded to the user. ("Accept" happens at the input gate,
"Forward" at the output gate.)

Because the language model is stochastic and may degrade in
unexpected ways, the two kinds of guard are **complementary** and
should be layered. Neither one subsumes the other: a clean prompt
can still produce a bad answer, and a suspicious prompt can still
produce a fine one.

```mermaid
graph LR
    subgraph S1["Unguarded path"]
        A1[Input] --> B1[LM] --> C1[Output: possibly unsafe]
    end
    subgraph S2["Guarded path"]
        A2[Input] --> B2[LM] --> C2[Candidate output]
        C2 --> G2{Guard predicate}
        G2 -->|"admissible"| H2[Forward candidate]
        G2 -->|"inadmissible"| I2[Replacement]
    end
```

## Encoding "if guard objected, swap" as dataflow

The idea is to express "if the guard objected, use the replacement,
otherwise use the original" with two operators that work on
**optional** values (values that may be `None`). Synalinks overloads
`^` (XOR) and `|` (OR) on DataModel-valued nodes. With `None`
denoting a missing value, they obey these rules:

- `None ^ x   = x`   and `None | x   = x`
- `x ^ None   = x`   and `x | None   = x`
- `x ^ y      = None` and `x | y      = x` (when both are present)

Two slogans to remember:

- **XOR (`^`) means "exactly one is present."** If one operand is
  `None`, the other one survives. If both are present, they cancel
  out to `None`.
- **OR (`|`) means "the first one present wins."** If both are
  present, the left operand is kept.

(Note: these are *not* the bitwise XOR/OR you may know from `int`s.
Here the operands are optional DataModels, not bits.)

The guarded pipeline is then a three-line composition:

```
warning      = OutputGuard(answer)        # warning : A | None
safe_answer  = warning ^ answer           # None if warning present, else answer
outputs      = warning | safe_answer      # warning if present, else answer
```

There are only two cases to check; let us walk through both.

- **Admissible case** (`warning = None`): the guard had no
  objection. Then `safe = None ^ answer = answer`, and
  `out = None | answer = answer`. The original answer passes
  through untouched.
- **Inadmissible case** (`warning = a'`): the guard returned a
  replacement `a'`. Then `safe = a' ^ answer = None` (both present,
  so XOR cancels), and `out = a' | None = a'`. The replacement
  wins.

**Invariant after the guard.** (Recall: an *invariant* is a property
that always holds, no matter which branch ran.) The final output
`out` is well-typed in `A` (it satisfies the answer schema) and is
either the language model's original answer or a guard-issued
replacement. It is never `None`, provided `G` itself produced a
value.

```mermaid
graph LR
    A[inputs] --> B[Generator]
    B --> C[answer]
    C --> D[OutputGuard]
    D --> E[warning]
    E --> F["xor: warning ^ answer"]
    C --> F
    F --> G[safe_answer]
    E --> H["or: warning | safe_answer"]
    G --> H
    H --> I[output]
```

## Building a Guard Module: the minimal contract

Concretely, a guard is a small Synalinks `Module` that receives the
generator's answer and decides what to do with it. It has two
responsibilities — one at runtime, one at build time:

1. **`call(inputs)`** runs on real data. Decide admissibility.
   Return `None` for admissible (safe) inputs; return a DataModel
   of the **same schema** as `inputs` for inadmissible (unsafe)
   ones. This is the per-example decision.
2. **`compute_output_spec(inputs)`** runs at build time. Declare
   the static schema of the node. Synalinks uses this declaration
   to type-check the graph *before* any data flows through — it is
   how you tell the framework "here is the shape of what I will
   produce."

**A common trap.** Returning a DataModel with a *different* schema
breaks the XOR/OR composition: both operands of `^` and `|` must
share a schema for the operators to type-check. Always return
`Answer(...)` (or whatever the input schema was), not a separate
`Warning` type.

```python
import synalinks

class OutputGuard(synalinks.Module):
    \"\"\"Substitutes blacklisted outputs with a fixed safe message.\"\"\"

    def __init__(self, blacklisted_words, replacement_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.replacement_message = replacement_message

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        answer = inputs.get("answer", "").lower()
        for word in self.blacklisted_words:
            if word.lower() in answer:
                # Schema-preserving replacement.
                return Answer(answer=self.replacement_message).to_json_data_model()
        return None

    async def compute_output_spec(self, inputs, training=False):
        # Declare the static type for the functional graph.
        return Answer.to_symbolic_data_model(name=self.name)
```

## Complete example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

class OutputGuard(synalinks.Module):
    def __init__(self, blacklisted_words, replacement_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.replacement_message = replacement_message

    async def call(self, inputs, training=False):
        if inputs is None:
            return None
        answer = inputs.get("answer", "").lower()
        for word in self.blacklisted_words:
            if word.lower() in answer:
                return Answer(answer=self.replacement_message).to_json_data_model()
        return None

    async def compute_output_spec(self, inputs, training=False):
        return Answer.to_symbolic_data_model(name=self.name)

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    inputs = synalinks.Input(data_model=Query)
    answer = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)
    warning = await OutputGuard(
        blacklisted_words=["harmful", "dangerous", "illegal"],
        replacement_message="I cannot provide information on that topic.",
    )(answer)
    safe_answer = warning ^ answer
    outputs = warning | safe_answer

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="output_guarded_qa",
    )
    result = await program(Query(query="What is the capital of France?"))
    print(result.get_json())

if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (abbreviated; with `llama3.2:latest`, on safe queries the
guard returns `None` and the original answer passes through):

```
{'answer': 'Paris'}
```

## Properties That Always Hold, and Failure Modes

The composition `warning | (warning ^ answer)` enforces three
**invariants** — properties guaranteed to hold for every input, in
both branches:

- **Totality.** If `answer` is not `None`, the output is not
  `None`. The guard never accidentally drops the response.
- **Schema closure.** The output is always in `A`, the answer
  schema. No surprise types reach downstream code.
- **Priority.** When the guard triggers, the replacement strictly
  supersedes the original. A "refusal" — the canned safe reply —
  always wins over the model's unsafe answer.

Failure modes worth anticipating — places where a real-world guard
can still let you down:

- **Lexical evasion.** Substring-matching blacklists are trivially
  defeated by paraphrase ("hazardous" instead of "dangerous"),
  transliteration ("d4ngerous"), or encoding (base64, ROT13). Treat
  substring filters as a baseline, not a real defense. For robust
  filtering, use a learned classifier or an LM judge inside the
  guard's `call`.
- **Schema drift.** Returning a replacement of a different schema
  breaks the XOR/OR types. Always preserve the input schema.
- **Over-blocking.** A too-aggressive blacklist replaces benign
  outputs (false alarms). Calibrate against held-out data,
  measuring both false-accept and false-forward rates on
  representative examples.

## Take-Home Summary

- An **output guard** is a post-generation filter — the mirror
  image of the input guards from [Guide 7](Input%20Guard.md). Together they bracket
  the LM call on both sides.
- **Contract:** the guard returns `None` to *admit* (no
  objection) and a replacement value to *block*. The
  replacement **must obey the same schema** as the original
  output, or the operator algebra below will not type-check.
- The composition **`warning | (warning ^ answer)`** expresses
  "if the guard objected, use the replacement, otherwise use
  the answer" as pure dataflow — no `if` statements in the
  program graph.
- The composition guarantees three invariants: **totality**
  (output is never dropped), **schema closure** (output is
  always in `A`), and **priority** (when the guard fires, the
  refusal wins).
- Substring blacklists are a **sieve, not a wall.** For real
  safety properties, layer them with classifiers or LM judges
  inside the guard's `call`.

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


# =============================================================================
# Output Guard Module
# =============================================================================


class OutputGuard(synalinks.Module):
    """Guard that replaces outputs containing blacklisted words.

    Returns None when output is safe, or a replacement Answer when output
    should be filtered.
    """

    def __init__(self, blacklisted_words, replacement_message, **kwargs):
        super().__init__(**kwargs)
        self.blacklisted_words = blacklisted_words
        self.replacement_message = replacement_message

    async def call(
        self,
        inputs: synalinks.JsonDataModel,
        training: bool = False,
    ) -> synalinks.JsonDataModel:
        """Return replacement if output should be filtered, None otherwise."""
        if inputs is None:
            return None

        answer = inputs.get("answer", "").lower()

        for word in self.blacklisted_words:
            if word.lower() in answer:
                return Answer(answer=self.replacement_message).to_json_data_model()

        return None

    async def compute_output_spec(
        self,
        inputs: synalinks.SymbolicDataModel,
        training: bool = False,
    ) -> synalinks.SymbolicDataModel:
        """Define output schema (same type as Answer for replacement)."""
        return Answer.to_symbolic_data_model(name=self.name)

    def get_config(self):
        """Serialization config."""
        return {
            "name": self.name,
            "blacklisted_words": self.blacklisted_words,
            "replacement_message": self.replacement_message,
        }


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_10_output_guard",
    # )

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    # -------------------------------------------------------------------------
    # Build Output Guarded Program
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Building Output Guarded Program")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Generate answer first
    answer = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    # Guard checks for blacklisted words in the output
    warning = await OutputGuard(
        blacklisted_words=["harmful", "dangerous", "illegal"],
        replacement_message="I cannot provide information on that topic.",
    )(answer)

    # XOR: If warning exists, invalidate the original answer
    safe_answer = warning ^ answer

    # OR: Return warning (replacement) if it exists, otherwise return safe_answer
    outputs = warning | safe_answer

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="output_guarded_qa",
    )

    print("\nProgram built successfully!")
    program.summary()

    # -------------------------------------------------------------------------
    # Test Safe Output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 1: Safe Output (normal question)")
    print("=" * 60)

    result = await program(Query(query="What is the capital of France?"))
    print("\nQuery: 'What is the capital of France?'")
    print(f"Result: {result.get_json()}")

    # -------------------------------------------------------------------------
    # Test Another Safe Output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: Another Safe Output")
    print("=" * 60)

    result = await program(Query(query="Explain what Python is used for"))
    print("\nQuery: 'Explain what Python is used for'")
    print(f"Result: {result.get_json()}")

    # -------------------------------------------------------------------------
    # Demonstrate the Pattern
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Output Guard Pattern Explanation")
    print("=" * 60)

    print("""
The output guard pattern works as follows:

1. INPUT arrives
2. GENERATOR produces an answer
3. OUTPUT GUARD checks the answer for blacklisted content
4. If blacklisted content found:
   - Guard returns a replacement answer
   - XOR: replacement ^ original = None (invalidates original)
   - OR: replacement | None = replacement (use replacement)
5. If no blacklisted content:
   - Guard returns None
   - XOR: None ^ original = original (keeps original)
   - OR: None | original = original (use original)

This pattern ensures unsafe outputs are replaced seamlessly!
""")


if __name__ == "__main__":
    asyncio.run(main())
