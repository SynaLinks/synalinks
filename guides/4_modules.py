"""
# Modules

In [Guide 3](https://synalinks.github.io/synalinks/guides/Programs/) we treated a `Program` as a flowchart made of `Module`s. So
far the only module you have seen up close is `Generator`. This guide
introduces the rest of the catalogue — the bricks you snap together to
build interesting programs.

The mental picture is, once again, **Lego**. Every module is a single
brick. It takes one structured piece of data in, produces one
structured piece of data out, and snaps together with other bricks. A
program is what you get when you click many bricks together.

A slightly more formal way to say it: a module is an asynchronous
function `f: DataModel -> DataModel`. **Asynchronous** just means the
function is defined with `async def`, so it can wait for slow
operations like an LM call without freezing the rest of the program. A
`DataModel` (you met it in [Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/)) is a Pydantic-based class that
pins down which fields exist and what type each one is.

Some modules carry **trainable state** — JSON objects that the
optimizer is allowed to rewrite during training. Each trainable
variable obeys a fixed schema (a subclass of `synalinks.Trainable`
— [Guide 12](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) shows you how to write your own). The two most common
shapes for that JSON object are:

- `instructions` — a variable whose primary field is the system
  prompt the module sends to the LM, and
- `examples` — a variable whose primary field is a list of
  few-shot examples demonstrating the task.

These are special cases. A trainable variable can in general hold
*any* structured data its schema describes — a persona, a
configuration record, a small knowledge base, anything you can
express as a Pydantic class. The important thing to internalize is
that these variables are **parameters of the module**, not
constants you hard-code. In a neural network, the parameters are
floating-point weights; here, they are JSON objects. The
optimizer's job is to improve them. Treat them accordingly.

A `Program`, recall, is a DAG (a directed acyclic graph — a flowchart
with no cycles) whose nodes are modules and whose arrows carry
DataModels. Synalinks checks types twice: once when you *wire* the
graph (using `SymbolicDataModel`, the schema-only stand-in for a real
value), and again at runtime when a real value flows through (using
ordinary Pydantic validation).

The split between **schema** (the static type — known at construction
time) and **value** (the actual runtime instance) is the central idea.
Everything below follows from it.

## Core Modules

### Input: the entry node

`Input` declares where data enters the graph. It performs no
computation. Its only job is to give the graph a typed entry point, the
same way the parameter list of a function tells you what arguments to
expect.

```python
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

# Declares a typed entry. No work happens here.
inputs = synalinks.Input(data_model=Query)
```

A useful analogy: `Input` plays the same role as the parameter list of
a Python function. `def f(x):` names `x` so the body can refer to it;
`Input(data_model=Query)` names a slot in the graph so everything
downstream can refer to it.

### Generator: calling a language model with a typed output

`Generator` is the only module that talks to a language model
directly. It turns its input DataModel into a prompt, asks the model
to fill in the output DataModel, and returns a validated instance.

Two terms used below, worth pinning down:

- **JSON schema**: a JSON document that describes the shape of *other*
  JSON — which fields exist, what type each one is, which are
  required. Every Synalinks DataModel comes with one automatically.
- **Constrained decoding**: when the LM produces output, it is
  restricted token by token to choices that keep the output valid
  against the schema. Tokens that would break the schema are simply
  not allowed to come out. The result is JSON that parses, every
  time.

```mermaid
graph LR
    A["Input DataModel"] --> B["Generator"]
    B --> C["Prompt construction"]
    C --> D["LM call"]
    D --> E["Constrained decoding"]
    E --> F["Pydantic validation"]
    F --> G["Output DataModel"]
```

```python
import synalinks

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

outputs = await synalinks.Generator(
    data_model=Answer,
    language_model=language_model,
    instructions="Be concise and accurate.",
)(inputs)
```

Three rules to memorize:

- **The output is either valid or you get an exception.** A
  `Generator` never returns a half-built object. Either it matches the
  schema, or it raises. There is no silent truncation.
- **The prompt is built from the schema.** Field names and their
  `description` strings end up in the prompt. Rename a field or
  rewrite a description and you change the model's behavior, even if
  no other code changed.
- **`instructions` is a parameter, not a constant.** A Synalinks
  optimizer can rewrite it during training, in the same way gradient
  descent rewrites a weight in a neural network.

Two common ways this goes wrong: empty or vague `description` strings
leave the model with too little to work with, and very deeply nested
schemas make constrained decoding more likely to fail on weaker models.

### Identity: a placeholder that passes data through

`Identity` returns its input unchanged. It is the mathematical
identity function `f(x) = x`, expressed as a module. You use it to
keep graphs symmetric: when one branch transforms the data and a
parallel branch should leave it alone, `Identity` fills the slot so
you do not need a special case for "no module here."

```python
unchanged = await synalinks.Identity()(inputs)
```

### Tool: turning a Python function into a module

`Tool` wraps an async Python function so an agent can call it. The
wrapper reads the function's signature and docstring to build a JSON
schema, which is the format that providers like OpenAI, Anthropic, and
Gemini expect when you declare a tool the model is allowed to call.

```python
import synalinks

@synalinks.saving.register_synalinks_serializable()
async def search_web(query: str):
    \"\"\"Search the web for information.

    Args:
        query (str): The search query.
    \"\"\"
    return {"results": [...]}

tool = synalinks.Tool(search_web)
```

Two rules to remember; these come from the LM providers
(OpenAI/Anthropic/etc.), not from Synalinks itself:

- **Every parameter must be required.** Tool-calling providers treat
  every declared parameter as required, so Python default values are
  rejected. If a parameter is *conceptually* optional, model that
  explicitly — for example, have callers pass `""` or `None` and
  treat that value as "absent" inside the function.
- **Every parameter needs an entry under `Args:` in the docstring.**
  That text becomes the parameter's `description` in the JSON schema.
  Forgetting one raises a `ValueError` when you wrap the function (so
  you catch it early), not later when an agent tries to call the
  tool.

## Control-Flow Modules

These modules give your program the LM equivalent of `if`, `elif`, and
`switch`. Use them when the *next step* of the flowchart needs to
depend on what the data actually looks like. This guide covers them
one brick at a time; [Guide 5](https://synalinks.github.io/synalinks/guides/Control%20Flow/) shows how to compose them with the
merging operators into routing, fan-out, and fallback patterns.

### Decision: pick one label from a fixed list

`Decision` asks the LM to pick exactly one label from a **closed
set** of choices that *you* supply. ("Closed" means the model is not
allowed to invent a new one.) The output schema is fixed:
`{"choice": <one of labels>}`. Conceptually, `Decision` is the LM
version of an `if`/`elif` chain where the condition is "what kind of
input is this?"

```mermaid
graph LR
    A["Input"] --> B["Decision"]
    B --> C{"Which label?"}
    C -->|"math"| D["math"]
    C -->|"general"| E["general"]
    C -->|"code"| F["code"]
```

```python
decision = await synalinks.Decision(
    question="What type of question is this?",
    labels=["math", "general", "code"],
    language_model=language_model,
)(inputs)
# decision is a DataModel with a single field "choice".
```

Because constrained decoding enforces the label set, the result is
guaranteed to be one of your labels. If you want a *free-form*
category instead, use an ordinary `Generator` whose output schema
contains a string field.

### Branch: classify, then route

`Branch` is the combination of a `Decision` and a *k*-way switch (a
`switch`/`case` statement with *k* possible paths). It returns a
tuple of *k* outputs, one slot per branch. At runtime, only the
branch picked by the classifier actually runs; the other *k* − 1
slots come back as `None`.

```python
(math_output, general_output) = await synalinks.Branch(
    question="Is this a math or general question?",
    labels=["math", "general"],
    branches=[
        synalinks.Generator(
            data_model=Answer,
            language_model=lm,
            instructions="You are a math expert.",
        ),
        synalinks.Generator(
            data_model=Answer,
            language_model=lm,
            instructions="You are a general knowledge expert.",
        ),
    ],
    language_model=lm,
)(inputs)

# Collapse the tuple to a single output via Or.
outputs = math_output | general_output
```

What happens, step by step:

1. The classifier picks an index `i` between 0 and *k* − 1.
2. Branch number `i` runs; the other branches skip and yield `None`.
3. Code downstream must cope with `None` from the skipped branches.
   The usual way to do that is the `Or` operator (`|`) you will meet
   in a moment, which returns the first operand that is not `None`.

### Action: let the LM call one specific tool

`Action` is the bridge between a typed input and a `Tool` call. You
give it a single `Tool` and a `LanguageModel`; at call time, the LM
reads the input DataModel, infers the tool's arguments from it, and
the framework actually runs the tool. The output bundles together
the arguments the LM produced *and* the value the tool returned.

Think of `Action` as a single-tool, single-shot version of a
function-calling agent: there is no loop, no choice between tools —
just "given this input, fill in this tool's arguments and run it."

```python
@synalinks.saving.register_synalinks_serializable()
async def calculate(expression: str):
    \"\"\"Calculate a math expression.

    Args:
        expression (str): A math expression such as '2 + 2'.
    \"\"\"
    return {"result": eval(expression, {"__builtins__": None}, {})}

outputs = await synalinks.Action(
    tool=synalinks.Tool(calculate),
    language_model=language_model,
)(inputs)
```

## Merging Modules

A merging module takes several DataModels and combines them into one.
What separates the four merging modules from each other is how they
react to two specific situations: (1) one of the inputs is missing
(it is `None`), and (2) two inputs both declare a field with the same
name (a **schema collision**).

```mermaid
graph LR
    A["DataModel A"] --> C["Merge module"]
    B["DataModel B"] --> C
    C --> D["Combined DataModel"]
```

The four operators form a small algebra. The table below is a lookup
chart: pick the row for the operator, then the column for which inputs
are missing, and read what happens. "Union fields" means the result
keeps every field from every input; "drop A" means A's fields are left
out of the result.

| Operator | Symbol | A=None | B=None | both present |
|----------|--------|--------|--------|--------------|
| Concat   | `+`    | drop A | drop B | union fields |
| And      | `&`    | None   | None   | union fields |
| Or       | `\\|`  | B      | A      | A            |
| Xor      | `^`    | B      | A      | None         |

### Concat (+)

Combines the fields of every non-`None` input into one DataModel. When
two inputs use the same field name, Synalinks renames the duplicates by
appending a numeric suffix (`answer`, `answer_1`, `answer_2`, ...) so
no information is lost.

```python
merged = await synalinks.Concat()([output_a, output_b])
```

### And (&)

Behaves like `Concat` when every input is present, but if any input is
`None`, the whole result is `None`. Use this when every branch is a
prerequisite for what comes next: missing any one means the next stage
should not run.

```python
merged = await synalinks.And()([output_a, output_b])
```

### Or (|)

Returns the first input that is not `None`. This is the standard way to
collapse a `Branch` back into a single output.

```python
result = await synalinks.Or()([primary, fallback])
```

### Xor (^)

Stands for "exclusive or". Returns `None` when both inputs are present.
This is useful as a guard: if a warning fires (so both the warning and
the data are present), the data path is suppressed.

```python
result = await synalinks.Xor()([warning, data])
```

## Masking Modules

Masking selects a subset of the fields of a DataModel. `InMask` keeps
only the fields you list (a whitelist); `OutMask` removes the fields
you list (a blacklist). In both cases the original DataModel is left
untouched, and a new DataModel with fewer fields is returned.

### InMask

```python
filtered = await synalinks.InMask(mask=["answer"])(full_output)
```

### OutMask

```python
filtered = await synalinks.OutMask(mask=["thinking"])(full_output)
```

Why narrowing fields matters:

- **Smaller prompts.** Generators further down the graph receive less
  text, which means fewer tokens and lower cost.
- **Information hiding.** A scratch field (for example `thinking`) used
  for intermediate work does not need to appear in the final output.
- **Cleaner training.** When the reward function only scores a subset
  of fields, the optimizer gets a clearer signal about what to improve.

## Test-Time Compute Modules

These modules spend *extra* LM work when you run the program, in
exchange for better accuracy — trading speed for quality.
"Test-time" is a term borrowed from machine learning: it means "at the
time the model is being used," as opposed to "training-time," when
weights or prompts are being adjusted.

### ChainOfThought

A **chain of thought** is a sequence of reasoning steps the model
writes out before committing to an answer — the LM equivalent of
"showing your work" on a math problem.

`ChainOfThought` adds a `thinking` field to the output schema, placed
*before* the fields you defined. Constrained decoders emit fields in
the order they appear in the schema, so the model writes the
reasoning first and the answer second. And because LMs generate one
token at a time — each token conditioned on the ones already written
— putting the reasoning *before* the answer means the answer ends up
*conditioned on* the reasoning. That tiny ordering trick is the entire
mechanism behind chain-of-thought prompting.

```python
outputs = await synalinks.ChainOfThought(
    data_model=Answer,
    language_model=language_model,
)(inputs)

print(result['thinking'])
print(result['answer'])
```

You do *not* add `thinking` to your DataModel yourself. The
`ChainOfThought` module inserts it for you. This is the main reason
field order matters in Synalinks — order changes *behavior*, not just
appearance.

### SelfCritique

Generates an answer, then scores it with a reward function, and returns
both the answer and the score. A downstream module can use the score to
decide whether to accept the answer, retry, or escalate to a stronger
model.

```python
outputs = await synalinks.SelfCritique(
    language_model=language_model,
)(inputs)
```

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

# =============================================================================
# Data Models
# =============================================================================

class Query(synalinks.DataModel):
    \"\"\"User question.\"\"\"
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    \"\"\"Simple answer.\"\"\"
    answer: str = synalinks.Field(description="The answer")

class AnswerWithThinking(synalinks.DataModel):
    \"\"\"Answer with reasoning.\"\"\"
    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")

# =============================================================================
# Main
# =============================================================================

async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    # -------------------------------------------------------------------------
    # Generator
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Module 1: Generator")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(inputs=inputs, outputs=outputs)
    result = await program(Query(query="What is Python?"))
    print(f"Generator output: {result['answer'][:100]}...")

    # -------------------------------------------------------------------------
    # Branch
    # -------------------------------------------------------------------------
    print("\\n" + "=" * 60)
    print("Module 2: Branch")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    (math_out, general_out) = await synalinks.Branch(
        question="Is this a math or general question?",
        labels=["math", "general"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                instructions="Show your calculations.",
            ),
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
            ),
        ],
        language_model=lm,
    )(inputs)

    outputs = math_out | general_out

    program = synalinks.Program(inputs=inputs, outputs=outputs)
    result = await program(Query(query="What is 15 * 23?"))
    print(f"Math result: {result['answer']}")

    # -------------------------------------------------------------------------
    # ChainOfThought
    # -------------------------------------------------------------------------
    print("\\n" + "=" * 60)
    print("Module 3: ChainOfThought")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(inputs=inputs, outputs=outputs)
    result = await program(Query(query="If I have 3 apples and give 1 away?"))
    print(f"Thinking: {result['thinking'][:100]}...")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Masking
    # -------------------------------------------------------------------------
    print("\\n" + "=" * 60)
    print("Module 4: Masking")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    full_output = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
    )(inputs)

    masked = await synalinks.InMask(mask=["answer"])(full_output)

    program = synalinks.Program(inputs=inputs, outputs=masked)
    result = await program(Query(query="What is 1+1?"))
    print(f"Masked fields: {list(result.get_json().keys())}")

if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (with `ollama/llama3.2:latest`; LM outputs are
non-deterministic, so exact strings will vary):

```
============================================================
Module 1: Generator
============================================================

Generator output: Python is a high-level, interpreted programming language that is widely used for various purposes su...

============================================================
Module 2: Decision
============================================================

Decision output: calculation
Decision output: factual

============================================================
Module 3: Branch (includes decision-making)
============================================================

Math branch result: 345
General branch result: William Shakespeare

============================================================
Module 4: ChainOfThought
============================================================

Thinking: To find out how many apples you will have left after giving 1 away, we need to subtract 1 from the t...
Answer: 2

============================================================
Module 5: Concat (Merging)
============================================================

Merged fields: ['answer', 'thinking', 'answer_1']

============================================================
Module 6: InMask and OutMask
============================================================

Masked output fields: ['answer']
Answer: 2
```

## Take-Home Summary

- **`Input`** declares the entry type. It does no work.
- **`Generator`** is the only module that calls a language model.
  Its prompt is determined entirely by the input and output
  schemas plus the trainable `instructions` variable (a JSON
  object whose primary field is the system-prompt text).
- **`Decision`** and **`Branch`** give you safe, fixed-size control
  flow. Because the labels are a closed set, the choice is always one
  you named — the model cannot invent a new category at runtime.
- **`Concat`, `And`, `Or`, `Xor`** are four ways to merge DataModels.
  Pick one by deciding how it should treat missing inputs.
- **`InMask`** and **`OutMask`** narrow a schema without changing the
  original DataModel.
- **`ChainOfThought`** uses the order of fields in the schema to make
  the model reason before it answers. In Synalinks, field order is
  part of the behavior.

## API References

- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Decision](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Decision%20module/)
- [Branch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Branch%20module/)
- [ChainOfThought](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/ChainOfThought%20module/)
- [InMask/OutMask](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Masking%20Modules/)
- [Tool](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Tool%20module/)
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
    """Simple answer."""

    answer: str = synalinks.Field(description="The answer")


class AnswerWithThinking(synalinks.DataModel):
    """Answer with reasoning."""

    thinking: str = synalinks.Field(description="Step by step thinking")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_4_modules",
    # )

    lm = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    # -------------------------------------------------------------------------
    # Generator Module
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Module 1: Generator")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="generator_demo",
    )

    result = await program(Query(query="What is Python?"))
    print(f"\nGenerator output: {result['answer'][:100]}...")

    # -------------------------------------------------------------------------
    # Decision Module
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 2: Decision")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    decision = await synalinks.Decision(
        question="What type of query is this?",
        labels=["factual", "opinion", "calculation"],
        language_model=lm,
    )(inputs)

    decision_program = synalinks.Program(
        inputs=inputs,
        outputs=decision,
        name="decision_demo",
    )

    result = await decision_program(Query(query="What is 2+2?"))
    print(f"\nDecision output: {result['choice']}")

    result = await decision_program(Query(query="Is Python a good language?"))
    print(f"Decision output: {result['choice']}")

    # -------------------------------------------------------------------------
    # Branch Module
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 3: Branch (includes decision-making)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Branch combines decision-making with routing
    (math_output, general_output) = await synalinks.Branch(
        question="Is this a math or general question?",
        labels=["math", "general"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                instructions="You are a math expert. Show your calculations.",
            ),
            synalinks.Generator(
                data_model=Answer,
                language_model=lm,
                instructions="You are a general knowledge expert.",
            ),
        ],
        language_model=lm,
    )(inputs)

    # Use OR to combine - only selected branch produces output
    outputs = math_output | general_output

    branch_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="branch_demo",
    )

    result = await branch_program(Query(query="What is 15 * 23?"))
    print(f"\nMath branch result: {result['answer']}")

    result = await branch_program(Query(query="Who wrote Hamlet?"))
    print(f"General branch result: {result['answer']}")

    # -------------------------------------------------------------------------
    # ChainOfThought Module
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 4: ChainOfThought")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    cot_program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="cot_demo",
    )

    result = await cot_program(Query(query="If I have 3 apples and give 1 away?"))
    print(f"\nThinking: {result['thinking'][:100]}...")
    print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Merging Modules
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 5: Concat (Merging)")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    branch_a = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        name="expert_a",
        instructions="You are expert A, brief answers.",
    )(inputs)

    branch_b = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
        name="expert_b",
        instructions="You are expert B, detailed answers.",
    )(inputs)

    merged = await synalinks.Concat()([branch_a, branch_b])

    merge_program = synalinks.Program(
        inputs=inputs,
        outputs=merged,
        name="merge_demo",
    )

    result = await merge_program(Query(query="What is AI?"))
    print(f"\nMerged fields: {list(result.get_json().keys())}")

    # -------------------------------------------------------------------------
    # Masking Modules
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Module 6: InMask and OutMask")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    full_output = await synalinks.Generator(
        data_model=AnswerWithThinking,
        language_model=lm,
    )(inputs)

    masked = await synalinks.InMask(mask=["answer"])(full_output)

    mask_program = synalinks.Program(
        inputs=inputs,
        outputs=masked,
        name="mask_demo",
    )

    result = await mask_program(Query(query="What is 1+1?"))
    print(f"\nMasked output fields: {list(result.get_json().keys())}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
