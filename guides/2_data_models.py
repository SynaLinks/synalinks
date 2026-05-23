"""
# Data Models

In [Guide 1](https://synalinks.github.io/synalinks/guides/Getting%20Started/) you met the three core ingredients: `DataModel`, `Generator`,
and `Program`. This guide zooms in on the first of them. Data models are
the language you use to describe what your LM should produce — and they
are the reason the rest of the framework can give you typed objects
instead of strings to parse.

Up to now, most of the Python code you have written passes plain values
around: strings, integers, lists, dictionaries. The moment you start
calling a language model the *natural* output is a blob of text, and the
moment after that you start writing fragile code: a `split` here, a regex
there, an `if "yes" in answer.lower()` over there. Each fragment looks
reasonable. Together they form a system where one weird response breaks
everything downstream.

A data model replaces that pattern. The idea is the same one behind a
Python `dataclass`: instead of returning a string and trusting the next
function to interpret it, you **declare what shape the answer must take**
and let the framework refuse to give you anything else.

In Synalinks vocabulary, a *data model* is a typed schema that fixes the
shape of any value crossing a module boundary. The word **schema** simply
means "a description of which fields exist and what types they hold" —
the header row of a spreadsheet is a perfectly good mental picture. More
concretely, a data model is a class that inherits from
`synalinks.DataModel` (which is itself a Pydantic class), and its field
annotations get translated into a **JSON Schema** — a standard,
machine-readable description of a JSON object — that the LM is forced to
follow.

**Pydantic**, in case you have never run into it, is a Python library
that turns type-annotated classes into runtime validators. If you have
used the `dataclass` decorator from the standard library, the *feel* is
similar; the difference is that Pydantic actually checks the types when
data flows in and raises a clean error if something is wrong. Synalinks
builds on top of it.

Once you commit to declaring a data model, three things become true at
the same time, and they are worth pausing to appreciate:

1. The prompt sent to the model now carries a machine-readable
   description of the target structure (the schema), not just
   hand-written hints like "please respond in JSON."
2. The model's output is policed — either by **constrained decoding**
   (the model is only allowed to emit tokens that keep the output
   syntactically valid, like a strict spell-checker for shape) or by
   **validate-and-retry** (bad outputs are detected and the call is
   reattempted). Either way, syntactically invalid output never escapes
   the module.
3. Downstream code sees a typed Python object, not a string. Writing
   `result["sentiment"]` is *statically meaningful*: that field will be
   one of the values you declared, or the program will have failed
   loudly earlier — never silently produced garbage in the middle.

This is the same step that took early programmers from `printf`-style
text everywhere to typed records: you give up a little flexibility at
the boundary in exchange for guarantees the rest of the program can
rely on.

## A Tale of Two Pipelines

The clearest way to see what data models replace is to compare the two
pipelines side by side. The first is the one you would write without a
schema; the second is what Synalinks gives you.

```mermaid
graph LR
    A["Text prompt"] --> B["LLM"]
    B --> C["Unstructured text"]
    C --> D["Ad-hoc parser"]
    D --> E["Hope"]
```

```mermaid
graph LR
    A["DataModel input"] --> B["Module"]
    B --> C["LLM + JSON schema"]
    C --> D["Validator"]
    D --> E["DataModel instance"]
```

The second pipeline has one property the first lacks: **every arrow is
typed**. When something goes wrong, the failure happens at the validator
— right next to its cause — instead of three function calls later when
some `dict.get(...)` quietly returns `None` and you spend the evening
chasing the source.

## Declaring a Data Model

A data model is just a Python class. The type annotations you write tell
Synalinks (and the LM) what each field must contain, and
`synalinks.Field(description=...)` attaches a one-line natural-language
note that becomes part of the prompt the model sees. Here is a small
example: a review analyzer that returns a sentiment label, a list of
key points, and a 1-to-10 rating.

```python
import synalinks
from typing import Literal
from enum import Enum

class Rating(int, Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10

class MovieReview(synalinks.DataModel):
    \"\"\"Analysis of a movie review.\"\"\"

    sentiment: Literal['positive', 'negative', 'neutral'] = synalinks.Field(
        description="The overall sentiment: positive, negative, or neutral"
    )
    key_points: list[str] = synalinks.Field(
        description="Main points mentioned in the review"
    )
    rating: Rating = synalinks.Field(
        description="Estimated rating from 1 to 10"
    )
```

There are two kinds of constraints in this class, and they behave very
differently. Think of "hard" as "the program refuses to continue" and
"soft" as "a strong suggestion to the model":

- The **type** (e.g. `Literal[...]`, `Rating`) is a *hard* constraint.
  After the model produces its output, the validator checks the value
  against the type. Anything outside the declared set is rejected
  outright.
- The **description** is a *soft* constraint, conveyed to the model as
  text in the prompt. It shapes what the model tends to produce, but
  nothing checks the description after the fact.

A rule of thumb worth burning in: if a malformed answer should cause the
program to refuse the result, encode the rule in the **type**. If it is
about style, tone, or intent, encode it in the **description**.

### The Field Function

`synalinks.Field` is a very thin wrapper over Pydantic's own `Field`. Its
`description` argument is the slot through which you talk to the model.
The string you put there appears in the prompt the LM reads — verbatim.
It is not metadata that the model politely ignores; it is instructions
the model will follow to the letter. Treat it the way you would treat a
function docstring being read by a co-author who has no other context.
Compare:

```python
# Specific, actionable: the model can act on this.
answer: str = synalinks.Field(
    description="A concise answer in 1-2 sentences, based only on the provided context"
)

# Vague: the model will fill in the gap with priors you did not choose.
answer: str = synalinks.Field(
    description="The answer"
)
```

## Which Types You Can Use

The Python types you can put on a `DataModel` field are exactly the ones
that have a clean counterpart in JSON. JSON itself has only a handful of
value kinds — strings, numbers, booleans, arrays, objects — so the table
below is essentially the list of Python types that survive the round
trip:

| Type              | JSON Schema | Example              |
|-------------------|-------------|----------------------|
| `str`             | string      | `"hello world"`      |
| `int`             | integer     | `42`                 |
| `float`           | number      | `3.14`               |
| `bool`            | boolean     | `true`               |
| `list[T]`         | array       | `["a", "b", "c"]`    |
| `dict`            | object      | `{"key": "value"}`   |
| `Enum`            | enum        | constrained choices  |
| `synalinks.Score` | enum        | 0.0 to 1.0, step 0.1 |

If a Python type maps cleanly to JSON Schema, you can use it. You can
also **nest** one data model inside another: the inner schema gets
referenced from the outer one using a `$ref` link, which works like a
footnote pointing to a definition elsewhere.

One edge case is worth flagging up front. A `dict` *without* a declared
value type accepts any JSON object, which throws away most of the safety
you bought by using a schema in the first place. Whenever you know the
keys in advance, prefer a nested `DataModel`.

### Enums: closed lists of allowed choices

When the answer must come from a small fixed set of options — say one of
`"low"`, `"medium"`, `"high"`, `"critical"` — encode it as an `Enum`. (A
**closed alphabet** is just jargon for "a fixed, finite list of allowed
values.") The schema then tells the model exactly which members exist,
and validation rejects anything outside the list.

```python
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskAnalysis(synalinks.DataModel):
    reasoning: str = synalinks.Field(
        description="Why this priority was assigned"
    )
    priority: Priority = synalinks.Field(
        description="The priority level of this task"
    )
```

The property guaranteed to hold here (the **invariant**, in math
jargon — "a thing that is always true no matter what happens") is this:
any value you read out as `result['priority']` is guaranteed to be a
member of the `Priority` enum. You can write
`if priority == Priority.HIGH:` without a defensive
`else: raise ValueError(...)` clause; the bad case never reaches this
point.

A small trap: the *order* in which you list enum members matters for
documentation but not for validation. If you later add
`BLOCKER = "blocker"`, old saved data that still contains `"critical"`
will validate fine — only *unknown* strings get rejected.

### synalinks.Score: a 0-to-1 scale split into buckets

A very common need is to ask the model for a confidence or quality score
between 0 and 1. `synalinks.Score` is a ready-made enum over the eleven
values `{0.0, 0.1, 0.2, ..., 1.0}`, with named members from `VERY_BAD`
(0.0) through `VERY_GOOD` (1.0). The fancy word for taking a continuous range
and splitting it into a fixed list of buckets is **discretization** —
that is all we have done here.

Why not let the model emit an arbitrary float like `0.8732`? Two
reasons. First, the model has no genuine internal sense in which
`0.8732` differs from `0.87`; the extra digits give *false precision*.
Second, ten buckets is roughly the resolution at which models can give
meaningfully different answers anyway, so the discretization throws
away no real information.

```python
class Analysis(synalinks.DataModel):
    result: str = synalinks.Field(description="The analysis result")
    confidence: synalinks.Score = synalinks.Field(
        description="Confidence in the result (0.0 = uncertain, 1.0 = certain)"
    )
```

Use `Score` for confidence, quality, similarity — anything that lives on
a normalized scale. Do *not* use it when you genuinely need continuous
values (for example, the target of a regression task); for that, declare
the field as `float` and validate the range yourself.

## Combining Data Models with Operators

Sooner or later you will want to combine two data models — for example,
gluing the output of one module onto the output of another, or stripping
some fields before passing data along. Synalinks defines a small set of
operators on data models for exactly this purpose, so you do not have to
write a throwaway adapter class every time.

The word **algebra** here just means "a few operators with predictable
rules" — the same way `+` and `*` are an algebra on numbers. Each
operator below takes two (or one) data models and returns a new data
model whose schema is built from the inputs.

```mermaid
graph TD
    A["DataModel A"] --> C["Operator (+, &, |, ^)"]
    B["DataModel B"] --> C
    C --> D["Composed DataModel"]
```

| Operator | Name   | Semantics                                                 |
|----------|--------|-----------------------------------------------------------|
| `+`      | Concat | Union of fields from both operands.                       |
| `&`      | And    | Like `+`, but the result is `None` if either is `None`.   |
| `\\|`    | Or     | First non-`None` value per field across the two operands. |
| `^`      | Xor    | Field-level exclusive-or: drop fields present in both.    |
| `~`      | Not    | Logical negation on boolean-valued fields.                |

A useful mental picture: think of each data model as a dictionary from
field name to value. The operators above are then set-like operations on
those dictionaries — combining, intersecting, taking differences.

### Stitching two branches' outputs together

```python
result1 = Analysis1(summary="First analysis", score=0.8)
result2 = Analysis2(details="Additional details", tags=["a", "b"])

# Compose into a single instance with all four fields.
combined = result1 + result2
# combined.get_json() ->
#   {"summary": "First analysis", "score": 0.8,
#    "details": "Additional details", "tags": ["a", "b"]}
```

A small invariant worth knowing: if the two operands have *no* field
names in common, then `a + b` and `b + a` produce the same result. (The
math word for this property is **commutative**.) If they *do* share a
field name, the operand on the right wins — so order does matter when
fields overlap, and you should be deliberate about which goes first.

## Masking: keeping (or dropping) a subset of fields

Sometimes you want only *part* of a data model's fields downstream — for
example, you want to hide the model's internal reasoning before you show
the answer to a user. The operation for picking a subset of fields from
a record is called **projection** (the same term you might have seen in
a database course, where it means selecting only some columns of a
table). Masking is how you do this without writing a brand-new
intermediate schema by hand.

### in_mask: Project Onto a Field Subset

```python
# Keep only 'answer' and 'confidence'; drop everything else.
filtered = await synalinks.ops.in_mask(
    full_result,
    mask=["answer", "confidence"]
)
```

### out_mask: Project Away a Field Subset

```python
# Drop 'thinking'; keep everything else.
filtered = await synalinks.ops.out_mask(
    full_result,
    mask=["thinking"]
)
```

The two operations are mirror images of each other: keeping a set of
fields with `in_mask` is the same as dropping the complementary set with
`out_mask`. Typical uses:

- Hiding scratchpad or chain-of-thought fields (the model's
  "thinking-out-loud" notes) from the final response shown to a user.
- Restricting training so that the loss function only looks at the
  fields you actually care about.
- Adapting between two modules whose schemas overlap but are not
  identical.

## Using Plain Pydantic Models Instead

`synalinks.DataModel` is a thin extension of `pydantic.BaseModel`. The
things it adds are the operator algebra (`+`, `&`, `|`, `^`, `~`), the
masking helpers, and two convenience methods,
`.to_json_data_model()` and `.to_symbolic_data_model()`. Everything else
is plain Pydantic.

So if you already have Pydantic models in your project — say, ones used
by your web API — and you do not want to declare them twice, you can
hand the schemas directly to Synalinks. Any module that accepts a
`data_model=` argument also accepts a `schema=` argument: a JSON Schema
dictionary, which any Pydantic class can produce via its
`.model_json_schema()` method.

```python
from pydantic import BaseModel, Field
import synalinks

class Query(BaseModel):
    query: str = Field(description="The user query")

class Answer(BaseModel):
    answer: str = Field(description="A concise answer")

# Build the program from schemas instead of DataModels.
inputs = synalinks.Input(schema=Query.model_json_schema())
outputs = await synalinks.Generator(
    schema=Answer.model_json_schema(),
    language_model=language_model,
)(inputs)

program = synalinks.Program(inputs=inputs, outputs=outputs)
```

At runtime, the program expects its input to be a Synalinks data object,
not a raw Pydantic instance. Wrap the Pydantic instance in a
`JsonDataModel`, which is just a tiny container pairing a raw JSON
dictionary with its schema:

```python
query = Query(query="What is the capital of France?")

payload = synalinks.JsonDataModel(
    schema=Query.model_json_schema(),
    json=query.model_dump(),
)

result = await program(payload)
```

### What you give up by going pure-Pydantic

Sticking with `pydantic.BaseModel` keeps your domain types independent
of Synalinks, but you lose a few conveniences:

- **No operators.** `+`, `&`, `|`, `^`, `~` are defined on
  `synalinks.DataModel`, not on `pydantic.BaseModel`.
- **No conversion helpers.** Instead of `.to_json_data_model()` and
  `.to_symbolic_data_model()` you call `.model_json_schema()` and
  `.model_dump()` yourself.
- `synalinks.Score` and other enums still work — they are ordinary
  Python enums and do not depend on the `DataModel` base class.

If you later decide you want the operator algebra or masking on a
shape you defined in plain Pydantic, you have two ways out: either
re-declare the fields under `synalinks.DataModel`, or wrap the instance
in a `JsonDataModel` and use `synalinks.ops.in_mask` /
`synalinks.ops.out_mask` directly.

## Putting It All Together

The example below stitches the ideas of this guide into one runnable
program. A `ReviewInput` flows into a `Generator` that returns a
`ReviewAnalysis`. The `Generator` is `await`ed because under the hood it
talks to the LM over the network — exactly the same `async`/`await`
pattern from [Guide 1](https://synalinks.github.io/synalinks/guides/Getting%20Started/).

```python
import asyncio
from enum import Enum
from dotenv import load_dotenv
import synalinks

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ReviewInput(synalinks.DataModel):
    \"\"\"A product review to analyze.\"\"\"
    review_text: str = synalinks.Field(
        description="The text of the product review"
    )

class ReviewAnalysis(synalinks.DataModel):
    \"\"\"Structured analysis of a review.\"\"\"
    sentiment: Sentiment = synalinks.Field(
        description="The overall sentiment of the review"
    )
    confidence: synalinks.Score = synalinks.Field(
        description="Confidence in the sentiment classification"
    )
    key_points: list[str] = synalinks.Field(
        description="Main points mentioned by the reviewer"
    )
    recommended: bool = synalinks.Field(
        description="Whether the reviewer would recommend the product"
    )

async def main():
    load_dotenv()
    synalinks.clear_session()

    language_model = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    inputs = synalinks.Input(data_model=ReviewInput)
    outputs = await synalinks.Generator(
        data_model=ReviewAnalysis,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="review_analyzer",
    )

    result = await program(
        ReviewInput(
            review_text="This laptop is amazing! Fast processor, great screen, "
            "and the battery lasts all day. Only complaint is it runs a bit warm. "
            "Would definitely buy again!"
        )
    )

    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Key Points: {result['key_points']}")
    print(f"Recommended: {result['recommended']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (with `ollama/llama3.2:latest`; exact wording is
nondeterministic across runs):

```
Sentiment: positive
Confidence: 0.9
Key Points: ['Fast processor', 'Great screen', 'Good battery life']
Recommended: True
```

## Take-Home Summary

If you remember nothing else from this guide, remember the following:

- A data model is **three things at once**: a Python type, a JSON
  Schema, and a fragment of the prompt the model reads. Change any one
  of them and the other two change with it.
- **Types are enforced; descriptions are not.** Anything that should
  cause a bad answer to be *rejected* belongs in the type. Anything
  about style or intent belongs in the description.
- **Enums close the output to a finite list of choices.** Use them
  whenever the answer should come from a small, fixed alphabet.
- **`synalinks.Score` splits `[0, 1]` into ten meaningful buckets.**
  Use it for confidence/quality/similarity; reach for plain `float`
  only when you genuinely need continuous values.
- The **operators** (`+`, `&`, `|`, `^`, `~`) compose schemas without
  hand-written adapter classes, and the **masking helpers** (`in_mask`,
  `out_mask`) project onto subsets of fields.
- **Plain Pydantic also works** via `schema=` and `JsonDataModel`, at
  the cost of giving up the operator algebra.

## API References

- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [JsonDataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20JsonDataModel%20class/)
- [Base DataModels](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
- [JSON Ops](https://synalinks.github.io/synalinks/Synalinks%20API/Ops%20API/JSON%20Ops/)
"""

import asyncio
from enum import Enum

from dotenv import load_dotenv

import synalinks


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewInput(synalinks.DataModel):
    """A product review to analyze."""

    review_text: str = synalinks.Field(description="The text of the product review")


class ReviewAnalysis(synalinks.DataModel):
    """Structured analysis of a review."""

    sentiment: Sentiment = synalinks.Field(
        description="The overall sentiment of the review"
    )
    confidence: synalinks.Score = synalinks.Field(
        description="Confidence in the sentiment classification"
    )
    key_points: list[str] = synalinks.Field(
        description="Main points mentioned by the reviewer"
    )
    recommended: bool = synalinks.Field(
        description="Whether the reviewer would recommend the product"
    )


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_2_data_models",
    # )

    language_model = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    inputs = synalinks.Input(data_model=ReviewInput)
    outputs = await synalinks.Generator(
        data_model=ReviewAnalysis,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="review_analyzer",
    )

    result = await program(
        ReviewInput(
            review_text="This laptop is amazing! Fast processor, great screen, "
            "and the battery lasts all day. Only complaint is it runs a bit warm. "
            "Would definitely buy again!"
        )
    )

    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Key Points: {result['key_points']}")
    print(f"Recommended: {result['recommended']}")


if __name__ == "__main__":
    asyncio.run(main())
