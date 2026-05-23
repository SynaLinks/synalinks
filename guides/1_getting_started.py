"""
# Getting Started with Synalinks

Welcome. This is the first of seventeen guides. By the end of this one you will
have written a tiny program that asks a language model a question and gets
back a Python object you can use directly — no string parsing, no fragile
regex, no `if "yes" in answer.lower()` hacks. We will cover just three
ingredients: a `DataModel`, a `Generator`, and a `Program`. Everything else
in the framework is built on top of these three.

**What you should already know.** Python basics: classes, functions, type
hints like `name: str`, and roughly what `async`/`await` does. If you have
written a `class` with attributes and called a function before, you are
ready. We will explain every Synalinks-specific term the first time it
appears.

## The Problem We Are Solving

A language model (LM) is, at its simplest, a very elaborate autocomplete.
You give it some text — a "prompt" — and it gives you back more text — a
"completion." That works beautifully for a chat window, where a human reads
the reply and decides what to do with it.

It works much less beautifully when the LM is not the whole product but a
*part* of a larger program. Imagine you are writing a tutor app: the user
types a math problem, an LM works out the answer, and the rest of your code
needs to know two things — was the answer correct, and how should we score
it? If the LM hands you back the string

```
Hmm, let me think... I believe the answer is 42, but it could also be 41.
```

you now have to write code that finds the number, decides which number
matters, and falls back gracefully when the model says "forty-two" in words
instead. Multiply this by every place your code touches the LM and you
spend more time parsing strings than building the app.

What we actually want is to treat an LM call like any other typed function:
something specific goes in, something specific comes out, and the type
system tells us what is in each. That requires three pieces the raw LM API
does not give us:

1. A **typed interface**. The call should consume and produce structured
   values — Python objects with named fields — rather than free text.
2. A way to **declare what we want**. Instead of hand-crafting a prompt
   that begs the model to "please respond in JSON," we should describe the
   shape of the answer once and let the framework handle the rest.
3. A way to **compose** several such calls (and ordinary Python code) into
   one bigger object you can call, save, load, and improve over time. If
   you have ever used Keras to stack neural-network layers into a single
   `Model`, this is the same idea applied to LM calls.

Synalinks provides these three pieces under the names `DataModel`,
`Generator`, and `Program`. We will meet each one in turn.

```mermaid
flowchart LR
    A["Untyped prompt string"] --> B["LM call"]
    B --> C["Untyped completion string"]
    D["Typed DataModel (input)"] --> E["Generator"]
    E --> F["Typed DataModel (output)"]
```

The top row is the raw experience. The bottom row is what Synalinks adds.
You describe the shape of the answer you want (this description is called
a **schema** — think of it like the header row of a spreadsheet, listing
which columns exist and what type each column holds), and the framework:

- builds an appropriate prompt for you,
- runs the LM in a mode that refuses to produce output of the wrong shape
  (this is called **constrained decoding** — picture a strict proofreader
  watching every word and crossing out anything that would break the
  format),
- parses the result back into a Python object you can use directly.

If something still goes wrong — say the LM produces gibberish — Synalinks
retries; if retries fail it raises a clean exception instead of silently
returning broken data. The promise to remember: **a successful call gives
you back a value that matches the shape you declared.** You will not be
writing `try: json.loads(...)` glue code in your application logic.

## Installation

Install the library the same way you would install any Python package:

```bash
pip install synalinks    # or, if you use uv: uv pip install synalinks
```

## Pointing the Code at a Language Model

Synalinks does not ship with its own LM — it talks to whichever one you
already have. For this guide we use a local copy of Llama via Ollama, which
runs on your laptop and needs no account or API key:

```bash
ollama serve && ollama pull llama3.2:latest
```

If you would rather use a hosted model (Gemini, Claude, GPT, etc.), put the
corresponding API key in a `.env` file in your project folder and change
one string in the code (the `model="..."` argument). Everything else stays
the same.

```bash
# Example .env entries for hosted providers:
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Ingredient 1: `DataModel` — describing what data looks like

A `DataModel` is a Python class that describes the *shape* of a piece of
data: what fields it has and what type each field holds. The closest
everyday analogy is a paper form with labeled blanks — "Name: ____",
"Age: ____" — except that here the blanks come with type rules ("Age must
be a whole number").

Under the hood, a `DataModel` is a Pydantic model. **Pydantic** is a widely
used Python library that turns type-annotated classes into runtime data
validators; if you have used the standard `dataclass` decorator, the feel
is similar, but Pydantic actually checks the types at runtime and raises
when something does not match. Synalinks adds two things on top:

1. Every field carries a short natural-language `description`. This
   description is given to the LM as part of the prompt.
2. The class can be exported as a **JSON Schema** — a standard,
   machine-readable description of what a JSON object should look like.
   That schema is what the LM is constrained to follow.

Here are two small `DataModel`s, one for the question we will send in and
one for the answer we expect back:

```python
import synalinks

class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="The question to answer")

class Answer(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Step-by-step reasoning")
    answer: str = synalinks.Field(description="The final answer")
```

Two facts about this code that are easy to underestimate:

- **The `description` is not a code comment.** The LM literally reads it.
  Synalinks weaves each description into the prompt, so this is your main
  lever for telling the model what should go in each field. Vague
  descriptions produce vague outputs.
- **Field order matters.** An LM writes its answer one token at a time,
  left to right. Whatever field appears first in your class gets filled
  in first. By putting `thinking` before `answer`, we force the model to
  reason out loud *before* committing to a final answer. This trick is
  called **chain-of-thought**, and it noticeably improves accuracy on
  multi-step problems. If you reversed the order, the model would commit
  to an answer first and then rationalize it — losing most of the
  benefit. A common beginner trap.

## Ingredient 2: `Generator` — one LM call, typed at both ends

A `Generator` is the smallest reusable piece that actually talks to a
language model. The mental model is simple: a `DataModel` goes in, a
different `DataModel` comes out, and an LM call happens in the middle. If
you think of an LM call as a typed function, a `Generator` *is* that
function.

In Synalinks vocabulary, a `Generator` is a kind of `Module`. (`Module`
is Synalinks' word for a reusable building block — exactly analogous to a
layer in Keras.) When you create one, you tell it what output shape you
want and which LM to talk to:

```python
generator = synalinks.Generator(
    data_model=Answer,
    language_model=language_model,
)
```

Calling it later (`await generator(x)`) makes one LM call, constrained to
produce something matching the `Answer` schema.

Why the `await`? LM calls spend essentially all of their time waiting on
the network. Python's `async`/`await` lets your program issue many such
calls in parallel without each one blocking the next. If `async` is new
to you, just read `await thing` as "wait until `thing` finishes, then
keep going."

## Ingredient 3: `Program` — bundle modules into something you can ship

A `Program` is a container that wraps one or more modules into a single
object you can call, save, load, and (later on) train. If `Generator` is
one typed function, `Program` is the whole pipeline that includes it. The
analogy to Keras is exact: `Module` is to `Generator` what `Layer` is to
`Dense`, and `Program` is the equivalent of `Model`.

Synalinks offers three equivalent ways to build a `Program`. In this
guide we use the **functional** form, which makes the data flow explicit:
you create a placeholder for the input, "call" your modules on that
placeholder, and hand the resulting input/output pair to `Program`.

```python
inputs  = synalinks.Input(data_model=Question)
outputs = await synalinks.Generator(
    data_model=Answer,
    language_model=language_model,
)(inputs)

program = synalinks.Program(inputs=inputs, outputs=outputs, name="qa_program")
```

Here is the subtle part — and the single most common confusion when
people first read this code. `Input(data_model=Question)` does **not**
create a `Question` object. It creates a *symbolic placeholder* that
stands for "a `Question` value that will arrive here later." When you
then call the generator on this placeholder, **no LM call happens**. You
are not running the pipeline; you are drawing it. Synalinks records an
edge in an internal graph saying "the generator will receive whatever
flows into this placeholder."

The LM only runs later, when you call `program(Question(question="..."))`
with a real `Question`. Compare it to ordinary Python: writing
`def f(x): return x + 1` defines a function but does not add anything;
only `f(3)` actually computes. Construction and execution are two
separate steps, and confusing them will make the code look magical when
it is not.

```mermaid
flowchart LR
    Q["Question (symbolic Input)"] --> G["Generator(data_model=Answer)"]
    G --> A["Answer (symbolic output)"]
    A --> P["Program(inputs=Q, outputs=A)"]
```

## End-to-End Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="The question to answer")

class Answer(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Your step-by-step reasoning process")
    answer: str = synalinks.Field(description="The final answer based on your reasoning")

async def main():
    load_dotenv()
    synalinks.clear_session()

    language_model = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
        description="A simple question-answering program",
    )

    result = await program(Question(question="What is the capital of France?"))
    print(f"Thinking: {result['thinking']}")
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

A representative run against `ollama/llama3.2:latest` might print:

```
Thinking: France has multiple capitals depending on the region
Answer: Paris
```

Look closely: the `answer` field is reliably `Paris`, but the `thinking`
field varies from run to run, and sometimes it is factually wrong (France
does not actually have multiple capitals) even when the answer it
produces is right. That is the LM speaking, not the framework.
**Synalinks guarantees the *shape* of the output, not its *truth*.** Making
the model more truthful is the job of techniques we meet later —
optimizers, rewards, and retrieval. For now, celebrate that the *shape*
worked: `result["answer"]` will always be a string, never `None`, never
a paragraph with the answer buried in the middle.

## A Detail That Bites People: `clear_session`

When you create a module without giving it a name, Synalinks invents one
for you — `generator_1`, `generator_2`, and so on, counted off a counter
that lives for the lifetime of the Python process. In a Jupyter notebook,
where you re-run cells without restarting the kernel, that counter just
keeps growing. Each re-run produces different module names, and since
those names appear in saved programs, log files, and traces, the same
code can produce different artifacts on different days.

`synalinks.clear_session()` resets the counter. The habit is simple:
call it once at the top of any script or notebook that builds modules,
right after your imports. Then your runs are reproducible.

## Four Things to Remember

If you take only four ideas from this guide, take these:

- **Construction is not execution.** Building a `Program` draws the
  pipeline; calling the program runs it. (Think: wiring up vs. powering
  on.)
- **A successful `Generator` call returns a typed object.** Its fields
  match exactly what you declared. You can access them with bracket
  notation (`result["answer"]`) or dot notation (`result.answer`) —
  whichever you prefer.
- **Field descriptions are part of the prompt.** Rewording a description
  changes how the program behaves, even though no Python logic changed.
  Treat descriptions with the care you would give to code, not to
  comments.
- **Field order is meaningful.** Reasoning fields belong *before*
  conclusion fields. The LM writes left to right, so whatever comes
  first influences whatever comes after.

## Where to Go Next

- **[Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/) — Data Models.** Nested objects, list fields, enums, custom
  validation. Most real programs use richer schemas than `Question` and
  `Answer`.
- **[Guide 3](https://synalinks.github.io/synalinks/guides/Programs/) — Programs.** The other two ways to build a `Program`
  (subclassing and the `Sequential` shortcut) and when to prefer each.
- **[Guide 4](https://synalinks.github.io/synalinks/guides/Modules/) — Modules.** The catalogue of pre-built modules beyond
  `Generator`: chain-of-thought, decision-making, voting, and more.

## API References

- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


class Question(synalinks.DataModel):
    """Input: A question from the user."""

    question: str = synalinks.Field(description="The question to answer")


class Answer(synalinks.DataModel):
    """Output: An answer with reasoning."""

    thinking: str = synalinks.Field(description="Your step-by-step reasoning process")
    answer: str = synalinks.Field(description="The final answer based on your reasoning")


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_1_getting_started",
    # )

    language_model = synalinks.LanguageModel(model="ollama/llama3.2:latest")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
        description="A simple question-answering program",
    )

    result = await program(Question(question="What is the capital of France?"))

    print(f"Thinking: {result['thinking']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
