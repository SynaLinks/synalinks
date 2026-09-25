"""
# Decision Models

A **decision model** answers typed questions about its input instead of
generating text. Ask it "is this about billing?" and it returns the
probability of *yes*; ask it to pick a team and it returns one of the teams
you listed, with the probability of each; ask it to rate an answer and it
returns a score along the levels you gave. It never writes a sentence.

That restriction is what makes it useful. A decision model (a "System One"
model, such as TypeSafe's `jev`) is much faster and cheaper than a language
model, its answers are **calibrated probabilities** rather than sampled
tokens, and an answer is **always** one of the options you gave: there is no
output to parse and no invalid label to recover from. That makes it a good
fit for everything a program decides rather than writes: routing,
classification, guards and grading.

What it does not do: reason step by step, generate text or values, call
tools, or read images. When a step needs any of those, keep a language model.

## Setting the API Key

Like language models, decision models read their credentials from
environment variables. Set `TYPESAFE_API_KEY` (and, to use another endpoint,
`TYPESAFE_BASE_URL`), ideally in a `.env` file kept out of version control:

```
TYPESAFE_API_KEY=your-api-key
```

and load it at the start of your script with `load_dotenv()`, as this guide
does. The key is read on every call, so it is never stored in the program's
config nor written to disk when you save a program.
If the key is missing, a call fails like any other failed call: it warns and
returns `None` (or asks the `fallback` model, if any).

```python
decision_model = synalinks.DecisionModel(model="typesafe/jev-latest")
```

Pin a versioned ID (e.g. `"typesafe/jev-1.13.0"`) once you have tuned
thresholds on a model, so an upgrade of `jev-latest` does not move them.

## The `decision_model` Argument

A `DecisionModel` is called like a `LanguageModel`, with chat messages and a
target output schema. The modules that can use one take it through their own
**`decision_model`** argument: `Generator`, `Decision`, `MultiDecision`,
`Branch`, `SelfCritique`, `RubricsAsJudge` and the rubric rewards. It is never
a `language_model`: a module given a decision model as its `language_model`
raises an error, so a decision model cannot end up in a module that needs to
generate text. The difference is what the schema may contain. The decision model turns each
field of the output data model into one question, asked with the field's
`description`:

| Field type | Question | Output value |
|---|---|---|
| `bool` | yes/no | `True` when the probability of yes is at least 0.5 |
| `Literal[...]` or a string `Enum` | pick one option | the most probable option |
| `noul_schema(...)` | yes/no | `{"noul": p}` |
| `choice_schema(...)` | pick one (described) option | `{"choice", "probabilities", "confidence"}` |
| `score_schema(...)` | rate along 2 to 10 ordered levels | `{"score", "legend", "probabilities", "confidence"}` |

The three `*_schema` helpers (in `synalinks.decision_models`) keep the
probabilities in the output, for when you need more than the top answer.

Any other field (a free-form `str`, a number, a list...) would need
generation, so it raises an `UnsupportedSchemaError`. Use
`decision_model.check_schema(schema)` to check a schema up front.

```python
class Triage(synalinks.DataModel):
    is_billing: bool = synalinks.Field(
        description="Is the ticket about billing?",
    )
    urgency: Literal["low", "medium", "high"] = synalinks.Field(
        description="How urgent is the ticket?",
    )

triage = synalinks.Generator(
    data_model=Triage,
    decision_model=decision_model,
    instructions="Triage the support tickets of an online shop.",
)
```

### The Context Is the State

The messages the `Generator` builds, the system message with the
instructions and few-shot examples, then the inputs, are sent to the
decision model as its **state**: the context every question is answered in.
So the instructions and examples work exactly as with a language model, and
optimizers improve them the same way (in-context learning).

All the questions of a call see the same state and are answered
independently and in parallel: put every question about an input in one data
model rather than making several calls.

## Decision Models in Modules

The modules whose output is a decision rather than a text accept a
`decision_model`. They switch to a data model made of questions when they get
one, so the output loses its free-text field:

- **`Decision`** asks the `question` as is, over the `labels`. The output is
  `{"choice": ...}`, without `thinking`.
- **`Branch`** routes through its `Decision`, so it works unchanged.
- **`MultiDecision`** asks one yes/no question per label, and keeps the
  labels answered yes: `{"choices": [...]}`, without `thinking`.
- **`SelfCritique`** grades the inputs on five levels, from "Very bad." to
  "Very good.". The output `reward` is normalized to [0, 1] as usual, but
  there is no `critique` (so `return_reward` must stay `True`).
- **`RubricsAsJudge`** (and the built-in rubric rewards, such as
  `Faithfulness`) grades every criterion as a score question in a single
  call. The `critique` lists the level and confidence of each criterion.

```mermaid
graph LR
    Q[Query] --> D{Decision<br/>decision model}
    D -->|easy| A1[Generator<br/>language model]
    D -->|difficult| A2[ChainOfThought<br/>language model]
    A1 --> O[Answer]
    A2 --> O
```

A common pattern is the one above: a decision model routes, a language model
writes. The cheap, fast decision runs on every request, and only the branch
that needs it pays for step by step reasoning.

## In-Context Learning

Decision models learn in context, like language models. Their answers are
conditioned on the whole state, so the instructions and the few-shot examples
of the system message change how they answer, without any weight update: show
a decision model a few tickets labeled the way your team triages them, and it
follows that labeling on the next ones.

That makes a program answered by a decision model **trainable the same way**
as one answered by a language model. The trainable variables do not change:
a `Generator` keeps its instructions and examples in its state whatever model
it calls, and so do the modules built on it (`Decision`, `Branch`,
`MultiDecision`, `SelfCritique`). Every optimizer works on them unchanged,
from `RandomFewShot`, which selects the few-shot examples, to **`OMEGA`**,
which evolves the instructions with a genetic algorithm.

With `OMEGA`, the language model given to the optimizer writes the new
candidate instructions (mutations and crossovers), the embedding model keeps
them diverse (Dominated Novelty Search), and the decision model answers the
program's questions with each candidate in its state. Since evaluating a candidate only costs decision model calls, many
candidates can be tried for the price of a few language model calls.

```python
program.compile(
    reward=synalinks.rewards.ExactMatch(),
    optimizer=synalinks.optimizers.OMEGA(
        language_model=language_model,  # writes the candidates
        embedding_model=embedding_model,  # keeps them diverse
    ),
)
history = await program.fit(x=x_train, y=y_train, epochs=4)
```

Here the program's modules use the decision model, and `OMEGA` uses a
language model: the one that proposes candidates never has to be the one
that answers.

## Grading While Training

Because a decision model grades all the criteria of a rubric in one cheap
call, `RubricsAsJudge` with a decision model is a practical reward to train a
program with: `compile()` it as the reward, then `fit()` as usual.

```python
program.compile(
    reward=synalinks.rewards.RubricsAsJudge(
        decision_model=decision_model,
        rubrics=[
            {"name": "correct", "description": "The answer is correct.", "weight": 2},
            {"name": "concise", "description": "No preamble, no filler."},
        ],
    ),
    optimizer=synalinks.optimizers.RandomFewShot(),
)
```

The decision model tracks its usage with the same counters as a language
model: calls, tokens, latency, cost (only input tokens are billed), failed
calls and fallback activations, split by phase (inference, reward,
optimizer). So the language model operational metrics (`TotalTokens`,
`Cost`, `AvgLatency`, `ErrorRate`, their `Reward*` and `Optimizer*`
variants...) count its calls too, including those of a judge in the reward
phase, and so do `ProgramCost` and `BudgetStopping`.

With observability enabled, each call is traced like a language model call
(a `CHAT_MODEL` span with the messages, token usage and cost), plus the
versioned model that answered and the raw answers with their probabilities
and confidence. A call that failed every retry is marked as
failed, even though it returns `None`.

## Reliability

- **Retries**: rate limiting (429), overload (529) and transient server
  errors are retried with backoff (`retry`, default 5 attempts); a
  `Retry-After` header is honored. Validation (422) and auth errors are not
  retried.
- **Validation**: every answer is checked against its question. A missing
  answer, a choice outside the options, or a probability outside [0, 1]
  fails the call instead of reaching your program.
- **Fallback**: `fallback=` takes another decision model to call when every
  attempt failed. Without one, a failed call returns `None`, which flows
  through the program like any other missing value.
- **Cache**: `cache_dir=` saves every response on disk, keyed by the full
  request, so reruns (e.g. evaluations) do not pay twice.

## Key Takeaways

- **Decide, don't generate**: a decision model answers yes/no, choice and
  score questions with calibrated probabilities, and nothing else.
- **Its own argument**: pass it as the `decision_model` of the modules that
  support it; the output data model defines the questions. A module never
  takes one as its `language_model`.
- **Fields are questions**: `bool`, `Literal`/`Enum` and the `*_schema`
  helpers, each asked with its `description`. Anything else raises an
  `UnsupportedSchemaError`.
- **The messages are the state**: instructions and examples reach the
  decision model through the system message.
- **In-context learning**: decision models learn from their instructions and
  examples, so every optimizer, `RandomFewShot` and `OMEGA` included, trains
  a program that uses them exactly as with a language model.
- **Route cheap, write expensive**: decide with a decision model, generate
  with a language model only where needed.
- **Credentials from the environment**: `TYPESAFE_API_KEY`, loaded from a
  `.env` file.

## API References

- [Decision Models API](https://synalinks.github.io/synalinks/Synalinks%20API/Decision%20Models%20API/)
- [OMEGA](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/OMEGA/)
- [Language Model operational metrics](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/Language%20Model%20operational%20metrics/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Decision](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Decision%20module/)
- [Branch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Branch%20module/)
"""

import asyncio
from typing import Literal

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Ticket(synalinks.DataModel):
    """A support ticket."""

    message: str = synalinks.Field(description="The customer message")


class Triage(synalinks.DataModel):
    """The questions a decision model answers about a ticket."""

    is_billing: bool = synalinks.Field(
        description="Is the ticket about billing?",
    )
    urgency: Literal["low", "medium", "high"] = synalinks.Field(
        description="How urgent is the ticket?",
    )


class Query(synalinks.DataModel):
    """User request."""

    query: str = synalinks.Field(description="The user query")


class Answer(synalinks.DataModel):
    """Final answer."""

    answer: str = synalinks.Field(description="The correct answer")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # Log every module call to the console.
    synalinks.enable_logging()

    # Reads `TYPESAFE_API_KEY` (and `TYPESAFE_BASE_URL`) from the environment.
    decision_model = synalinks.DecisionModel(model="typesafe/jev-latest")
    language_model = synalinks.LanguageModel(model="ollama/mistral:latest")

    # -------------------------------------------------------------------------
    # 1. A Generator with a decision model: every field is a question
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Triage: a Generator answered by a decision model")
    print("=" * 60)

    decision_model.check_schema(Triage.get_schema())

    inputs = synalinks.Input(data_model=Ticket)
    outputs = await synalinks.Generator(
        data_model=Triage,
        decision_model=decision_model,
        instructions="Triage the support tickets of an online shop.",
    )(inputs)
    triage = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="ticket_triage",
        description="Triage the support tickets",
    )

    result = await triage(
        Ticket(message="I was charged twice for my order. Please fix this today!")
    )
    print(result.prettify_json())

    # -------------------------------------------------------------------------
    # 2. Route with a decision model, write with a language model
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Routing: a Branch decided by a decision model")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    (easy, difficult) = await synalinks.Branch(
        question="What is the difficulty level of the query?",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=Answer,
                language_model=language_model,
            ),
            synalinks.ChainOfThought(
                data_model=Answer,
                language_model=language_model,
            ),
        ],
        decision_model=decision_model,
        return_decision=False,
    )(inputs)
    outputs = easy | difficult
    router = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="conditional_reasoning",
        description="Think step by step only when the query needs it",
    )

    for query in [
        "What is the capital of France?",
        "A train leaves at 9:40 and arrives at 13:15. How long is the trip?",
    ]:
        result = await router(Query(query=query))
        print(result.prettify_json())

    # -------------------------------------------------------------------------
    # 3. Grade with rubrics: every criterion in one decision model call
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Grading: RubricsAsJudge with a decision model")
    print("=" * 60)

    reward = synalinks.rewards.RubricsAsJudge(
        decision_model=decision_model,
        rubrics=[
            {
                "name": "correct",
                "description": "The answer matches the reference answer.",
                "weight": 2.0,
            },
            {
                "name": "concise",
                "description": "No preamble, no restating the question.",
            },
        ],
    )
    grades = await reward.program(
        [
            Answer(answer="The trip takes 3 hours and 35 minutes."),
            Answer(answer="Sure! The trip takes 3 hours and 35 minutes."),
        ]
    )
    print(grades.prettify_json())

    # The same reward trains a program: `program.compile(reward=reward, ...)`
    # then `await program.fit(...)`.

    print(f"Decision model calls: {decision_model.cumulated_calls}")
    print(f"Decision model cost: ${decision_model.cumulated_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
