# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Decision Models

Most steps of a program *write*: an answer, a summary, a plan. But many steps
only *decide*: is this ticket about billing? Which branch should handle this
query? Is this answer grounded in its context? A language model can do both,
but for a decision it is overkill: it generates tokens you then have to trust
to be one of your labels.

A **decision model** does only the second job. It never writes a sentence:
it answers typed questions about its input, a yes or a no, one option out of
a list, or a score along ordered levels, each with **calibrated
probabilities**. Its answer is always one of the options you gave, so there is
nothing to parse and no invalid label to recover from, and it is much faster
and cheaper than a language model.

The mental model is a form, not an essay. A language model writes the essay;
a decision model ticks the boxes of a form you designed. When a step of your
program is a form, give it to a decision model.

## Decision Models vs. Language Models

| | Language model | Decision model |
|---|---|---|
| Output | Generated text (or JSON) | Answers to typed questions |
| Guarantee | Constrained by the schema | Always one of your options |
| Confidence | None | Calibrated probabilities |
| Reasoning | Step by step, tools, images | None |
| Cost and latency | Higher | Much lower |
| Best for | Writing, reasoning, acting | Routing, classification, guards, grading |

The two are complementary, and the best programs use both: a decision model
for every *decision* on the path, a language model only where something has
to be *written*.

## Fields Are Questions

A decision model is driven by a data model, like a language model
([Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/)). The
difference is that each field is a *question*, asked with the field's
`description`:

- a `bool` field is a yes/no question,
- a `Literal` (or string `Enum`) field picks one of its options,
- a `score_schema` field rates along ordered levels (see the API reference).

```python
class Triage(synalinks.DataModel):
    is_billing: bool = synalinks.Field(
        description="Is the ticket about billing?",
    )
    urgency: Literal["low", "medium", "high"] = synalinks.Field(
        description="How urgent is the ticket?",
    )
```

What a decision model cannot answer is a field that has to be *written*: a
free-form string, a number, a list. Synalinks refuses such a data model up
front instead of failing at run time.

All the questions of a data model are answered in one call, in parallel, so
put every question about an input in the same data model.

## The `decision_model` Argument

A decision model is set up like a language model, with its API key in the
environment (`TYPESAFE_API_KEY`, e.g. in a `.env` file):

```python
decision_model = synalinks.DecisionModel(model="typesafe/jev-latest")
```

The modules that can decide take it through their own `decision_model`
argument: `Generator`, `Decision`, `Branch`, `MultiDecision`, `SelfCritique`
and `RubricsAsJudge`. It is never passed as a `language_model`, so a decision
model cannot end up in a module that needs to write. With
`synalinks.set_default_decision_model(...)`, these modules use it by default,
the same way `set_default_language_model` works for language models.

```python
triage = synalinks.Generator(
    data_model=Triage,
    decision_model=decision_model,
    instructions="Triage the support tickets of an online shop.",
)
```

Given a decision model, the decision modules keep their job but drop their
free-text part: a `Decision` answers without its `thinking`, a `SelfCritique`
grades without writing a `critique`, and a `RubricsAsJudge` grades every
criterion in a single call.

## Route Cheap, Write Expensive

The pattern this enables is the one below: a decision model routes each query
([Guide 5](https://synalinks.github.io/synalinks/guides/Control%20Flow/)), and
only the branch that needs it pays for a language model reasoning step by
step.

```mermaid
graph LR
    Q[Query] --> D{Branch<br/>decision model}
    D -->|easy| A1[Generator<br/>language model]
    D -->|difficult| A2[ChainOfThought<br/>language model]
    A1 --> O[Answer]
    A2 --> O
```

The routing decision runs on every request, so making it cheap and fast
matters more than anywhere else in the program.

## Learning in Context

A decision model answers in the context of the whole conversation: the
system message, with the instructions and the few-shot examples, then the
inputs. So it learns in context like a language model: show it a few tickets
labeled the way your team triages them, and it follows that labeling.

That means a program answered by a decision model trains **the same way**
([Guide 15](https://synalinks.github.io/synalinks/guides/Training/)). The
trainable variables are the same instructions and examples, and every
optimizer works on them: `RandomFewShot` selects the examples, and `OMEGA`
evolves the instructions, with its own language model writing the candidates
while the decision model answers with each of them. As evaluating a candidate
only costs decision model calls, many candidates can be tried cheaply.

A decision model is also a natural **judge**
([Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/)): a
`RubricsAsJudge` given one grades every criterion in one cheap call, which
makes it a practical reward to train with.

## Complete Example

The example below builds three programs with a decision model:

1. **Triage**: a `Generator` whose data model is made of questions answers a
   support ticket.
2. **Routing**: a `Branch` decided by the decision model sends easy queries to
   a plain `Generator` and difficult ones to a `ChainOfThought`, both answered
   by a language model.
3. **Grading**: a `RubricsAsJudge` grades two answers against weighted
   criteria in one call.

## Take-Home Summary

- A **decision model** answers typed questions (yes/no, one of a list, a
  score) with calibrated probabilities; it never writes text.
- **Fields are questions**: `bool` and `Literal` fields, each asked with its
  `description`. A field that has to be written needs a language model.
- Modules that decide take it through their **`decision_model`** argument,
  never as a `language_model`; `set_default_decision_model()` sets a default.
- **Route cheap, write expensive**: decide with a decision model, write with
  a language model only where needed.
- It **learns in context**, so programs using it train like any other, with
  every optimizer, `OMEGA` included, and it makes a fast, cheap judge.

## API References

- [DecisionModel](https://synalinks.github.io/synalinks/Synalinks%20API/Decision%20Models%20API/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Decision](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Decision%20module/)
- [Branch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Branch%20module/)
- [RubricsAsJudge](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/RubricsAsJudge%20reward/)
- [OMEGA](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/OMEGA/)
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
