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
- a score field (`synalinks.Rating`, `synalinks.Score`, ...) rates along its
  scale, and the answer is a value of that scale.

```python
class Triage(synalinks.DataModel):
    is_billing: bool = synalinks.Field(
        description="Is the ticket about billing?",
    )
    urgency: Literal["low", "medium", "high"] = synalinks.Field(
        description="How urgent is the ticket?",
    )
    frustration: synalinks.Rating = synalinks.Field(
        description="How frustrated is the customer, from 1 (calm) to 5 (angry)?",
    )
```

For a score, the decision model does not pick a single level: it spreads its
probability over the levels, and the answer is the level at the center of
that spread. A customer torn between "annoyed" and "angry" lands between the
two, rather than on whichever a model happened to generate. Say what the
scale means in the `description`, as above: that is how the decision model
knows which end is which.

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

## Decision Models in Modules

Given a decision model, each of these modules keeps its job but drops its
free-text part: there is no `thinking` to write before a decision, and no
`critique` to write before a grade. What remains is the decision itself,
which the decision model makes faster and more reliably. The examples below
run on a support ticket (`x0 = synalinks.Input(data_model=Ticket)`).

### `Generator`

The `Generator` is the most general of them: give it a data model made of
questions, and the decision model answers them all at once. The
`instructions` and examples you give the `Generator` still apply: they form
the system message the questions are answered in, exactly as for a language
model. Use it when a step needs several answers about the same input, like
triaging a ticket.

```python
triage = await synalinks.Generator(
    data_model=Triage,
    decision_model=decision_model,
    instructions="Triage the support tickets of an online shop.",
)(x0)
# {"is_billing": true, "urgency": "high", "frustration": 4}
```

### `Decision`

A `Decision` picks one label out of a closed list: single-label
classification. With a language model, it first writes its reasoning in a
`thinking` field, then picks. With a decision model, the `question` is asked
as is, the labels are the options, and the output is only the `choice`. Since
the question is all the decision model reads besides the input, phrase it
about the input, and give labels that speak for themselves.

```python
team = await synalinks.Decision(
    question="Which team should handle the ticket?",
    labels=["billing", "technical", "sales"],
    decision_model=decision_model,
)(x0)
# {"choice": "billing"}
```

A decision model also knows how sure it is. Set `min_confidence`, and a
decision it is not sure enough about is not taken: the `Decision` returns
`None` instead of a guess, so the program can hand the input to something
else, a stronger model or a human.

```python
team = await synalinks.Decision(
    question="Which team should handle the ticket?",
    labels=["billing", "technical", "sales"],
    decision_model=decision_model,
    min_confidence=0.7,
)(x0)
# {"choice": "billing"}, or None when the model is unsure
```

### `MultiDecision`

A `MultiDecision` picks every label that applies: multi-label
classification. A decision model answers it as one yes/no question per
label ("Which topics does the ticket mention? Does the label 'payment'
apply?"), and the labels whose probability reaches the `threshold` (0.5 by
default) are kept. Raise it to keep only the labels the model is sure about,
lower it to miss fewer. As each label is judged on its own, the result can
hold several labels, or none at all when nothing applies.

```python
topics = await synalinks.MultiDecision(
    question="Which topics does the ticket mention?",
    labels=["payment", "delivery", "account"],
    decision_model=decision_model,
)(x0)
# {"choices": ["payment", "account"]}
```

### `Branch`

A `Branch` is a `Decision` wired to one module per label: the module of the
chosen label runs, and the other branches return `None` (see
[Guide 5](https://synalinks.github.io/synalinks/guides/Control%20Flow/)). The
decision model only makes the choice: the branches keep their own models, so
a branch can still be a language model writing an answer. This is where a
decision model pays off most, since the routing runs on every input. A
`Branch` takes a `min_confidence` too (a `threshold` with a `MultiDecision`):
when the decision is not taken, no branch runs.

```python
(billing, technical) = await synalinks.Branch(
    question="Which team should handle the ticket?",
    labels=["billing", "technical"],
    branches=[billing_agent, technical_agent],
    decision_model=decision_model,
)(x0)
```

### `SelfCritique`

A `SelfCritique` grades its inputs, typically an answer, with a `reward`
between 0 and 1. A decision model rates the inputs on five levels, from
"Very bad." to "Very good.", and the reward follows the probability of each
level, so an answer the model hesitates about lands between two levels
rather than on one. It does not write the critique a language model would,
so it only works with a reward (`return_reward=True`, the default).

```python
graded = await synalinks.SelfCritique(decision_model=decision_model)(reply)
# {..., "reward": 0.75}
```

### `RubricsAsJudge`

A `RubricsAsJudge` grades an answer against named criteria and combines them
by weight ([Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/)).
A decision model asks each criterion as a question ("How well does the
answer meet this criterion: ...?") over five levels, from "Does not meet the
criterion at all." to "Fully meets the criterion.", and grades all the
criteria in a single call, however many there are. Instead of a written
critique, the `critique` lists the level and confidence of each criterion,
which is often all you need to see why an answer lost points. The built-in
rubric rewards, such as `Faithfulness` or `Toxicity`, take a `decision_model`
the same way.

```python
reward = synalinks.rewards.RubricsAsJudge(
    rubrics=[
        {"name": "correct", "description": "The answer is correct.", "weight": 2},
        {"name": "concise", "description": "No preamble, no filler."},
    ],
    decision_model=decision_model,
)
toxicity = synalinks.rewards.Toxicity(decision_model=decision_model)
```

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

## How Sure Is Sure Enough?

`min_confidence` and `threshold` trade one kind of mistake for another. Set
them low, and the program always decides, including when it should not. Set
them high, and it only decides when it is sure, at the price of handing more
inputs over to a slower path. Where the balance lies depends on what a wrong
decision costs you, compared with an unanswered one, and on your data: there
is no good default.

These thresholds are **hyperparameters**, not parameters: training does not
change them, you fix them before it starts. So rather than guessing, search
for them, like any other hyperparameter, with the KerasTuner tuners of
[Guide 17](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/).
The one thing to get right is the reward. If an unanswered input scored 0,
like a wrong one, the search would learn that abstaining never pays and set
the threshold to 0. Give it a score between a right and a wrong decision,
reflecting what an abstention is worth to you:

```python
import synalinks

synalinks.disable_keras_backend()  # before using the tuners (see Guide 17)


async def triage_reward(y_true, y_pred):
    if y_pred is None:
        return 0.5  # not decided: better than wrong, worse than right
    return 1.0 if y_pred.get("choice") == y_true.get("choice") else 0.0


async def build_program(hp):
    min_confidence = hp.Float("min_confidence", 0.0, 0.9, step=0.1)
    synalinks.clear_session()
    inputs = synalinks.Input(data_model=Ticket)
    outputs = await synalinks.Decision(
        question="Which team should handle the ticket?",
        labels=["billing", "technical", "sales"],
        decision_model=decision_model,
        min_confidence=min_confidence,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)
    program.compile(reward=triage_reward)
    return program


tuner = synalinks.tuners.GridSearch(
    build_program,
    objective=synalinks.tuners.Objective("val_reward", direction="max"),
    directory="tuning",
    project_name="triage_min_confidence",
)
tuner.search(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=1)
```

The grid tries every threshold from 0 to 0.9, and the best one is the
balance that fits your data, not a number picked by hand. Decision model
calls are cheap, so trying many thresholds costs little.

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

The example below runs every module above with a decision model:

1. **Triage**: a `Generator` answers the questions of its data model about a
   support ticket, a `Decision` picks the team that handles it, and a
   `MultiDecision` the topics it mentions.
2. **Routing**: a `Branch` decided by the decision model sends easy queries to
   a plain `Generator` and difficult ones to a `ChainOfThought`, both answered
   by a language model.
3. **Grading**: a `SelfCritique` grades a reply, and a `RubricsAsJudge` grades
   two answers against weighted criteria in one call.

## Take-Home Summary

- A **decision model** answers typed questions (yes/no, one of a list, a
  score) with calibrated probabilities; it never writes text.
- **Fields are questions**: `bool`, `Literal` and score (`synalinks.Rating`,
  `synalinks.Score`) fields, each asked with its `description`. A field that
  has to be written needs a language model.
- Modules that decide take it through their **`decision_model`** argument,
  never as a `language_model`; `set_default_decision_model()` sets a default.
- **Route cheap, write expensive**: decide with a decision model, write with
  a language model only where needed.
- It **knows how sure it is**: `min_confidence` and `threshold` let a
  decision be skipped rather than guessed, and KerasTuner finds the values
  that fit your data.
- It **learns in context**, so programs using it train like any other, with
  every optimizer, `OMEGA` included, and it makes a fast, cheap judge.

## API References

- [DecisionModel](https://synalinks.github.io/synalinks/Synalinks%20API/Decision%20Models%20API/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [Decision](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Decision%20module/)
- [Branch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Branch%20module/)
- [RubricsAsJudge](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/RubricsAsJudge%20reward/)
- [OMEGA](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/OMEGA/)
- [Tuners](https://synalinks.github.io/synalinks/Synalinks%20API/Tuners/)
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
    frustration: synalinks.Rating = synalinks.Field(
        description="How frustrated is the customer, from 1 (calm) to 5 (angry)?",
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
    print("Triage: a Generator, a Decision and a MultiDecision")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Ticket)
    answers = await synalinks.Generator(
        data_model=Triage,
        decision_model=decision_model,
        instructions="Triage the support tickets of an online shop.",
    )(inputs)
    team = await synalinks.Decision(
        question="Which team should handle the ticket?",
        labels=["billing", "technical", "sales"],
        decision_model=decision_model,
    )(inputs)
    topics = await synalinks.MultiDecision(
        question="Which topics does the ticket mention?",
        labels=["payment", "delivery", "account"],
        decision_model=decision_model,
    )(inputs)
    # The three read the same input, so they run in parallel.
    triage = synalinks.Program(
        inputs=inputs,
        outputs=answers + team + topics,
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
    print("Grading: SelfCritique and RubricsAsJudge with a decision model")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Answer)
    outputs = await synalinks.SelfCritique(decision_model=decision_model)(inputs)
    critic = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="self_critique",
        description="Grade a reply",
    )
    graded = await critic(Answer(answer="The trip takes 3 hours and 35 minutes."))
    print(graded.prettify_json())

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
