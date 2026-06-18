# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Consul of Models

[Guide 29](https://synalinks.github.io/synalinks/guides/Mixture%20of%20Models/) built a mixture of models: several models answer the
same question *in parallel and in isolation*, then one aggregator
reconciles their answers. The proposers never see each other. That is
the pattern's strength (errors stay independent) and its limit (a
proposer cannot be *talked out of* a mistake, because no one talks to
it).

This guide removes that wall. A **consul** is a small council of
models that **deliberate**: they answer in *rounds*, and on every
round each member reads what the others said in the previous round
before speaking again. A model that started out wrong can change its
mind after seeing a stronger argument. At the end, a **chair**
(itself a model) reads the whole transcript and delivers the
council's verdict. This is the **multi-agent debate** setup of
[Improving Factuality and Reasoning through Multiagent Debate (Du, Li, Torralba, Tenenbaum, Mordatch — 2024)](https://arxiv.org/abs/2305.14325),
with the chair playing the role of the judge in
[Multi-Agent Debate (Liang, He, Jiao, Wang, Wang, Wang, Yang, Tu, Shi — 2023)](https://arxiv.org/abs/2305.19118).

The mental model is a jury, not a survey. A survey (mixture of models)
collects independent opinions once. A jury (a consul) lets jurors
argue, concede, and converge — the discussion itself is where the
value is.

## Consul vs. Mixture of Models

Both are multi-model. The difference is **interaction over time**:

| | Mixture of Models | Consul |
|---|---|---|
| Members see each other | Never | Every round |
| Number of passes | One | `rounds` (e.g. 2–3) |
| Error correlation | Stays low | Can rise (groupthink risk) |
| Best for | Decorrelating one-shot errors | Questions where an *argument* changes the answer |
| Final step | Aggregator | Chair (reads the debate) |

Use a consul when the question has a **persuadable** quality — a
reasoning trap, an ambiguous spec, a trade-off with no single right
answer — where *seeing the case made* is what flips a wrong intuition.
The classic example is a cognitive-reflection question: the intuitive
answer is wrong, and one member spelling out the algebra is enough to
pull the others off the wrong answer.

The cost is a real risk: deliberation can produce **groupthink**. If a
confident member states a wrong answer early, the others may anchor to
it instead of correcting it. Two rounds is usually the sweet spot —
enough for a correction to propagate, not so many that the council
just keeps agreeing with itself.

## The Architecture

```mermaid
sequenceDiagram
    participant User
    participant Consul
    participant Members as Members (different models)
    participant Chair

    User->>Consul: Question
    loop For each round
        Consul->>Members: question + transcript so far
        Members-->>Consul: each member's position
        Note over Consul: append every position to the transcript
    end
    Consul->>Chair: question + full transcript
    Chair-->>Consul: final verdict
    Consul-->>User: Answer
```

The whole thing rests on one shared object: the **transcript**, a
running text log of who said what. It is just a string. Each member
gets the question plus the transcript-so-far as input; its answer is
appended to the transcript; the next member sees the longer version.
A plain string (rather than a structured list) is the right choice
here because the *only* consumer is another language model — text is
exactly what it reads best, and it keeps the data model trivial.

## The Data Models

Three small models carry the conversation:

```python
class Question(synalinks.DataModel):
    \"\"\"The question put to the council.\"\"\"
    question: str = synalinks.Field(description="The question to deliberate")

class DebateState(synalinks.DataModel):
    \"\"\"What a member (or the chair) sees on its turn.\"\"\"
    question: str = synalinks.Field(description="The question under debate")
    transcript: str = synalinks.Field(description="Everything said so far")

class Position(synalinks.DataModel):
    \"\"\"One member's contribution on one turn.\"\"\"
    position: str = synalinks.Field(description="This member's current position")
```

The chair reuses `DebateState` as input and emits a final `Answer`.
Members are `ChainOfThought` modules, so each also produces an internal
`thinking` field — useful to inspect, but we only append the public
`position` to the transcript to keep it readable.

## Building the Members

Each council seat is an ordinary **functional `Program`** wrapping a
`ChainOfThought`, backed by a *different* model and given a
deliberative instruction:

```python
async def build_member(name, language_model):
    inputs = synalinks.Input(data_model=DebateState)
    outputs = await synalinks.ChainOfThought(
        data_model=Position,
        language_model=language_model,
        instructions=(
            "You are one member of a council answering a question. Read the "
            "question and the transcript of what other members said. State "
            "your position and reasoning. If you disagree with someone, say "
            "why. If an argument convinced you, update your view. Be concise."
        ),
        name=f"member_{name}",
    )(inputs)
    return synalinks.Program(inputs=inputs, outputs=outputs, name=f"member_{name}")
```

The instruction does the deliberative heavy lifting. The two phrases
that matter are *"if you disagree, say why"* (keeps the debate from
collapsing into instant agreement) and *"if an argument convinced you,
update your view"* (gives members explicit permission to change their
mind — without it, models tend to stubbornly defend their first
answer).

## The Consul Program

The council itself is a **subclassed `Program`** ([Guide 1b / Code
Examples → Subclassing](https://synalinks.github.io/synalinks/Code%20Examples/Subclassing/)). Its `call` method holds the round loop —
ordinary Python `for` loops over rounds and members, which is exactly
why we subclass rather than use the functional API: the control flow
is dynamic and data-dependent.

```python
class Consul(synalinks.Program):
    def __init__(self, members=None, chair=None, rounds=2, **kwargs):
        super().__init__(**kwargs)
        self.members = members   # {name: Program}
        self.chair = chair       # Program
        self.rounds = rounds

    async def call(self, inputs, training=False):
        question = inputs.get("question")
        transcript = "(no statements yet)"
        for r in range(self.rounds):
            for name, member in self.members.items():
                state = DebateState(question=question, transcript=transcript)
                opinion = await member(state)
                line = f"Round {r + 1} — {name}: {opinion['position']}"
                transcript = (
                    line if transcript.startswith("(no")
                    else transcript + "\\n" + line
                )
        return await self.chair(
            DebateState(question=question, transcript=transcript)
        )
```

Two design decisions are worth naming:

1. **Members speak in turn, not simultaneously.** Within a round we
   append each member's position immediately, so later speakers in the
   *same* round already see the earlier ones — a roundtable, not a
   sealed-ballot vote. This propagates a good correction faster (it can
   take effect the same round). If you would rather avoid early speakers
   biasing later ones, run a round's members in parallel (share one
   `inputs`, as in [Guide 29](https://synalinks.github.io/synalinks/guides/Mixture%20of%20Models/)) and append all positions only at the
   round boundary.
2. **A fixed round count, not convergence detection.** We stop after
   `rounds` passes. You *could* stop early once positions stop changing
   — but detecting genuine consensus (rather than parroting) is its own
   hard problem, and a small fixed budget is the robust default.

For serialization, a subclassed program implements `get_config` /
`from_config`. Here they (de)serialise the members dict, the chair, and
the round count:

```python
def get_config(self):
    return {
        "name": self.name,
        "description": self.description,
        "rounds": self.rounds,
        "members": {
            n: synalinks.saving.serialize_synalinks_object(p)
            for n, p in self.members.items()
        },
        "chair": synalinks.saving.serialize_synalinks_object(self.chair),
    }

@classmethod
def from_config(cls, config):
    members = {
        n: synalinks.saving.deserialize_synalinks_object(c)
        for n, c in config.pop("members").items()
    }
    chair = synalinks.saving.deserialize_synalinks_object(config.pop("chair"))
    return cls(members=members, chair=chair, **config)
```

## Complete Example

We put a **cognitive-reflection** question to the council — the
bat-and-ball problem, whose intuitive answer (10¢) is wrong and whose
correct answer (5¢) needs one line of algebra. Watch the transcript:
typically a member blurts the intuitive 10¢, another lays out
`bat = ball + 1.00` and `ball + bat = 1.10 → ball = 0.05`, and the
council converges on 5¢ before the chair rules.

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Question(synalinks.DataModel):
    \"\"\"The question put to the council.\"\"\"
    question: str = synalinks.Field(description="The question to deliberate")

class DebateState(synalinks.DataModel):
    \"\"\"What a member (or the chair) sees on its turn.\"\"\"
    question: str = synalinks.Field(description="The question under debate")
    transcript: str = synalinks.Field(description="Everything said so far")

class Position(synalinks.DataModel):
    \"\"\"One member's contribution on one turn.\"\"\"
    position: str = synalinks.Field(description="This member's current position")

class Answer(synalinks.DataModel):
    \"\"\"The council's final verdict.\"\"\"
    answer: str = synalinks.Field(description="The council's final answer")

async def build_member(name, language_model):
    inputs = synalinks.Input(data_model=DebateState)
    outputs = await synalinks.ChainOfThought(
        data_model=Position,
        language_model=language_model,
        instructions=(
            "You are one member of a council answering a question. Read the "
            "question and the transcript of what other members said. State "
            "your position and reasoning. If you disagree with someone, say "
            "why. If an argument convinced you, update your view. Be concise."
        ),
        name=f"member_{name}",
    )(inputs)
    return synalinks.Program(inputs=inputs, outputs=outputs, name=f"member_{name}")

async def build_chair(language_model):
    inputs = synalinks.Input(data_model=DebateState)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=language_model,
        instructions=(
            "You are the chair of the council. Read the full transcript, "
            "weigh the arguments, resolve disagreements, and deliver the "
            "council's final answer with a one-line justification."
        ),
        name="chair",
    )(inputs)
    return synalinks.Program(inputs=inputs, outputs=outputs, name="chair")

class Consul(synalinks.Program):
    def __init__(self, members=None, chair=None, rounds=2, **kwargs):
        super().__init__(**kwargs)
        self.members = members
        self.chair = chair
        self.rounds = rounds

    async def call(self, inputs, training=False):
        question = inputs.get("question")
        transcript = "(no statements yet)"
        for r in range(self.rounds):
            for name, member in self.members.items():
                state = DebateState(question=question, transcript=transcript)
                opinion = await member(state)
                line = f"Round {r + 1} — {name}: {opinion['position']}"
                transcript = (
                    line if transcript.startswith("(no")
                    else transcript + "\\n" + line
                )
        return await self.chair(
            DebateState(question=question, transcript=transcript)
        )

async def main():
    load_dotenv()
    synalinks.clear_session()

    members = {
        "gemma": await build_member(
            "gemma", synalinks.LanguageModel(model="ollama/gemma:latest")),
        "mistral": await build_member(
            "mistral", synalinks.LanguageModel(model="ollama/mistral:latest")),
        "qwen": await build_member(
            "qwen", synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")),
    }
    chair = await build_chair(synalinks.LanguageModel(model="ollama/mistral:latest"))

    consul = Consul(members=members, chair=chair, rounds=2, name="consul")

    result = await consul(Question(question=(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
        "than the ball. How much does the ball cost?"
    )))
    print(f"Council verdict: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (one run; wording varies by model and run):

```
Council verdict: The ball costs $0.05.
```

(Since `bat = ball + 1.00` and `ball + bat = 1.10`, we get
`2*ball + 1.00 = 1.10`, so `ball = 0.05` and `bat = 1.05`.)

The headline result — 5¢, not the tempting 10¢ — is what a single
small model often gets *wrong* on this question. The council gets it
right not because any one member is reliable, but because once one
member writes down the algebra, the others can read it and concede.
That is the property a consul buys you over a mixture of models: a
wrong first answer is **recoverable** mid-run.

## Take-Home Summary

- A **consul** is a council of different models that
  **deliberate in rounds**: each member sees the others' prior
  statements before speaking again, and a **chair** delivers the final
  verdict.
- It differs from a **mixture of models** ([Guide 29](https://synalinks.github.io/synalinks/guides/Mixture%20of%20Models/)) by adding
  *interaction over time* — members can be argued out of a mistake.
  Use it for **persuadable** questions (reasoning traps, ambiguous
  specs), not for decorrelating one-shot errors.
- The **transcript is a plain string** every member reads and appends
  to; it is the only shared state.
- Members are **functional `Program`s** with deliberative instructions;
  the **`Consul` is a subclassed `Program`** whose `call` holds the
  round loop — dynamic control flow is exactly what subclassing is for.
- **Two rounds is the usual sweet spot.** More rounds risk
  **groupthink**: the council agreeing with itself rather than
  improving.

## API References

- [ChainOfThought](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/ChainOfThought%20module/)
- [Program (subclassing)](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
- [Saving and serialization](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/Program%20Saving%20API/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Question(synalinks.DataModel):
    """The question put to the council."""

    question: str = synalinks.Field(description="The question to deliberate")


class DebateState(synalinks.DataModel):
    """What a member (or the chair) sees on its turn."""

    question: str = synalinks.Field(description="The question under debate")
    transcript: str = synalinks.Field(description="Everything said so far")


class Position(synalinks.DataModel):
    """One member's contribution on one turn."""

    position: str = synalinks.Field(description="This member's current position")


class Answer(synalinks.DataModel):
    """The council's final verdict."""

    answer: str = synalinks.Field(description="The council's final answer")


# =============================================================================
# Council members and chair (functional Programs)
# =============================================================================


async def build_member(name, language_model):
    """Build one council seat: a ChainOfThought backed by one model."""
    inputs = synalinks.Input(data_model=DebateState)
    outputs = await synalinks.ChainOfThought(
        data_model=Position,
        language_model=language_model,
        instructions=(
            "You are one member of a council answering a question. Read the "
            "question and the transcript of what other members said. State "
            "your position and reasoning. If you disagree with someone, say "
            "why. If an argument convinced you, update your view. Be concise."
        ),
        name=f"member_{name}",
    )(inputs)
    return synalinks.Program(inputs=inputs, outputs=outputs, name=f"member_{name}")


async def build_chair(language_model):
    """Build the chair that synthesises the council's final verdict."""
    inputs = synalinks.Input(data_model=DebateState)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=language_model,
        instructions=(
            "You are the chair of the council. Read the full transcript, "
            "weigh the arguments, resolve disagreements, and deliver the "
            "council's final answer with a one-line justification."
        ),
        name="chair",
    )(inputs)
    return synalinks.Program(inputs=inputs, outputs=outputs, name="chair")


# =============================================================================
# The Consul: a subclassed Program holding the round loop
# =============================================================================


class Consul(synalinks.Program):
    """A council of models that deliberate in rounds, then a chair rules."""

    def __init__(self, members=None, chair=None, rounds=2, **kwargs):
        super().__init__(**kwargs)
        self.members = members
        self.chair = chair
        self.rounds = rounds

    async def call(self, inputs, training=False):
        question = inputs.get("question")
        transcript = "(no statements yet)"
        for r in range(self.rounds):
            for name, member in self.members.items():
                state = DebateState(question=question, transcript=transcript)
                opinion = await member(state)
                line = f"Round {r + 1} — {name}: {opinion['position']}"
                transcript = (
                    line if transcript.startswith("(no") else transcript + "\n" + line
                )
        return await self.chair(DebateState(question=question, transcript=transcript))

    def get_config(self):
        return {
            "name": self.name,
            "description": self.description,
            "rounds": self.rounds,
            "members": {
                n: synalinks.saving.serialize_synalinks_object(p)
                for n, p in self.members.items()
            },
            "chair": synalinks.saving.serialize_synalinks_object(self.chair),
        }

    @classmethod
    def from_config(cls, config):
        members = {
            n: synalinks.saving.deserialize_synalinks_object(c)
            for n, c in config.pop("members").items()
        }
        chair = synalinks.saving.deserialize_synalinks_object(config.pop("chair"))
        return cls(members=members, chair=chair, **config)


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # Log every module call (each member's turn, the chair) to the console.
    synalinks.enable_logging()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_30_consul",
    # )

    # Three DIFFERENT models take the three council seats.
    members = {
        "gemma": await build_member(
            "gemma", synalinks.LanguageModel(model="ollama/gemma:latest")
        ),
        "mistral": await build_member(
            "mistral", synalinks.LanguageModel(model="ollama/mistral:latest")
        ),
        "qwen": await build_member(
            "qwen", synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")
        ),
    }
    chair = await build_chair(synalinks.LanguageModel(model="ollama/mistral:latest"))

    consul = Consul(members=members, chair=chair, rounds=2, name="consul")

    # -------------------------------------------------------------------------
    # Put a cognitive-reflection question to the council
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Consul: a council of models deliberating in rounds")
    print("=" * 60)

    question = (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
        "than the ball. How much does the ball cost?"
    )
    print(f"\nQuestion: {question}")

    result = await consul(Question(question=question))
    print(f"\nCouncil verdict: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
