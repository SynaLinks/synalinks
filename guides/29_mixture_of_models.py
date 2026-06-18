# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Mixture of Models

[Guide 6](https://synalinks.github.io/synalinks/guides/Agents/) built a *single* agent: one model, one tool loop, one
answer. This guide wires several models together into one program.
The simplest way to do that — and a good first multi-model pattern to
learn — is a **mixture of models**: the model-level form of
[Mixture-of-Agents (Wang, Wang, Athiwaratkun, Zhang, Zou — 2024)](https://arxiv.org/abs/2406.04692).
We say "models" rather than "agents" because the members here are plain
reasoners, not tool-using agents — though, as you will see, swapping
in real agents is a one-line change.

The idea is borrowed from how a newsroom works. You do not ask one
journalist to write the definitive story alone. Several reporters each
draft the same story independently; then an editor reads every draft
and writes the final version, keeping the strongest parts of each and
dropping the mistakes. No single reporter has to be perfect, because
the editor's job is to reconcile them.

A mixture of models is exactly that, with language models in both
roles:

- A layer of **proposers** — several models that answer the *same*
  question *in parallel*, independently of one another.
- A single **aggregator** — one more model that reads the original
  question together with every proposal and synthesises one final
  answer.

The bet is that *diverse* mistakes cancel out. If three different
models are wrong, they are usually wrong in *different* ways, and the
aggregator can see the disagreement and resolve it. This is the same
reason ensembles beat single models in classical machine learning —
here we just do it with whole models instead of classifiers.

## Why Not Just Use One Big Model?

Two reasons, one practical and one about reliability.

1. **Diversity is cheap insurance.** A single model has a single set
   of blind spots. Run the same prompt twice and you get correlated
   errors. Run three *different* models and their errors decorrelate,
   so a majority is more often right than any one of them alone.
2. **Heterogeneity is a feature, not a bug.** Different model families
   (say a Gemma, a Mistral, and a Qwen) were trained on different data
   with different recipes. They are good at different things. The
   aggregator gets to pick the best of each, rather than being stuck
   with whatever one model happened to know.

The cost is more LM calls — `N` proposers plus one aggregator instead
of one call. Because the proposers are independent, Synalinks runs
them **concurrently**, so the wall-clock cost is roughly *one slow
proposer plus the aggregator*, not the sum of all of them.

## The Architecture

```mermaid
flowchart TD
    Q["Query"] --> A1["Proposer A (model 1)"]
    Q --> A2["Proposer B (model 2)"]
    Q --> A3["Proposer C (model 3)"]
    A1 --> M["resilient merge (|): query + surviving proposals"]
    A2 --> M
    A3 --> M
    M --> AGG["Aggregator (ChainOfThought)"]
    AGG --> O["Final Answer"]
```

This maps onto three Synalinks features you have already met
separately, now combined:

- **Parallel branches** (Code Examples → *Parallel Branches*): when
  several modules consume the *same* input node, Synalinks schedules
  them concurrently. The three proposers all read `inputs`, so they
  fan out automatically — you do not write any threading code.
- **The resilient-merge operator `|`** (Code Examples → *Data Model
  Operators*): merges the proposals (and the original query) into a
  single data model to feed the aggregator. When two fields share a
  name, the merge keeps both by adding a numeric suffix — `candidate`,
  `candidate_1`, `candidate_2` — so no proposal is silently dropped.
  We use `|` rather than `+` so a *failed* proposer is dropped instead
  of crashing the panel — see *Resilient Merging* below.
- **Reasoning modules** ([Guide 4](https://synalinks.github.io/synalinks/guides/Modules/)): each proposer here is a
  `ChainOfThought`, so every model reasons before committing to a
  candidate. A proposer can equally well be a full
  `FunctionCallingAgent` — see *Giving Proposers Tools* below.

## What Counts as a Proposer?

A proposer is *any module that maps the query to a candidate answer*.
That is deliberately broad. In this guide each proposer is a
`ChainOfThought` — a model that thinks step by step and emits one
candidate. We use `ChainOfThought` rather than tool-using agents for
the runnable example for one practical reason: **a mixture of models
wants maximum model diversity, and not every model supports tool
calling.** Among common local models, Gemma, Qwen-2, and DeepSeek-R1
reject tool definitions outright, while Mistral and Qwen-3 accept
them. Restricting the panel to tool-capable models would throw away
exactly the diversity we are after. Plain reasoning proposers let *any*
model join the panel.

## Building the Proposer Layer

Each proposer is identical in shape but backed by a *different*
language model:

```python
proposer = await synalinks.ChainOfThought(
    data_model=Proposal,   # each proposer emits {candidate}
    language_model=lm_a,   # a DIFFERENT model per proposer
    name="proposer_a",
)(inputs)
```

`data_model=Proposal` keeps the proposer's output narrow — a single
`candidate` field. (The `ChainOfThought` still produces a `thinking`
field too, which is what we *want* per proposer; we simply do not
forward it to the aggregator.) Give every proposer the **same input
node** so they run in parallel:

```python
inputs = synalinks.Input(data_model=Query)

p_a = await build_proposer(lm_a, "proposer_a")(inputs)
p_b = await build_proposer(lm_b, "proposer_b")(inputs)
p_c = await build_proposer(lm_c, "proposer_c")(inputs)
```

## Merging and Aggregating

Merge the original query with every proposal, then hand the bundle to
the aggregator:

```python
merged = inputs | p_a | p_b | p_c
# all present -> {query, candidate, candidate_1, candidate_2}

final = await synalinks.ChainOfThought(
    data_model=Answer,
    language_model=aggregator_lm,
    instructions=(
        "You are given a question and several candidate answers from "
        "different models. They may disagree. Reason about which parts "
        "are correct, reconcile them, and produce one final answer."
    ),
)(merged)
```

### Resilient Merging: Why `|`, Not `+`

A proposer can *fail* — a model is down, a request times out, an
output is rejected — and in Synalinks a failed branch surfaces as
`None`. How you merge decides what that one failure does to the whole
panel:

| Operator | All proposers OK | One proposer is `None` |
|----------|------------------|-------------------------|
| `+` (concat) | merges | **raises** — one dead proposer crashes the run |
| `&` (and) | merges | bundle becomes `None` — the run *degrades* to nothing |
| `\\|` (or) | merges | the `None` is **dropped**, survivors are kept |

`+` is the wrong choice for a panel: a single flaky proposer takes the
whole program down with an exception. `|` is the resilient default —
it **keeps going with whatever proposers did succeed**. `inputs | p_a |
p_b | p_c` keeps the query (always present) merged with every survivor
and silently drops the failures, so the aggregator still runs on a
partial panel. A mixture of models is supposed to be *robust through
redundancy*; `|` is what delivers that — losing one of three models
costs you one candidate, not the whole answer.

If instead you want a strict "all proposers reported, or nothing" gate,
use `&` (logical and): it merges only when every proposer succeeded and
otherwise yields `None` (the aggregator is skipped and the program
returns `None`, which you can detect and handle). Reach for `&` only
when a partial panel is *worse* than no answer; for the usual case,
`|` is the one you want.

One wart worth knowing: the *built* schema of a `|` chain advertises
only the first operand's fields (`{query}` here), because logical-or
cannot know at build time which proposals will be present. The data
still flows in full at **runtime** — the aggregator reads the actual
input *values*, not the declared schema (it uses the default
`use_inputs_schema=False`), so it sees every surviving candidate and
the panel works as intended. Don't be surprised, though, if
`plot_program` shows only `query` on the merge node. (`&` and `+` do
report the full merged schema at build time, since they assume all
operands are present.)

Two things make the aggregator work:

1. **It sees the original question.** We merge `inputs` in first, so
   the aggregator can check the candidates against what was actually
   asked instead of trusting them blindly.
2. **It is told the candidates may disagree.** The `instructions`
   frame the task as *reconciliation*, not *summarisation*. That
   nudges the model to resolve conflicts rather than average them into
   mush. Using `ChainOfThought` gives it a `thinking` field to work
   the disagreement through before committing.

## Giving Proposers Tools

Because a proposer is just a module, you can upgrade any of them to a
full agent from [Guide 6](https://synalinks.github.io/synalinks/guides/Agents/) — same input node, same `Proposal`
output, so the rest of the program is unchanged:

```python
proposer = await synalinks.FunctionCallingAgent(
    data_model=Proposal,
    language_model=tool_capable_lm,      # must support function calling!
    tools=[calculator_tool],
    autonomous=True,
    max_iterations=5,
    return_inputs_with_trajectory=False, # emit only the proposal, not the trace
    name="proposer_a",
)(inputs)
```

Two cautions when you do this:

- **The model must support tool calling.** A `FunctionCallingAgent`
  on a model that does not (e.g. Gemma) fails every step and returns
  an empty proposal — a dead branch in your panel.
- **Set `return_inputs_with_trajectory=False`.** The default (`True`,
  from [Guide 6](https://synalinks.github.io/synalinks/guides/Agents/)) attaches each agent's full trajectory *and* a
  copy of the original query to its output. Merging three of those
  produces duplicated queries and trajectories — noise the aggregator
  does not need. Turn it off so each proposer returns just its
  `candidate`.

You can freely mix the two: some proposers plain reasoners, some
tool-using agents. The aggregator does not care where a candidate came
from.

## Stacking Layers (Deep MoA)

A single proposer layer is already useful. The original Mixture-of-
Agents work goes further: it **stacks** layers. The proposals from
layer 1 become *context* for layer 2's proposers, which refine them;
their outputs feed layer 3, and so on, with a final aggregator at the
top. Each layer gets to react to the previous layer's collective
output.

You do not need new machinery for this — it is the same two moves
(parallel proposers, then `+`) repeated. A second layer simply takes
the merged bundle as its input instead of the raw query:

```python
# Layer 1: propose from the raw query
layer1 = inputs | p_a | p_b | p_c

# Layer 2: each refiner reads the query AND every layer-1 proposal
r_a = await build_refiner(lm_a)(layer1)
r_b = await build_refiner(lm_b)(layer1)
layer2 = inputs | r_a | r_b

# Final aggregator on top of layer 2
final = await aggregator(layer2)
```

More layers cost more calls for diminishing returns; two or three is
usually plenty. Start with one layer, measure, and only deepen if the
aggregator is still being dragged down by weak proposals.

## Complete Example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Query(synalinks.DataModel):
    \"\"\"User request.\"\"\"
    query: str = synalinks.Field(description="User request")

class Proposal(synalinks.DataModel):
    \"\"\"One model's candidate answer.\"\"\"
    candidate: str = synalinks.Field(description="This model's answer")

class Answer(synalinks.DataModel):
    \"\"\"Final reconciled answer.\"\"\"
    answer: str = synalinks.Field(description="Final answer to the user")

async def main():
    load_dotenv()
    synalinks.clear_session()

    # Three DIFFERENT models -> decorrelated errors
    proposer_models = [
        ("proposer_gemma", synalinks.LanguageModel(model="ollama/gemma:latest")),
        ("proposer_mistral", synalinks.LanguageModel(model="ollama/mistral:latest")),
        ("proposer_qwen", synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")),
    ]
    aggregator_lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    inputs = synalinks.Input(data_model=Query)

    # Proposer layer: all share `inputs` -> run in parallel
    proposals = []
    for name, lm in proposer_models:
        p = await synalinks.ChainOfThought(
            data_model=Proposal,
            language_model=lm,
            name=name,
        )(inputs)
        proposals.append(p)

    # Resilient-merge query + every proposal into one bundle. Using | (not +)
    # means a failed proposer is dropped, not fatal: the aggregator still
    # runs on whatever proposers survived.
    merged = inputs
    for p in proposals:
        merged = merged | p

    # Aggregator reconciles the candidates
    final = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=aggregator_lm,
        instructions=(
            "You are given a question and several candidate answers from "
            "different models. They may disagree. Reason about which parts "
            "are correct, reconcile them, and produce one final answer."
        ),
        name="aggregator",
    )(merged)

    program = synalinks.Program(
        inputs=inputs,
        outputs=final,
        name="mixture_of_models",
        description="Several models propose, one aggregator reconciles",
    )

    result = await program(Query(query=(
        "A bookstore sold 23 books on Monday, three times as many on "
        "Tuesday, and 17 fewer on Wednesday than Tuesday. How many books "
        "did it sell over the three days?"
    )))
    print(f"Final answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Expected output (one run; the exact wording varies by model and run):

```
Final answer: The bookstore sold a total of 144 books over the
three days.
```

(23 on Monday, 69 on Tuesday — three times Monday — and 52 on
Wednesday — 17 fewer than Tuesday — sum to 144.)

The interesting part is not the final number but *how the proposers
disagreed on the way there*. With small open-weight models you will
typically see at least one proposer slip — miscomputing Tuesday, or
forgetting to subtract on Wednesday — while the others get it right.
The aggregator, seeing two candidates agree on `144` and one dissent,
sides with the majority. That is the whole value of the pattern: it
turns three unreliable answers into one more-reliable answer, without
any single model having to be trustworthy on its own.

## Take-Home Summary

- A **mixture of models** is two layers: several **proposers** answer
  the same question in parallel, and one **aggregator** reconciles
  their proposals into a final answer.
- **Diversity does the work.** Use *different* models as proposers so
  their errors decorrelate; the aggregator resolves the disagreement.
- **A proposer is any module.** Reasoning `ChainOfThought`s keep the
  panel open to every model (tool calling is not universally
  supported); swap in a `FunctionCallingAgent` when a proposer needs
  tools.
- **Parallelism is automatic.** Proposers that share the same input
  node run concurrently — wall-clock cost is one slow proposer plus
  the aggregator, not the sum.
- **Merge with `|`, not `+`.** `|` auto-suffixes colliding field names
  (`candidate`, `candidate_1`, ...) just like `+`, but a *failed*
  proposer (a `None` branch) is dropped and the survivors are kept —
  `+` would crash the whole panel and `&` would collapse it to `None`.
  `|` is what makes the panel robust through redundancy. Merge the
  query in too, so the aggregator can check the candidates.
- **Stack layers** for a deep MoA — feed the merged proposals back in
  as context — but only deepen if one layer is not enough.

## API References

- [ChainOfThought](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/ChainOfThought%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [Merging Modules (Or `|`, And `&`, Concat `+`)](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Merging%20Modules/)
- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Query(synalinks.DataModel):
    """User request."""

    query: str = synalinks.Field(description="User request")


class Proposal(synalinks.DataModel):
    """One model's candidate answer."""

    candidate: str = synalinks.Field(description="This model's answer")


class Answer(synalinks.DataModel):
    """Final reconciled answer."""

    answer: str = synalinks.Field(description="Final answer to the user")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # Log every module call (proposers, merge, aggregator) to the console.
    synalinks.enable_logging()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_29_mixture_of_models",
    # )

    # Three DIFFERENT models -> decorrelated errors. Any model works as a
    # proposer because ChainOfThought needs no tool-calling support.
    proposer_models = [
        ("proposer_gemma", synalinks.LanguageModel(model="ollama/gemma:latest")),
        ("proposer_mistral", synalinks.LanguageModel(model="ollama/mistral:latest")),
        ("proposer_qwen", synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")),
    ]
    aggregator_lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    # -------------------------------------------------------------------------
    # Build the mixture-of-models program (functional API)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Mixture of Models: proposers -> aggregator")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Proposer layer: all share `inputs`, so they run in parallel.
    proposals = []
    for name, lm in proposer_models:
        proposal = await synalinks.ChainOfThought(
            data_model=Proposal,
            language_model=lm,
            name=name,
        )(inputs)
        proposals.append(proposal)

    # Resilient-merge the original query with every proposal into one
    # bundle. We use | (not +) so a failed proposer is dropped instead of
    # crashing the run: the aggregator still runs on whatever survived.
    # Colliding field names are auto-suffixed: candidate, candidate_1, ...
    merged = inputs
    for proposal in proposals:
        merged = merged | proposal

    # Aggregator reconciles the candidates into one final answer.
    final = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=aggregator_lm,
        instructions=(
            "You are given a question and several candidate answers from "
            "different models. They may disagree. Reason about which parts "
            "are correct, reconcile them, and produce one final answer."
        ),
        name="aggregator",
    )(merged)

    program = synalinks.Program(
        inputs=inputs,
        outputs=final,
        name="mixture_of_models",
        description="Several models propose, one aggregator reconciles",
    )
    program.summary()

    # -------------------------------------------------------------------------
    # Run it on a multi-step word problem
    # -------------------------------------------------------------------------
    question = (
        "A bookstore sold 23 books on Monday, three times as many on "
        "Tuesday, and 17 fewer on Wednesday than Tuesday. How many books "
        "did it sell over the three days?"
    )
    print(f"\nQuestion: {question}")

    result = await program(Query(query=question))
    print(f"\nFinal answer: {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
