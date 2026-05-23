# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Recursive Language Model Agent

This is the last guide in the series, and we now build on everything
that came before. [Guide 6](https://synalinks.github.io/synalinks/guides/Agents/) introduced agents — a loop that decides,
acts, observes, and repeats. This guide presents a more powerful
agent design: one that can **spawn a smaller helper agent for
sub-tasks**, the way a recursive function in CS101 calls itself on
a smaller problem.

## What Is a Recursive LM Agent?

Imagine an agent that, when faced with too much work, can hand off a
sub-task to a smaller helper agent. Each helper has its own budget;
the parent waits for the helper's answer and incorporates it. That
is the intuition; the rest of this section makes it precise.

A **recursive language model agent** (RLM agent, "RLM" for short) is
an agent A whose forward computation may invoke *another* language
model M' on a strictly smaller subproblem before returning.
Formally, A is a function:

    A : (query, inputs) -> answer

In English: A takes a question and some inputs, and it returns an
answer. The implementation is a loop. On each turn A calls a single
tool, `run_python_code(code=...)`, with a small piece of
Python code (a "**snippet**"). The snippet runs in a **sandbox** — a
restricted Python environment — that exposes two helpers, called the
**recursive primitives** `llm_query` and `llm_query_batched`, which
forward sub-problems to a second language model M'. These primitives
(and `submit`) are callables *inside* the sandbox, not separate tools:
A reaches them only from the code it passes to `run_python_code`.
Whatever the snippet prints or returns becomes an **observation** — a
log line A can read on its next turn — and the loop continues.

The recursion is **value-recursive**: A does not call itself on the
same inputs. Instead, on each turn, A delegates a *piece* of the
inputs to M' — a slice of text, a regex match, a single paragraph.
("Projection" in this context just means "a smaller view of the
inputs.")

How do we guarantee the loop stops? Termination is enforced
externally by three **decreasing measures**. A decreasing measure
is a number that must strictly shrink on every step; once it hits
zero, the loop is forced to end. RLM uses three of them at once:

1. **iteration count** — how many turns A is allowed,
2. **sub-LM call budget** — how many times A can call M',
3. **per-snippet wall clock** — how long one snippet may run.

Because each measure decreases on every step, the whole thing must
stop eventually.

The construction follows
[Recursive Language Models (Zhang, Kraska, Khattab — 2025)](https://arxiv.org/abs/2512.24601).

## Why Bother With This Design?

Suppose the input is very large. Let *n* be the size of the input in
**tokens** (a token is the model's unit of text, roughly a short
word fragment) and *k* the size of the question. When *n* is huge —
a whole book, a long log file — three costs grow with *n*:

1. **Prompt cost.** The price of one LM call grows roughly linearly
   in prompt length. A 200,000-token prompt is mostly tokens the
   model never actually needs.
2. **Latency.** Most providers cap throughput per input token, so
   waiting time grows with *n*, not with the size of the answer.
3. **Recall.** Past a model-specific threshold, the model gets
   *worse* at finding the right fact inside its own context — the
   well-known "lost in the middle" effect — even when the fact is
   present.

Two standard fixes you might have already met:

- **Stuffing.** Just paste the full input into the prompt. Simple,
  but pays all three costs every time.
- **Offline retrieval.** Pre-process the input into chunks, store
  them in a vector index, and retrieve the top few at query time.
  This is the RAG pipeline from [Guide 7](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/). It is cheap at query
  time, but the chunk size and the embedding model are picked
  *before the question is known*, which puts a ceiling on how good
  retrieval can ever be.

**RLM is a third option.** The primary LM (the orchestrator) is
*never* given the full input. It is given an `InputsSummary` — a
list of field names, types, sizes, and short previews. The full
value lives inside a persistent Python sandbox, bound to the
variable `inputs[field]`. On each turn the primary LM writes a
Python snippet that decides — conditional on the question *and* on
what it has already learned — which slice of the input is worth a
sub-LM call.

The key win: the chunking is chosen **after** the question is
known, by code that the model itself wrote.

```mermaid
flowchart TD
    A["Long input and query"] --> S["InputsSummary (previews + sizes)"]
    S --> P["Primary LM"]
    P --> C["Python snippet"]
    C --> X["Sandbox: inputs[field] holds full value"]
    X --> Q{"semantic work?"}
    Q -- "yes" --> L["llm_query / llm_query_batched"]
    Q -- "no"  --> R["Pure code (regex, slicing, set ops)"]
    L --> O["Observation"]
    R --> O
    O --> P
    P -- "done" --> SU["submit(result)"]
```

Recall: an **invariant** is a property that holds at every step of
the loop. Here, the invariant is: the primary LM's context contains
only the summary, the catalog of available tools, and the
accumulated **trajectory** (the running log of past snippets and
observations). Each sub-LM call sees only the small piece of text
the snippet explicitly hands it. A hard cap on the number of sub-LM
calls per `agent(...)` invocation bounds the total cost.

## A Minimal Example

```python
import synalinks

class Doc(synalinks.DataModel):
    text: str

class Answer(synalinks.DataModel):
    answer: str

primary = synalinks.LanguageModel(model="ollama/qwen3:8b")

inputs = synalinks.Input(data_model=Doc)
outputs = await synalinks.RLM(
    data_model=Answer,
    language_model=primary,
)(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs)

long_text = open("book.txt").read()
result = await agent(Doc(text=long_text))
print(result.prettify_json())
```

The whole `book.txt` lives in the sandbox; the primary LM is given
only the field's preview and its length.

## Inputs as an External Environment

Think of the input as a library shelf in another room. The model
holds an index card listing what books are on the shelf and how
long each one is; the books themselves stay on the shelf. Whenever
the model needs to actually read a book, it writes a snippet that
fetches the right passage by name.

More precisely: let *x* be the user input. The sandbox sets up a
global Python dictionary called `inputs` so that `inputs[field]` is
the **full**, untruncated value of `x[field]`. The primary LM
never sees *x*. It sees only an `InputsSummary`:

```text
fields:
  - name: text
    type: str
    size: 482917         # full character length
    preview: "Chapter 1...  (first 200 chars)"
    truncated: true
```

Two consequences of this design:

- **Read through the binding.** Snippets must say `inputs[field]`,
  not retype text out of the preview. The preview is a *lossy*
  view (only the first few hundred characters); the binding
  `inputs[field]` is the real, full value. Copying from the
  preview silently smuggles in truncated data.
- **Aggregate in code, then query for meaning.** Structural
  questions ("where," "how many," "what shape?") have
  deterministic answers and should be handled by Python: regular
  expressions (regex — pattern matching on strings), slicing
  (taking sub-ranges), set operations. Semantic questions ("what
  does this mean?", "is this about X?") should be answered by
  `llm_query` on spans the code has already isolated.

This inverts the usual pipeline: the model is the orchestrator and
stays small; the data is the environment and stays external.

## The Two Recursive Primitives

A **primitive** here means a built-in helper function the snippet
can call. Besides `submit` (which returns the final answer), the
sandbox exposes two **asynchronous** primitives — "asynchronous"
meaning they use Python's `await` so several calls can run
concurrently.

### `llm_query(prompt)`

```python
async def main():
    out = await llm_query(prompt="Summarize this paragraph: ...")
    print(out["result"])

asyncio.run(main())
```

Sends one prompt to the sub-LM (which model that is can be
configured — see below) and returns a dict `{"result": <text>}`.
Use it for **semantic** work on a span the code has already pulled
out: classify a paragraph, summarize a span, extract names,
reformat a quote. Each call decrements the shared budget by one.

### `llm_query_batched(prompts)`

```python
async def main():
    out = await llm_query_batched(prompts=[
        "Is this paragraph about finance? <p1>",
        "Is this paragraph about finance? <p2>",
        "Is this paragraph about finance? <p3>",
    ])
    finance = [p for p, label in zip(paragraphs, out["result"]) if "yes" in label.lower()]

asyncio.run(main())
```

Same idea as `llm_query`, but sends a *list* of prompts in
parallel using `asyncio.gather` (a Python construct that fires off
several async calls at once and waits for all of them). It returns
`{"result": [<text-or-error>, ...]}` in the same order as the
input prompts.

**Prefer the batched form** over a Python `for` loop of single
`llm_query` calls. A loop calls the sub-LM *m* times sequentially
and stacks up *m* latencies one after another — it will run out of
the per-snippet timeout long before it runs out of budget. If a
single prompt in the batch fails, its slot contains a string of
the form `"[error] <ExceptionType>: <message>"`, so check for that
prefix before treating an element as real data.

### Budget invariant

Picture a small allowance: every sub-LM call costs one coin out of
a shared jar. The jar must never go below zero.

Both primitives draw from a single counter *c*, initialized to
`max_llm_calls` (default 50). On a successful single call,
*c := c − 1*; on a successful batch of size *m*,
*c := c − m*. Two short-circuit rules preserve the invariant
*c ≥ 0* (the jar is never overdrawn):

- A `llm_query` call when c == 0 returns immediately with `error`
  set, and does not decrement c.
- A `llm_query_batched(prompts)` call where c < len(prompts) returns
  immediately, all-or-nothing — it never partially fulfils a batch.

In plain words: if you cannot afford the whole batch, you get
nothing — the budget is never overshot.

The counter is reset at the start of every `agent(...)` invocation,
so two concurrent invocations don't share a jar.

A `llm_query` short-circuit reads:

```text
{"result": "", "error": "sub-LM call budget exhausted (50/50). ..."}
```

A `llm_query_batched` short-circuit reads:

```text
{"result": [], "error": "sub-LM call budget would be exceeded: 45 + 10 > 50. ..."}
```

In both cases the model is told to fall back to code-only
aggregation. The rule of thumb: **always check `error` before
using `result`.**

## Adding Your Own Tools

`RecursiveLanguageModelAgent` accepts a `tools=` argument. You can
plug in your own helper functions — fetch a URL, query a database,
hit a search API — and the model can call them from inside its
snippets. The names `llm_query`, `llm_query_batched`, and `submit`
are reserved; every other tool is bound as a global async function
inside the sandbox and can be combined with the recursive
primitives in the same snippet.

**A note on trust.** A tool *body* runs on the host machine with
full Python privileges — filesystem, network, third-party
libraries. The sandbox restricts only the snippet itself; once
execution crosses into a host-side tool, the sandbox no longer
protects you. In practice, each tool is the **trust boundary**:
validate the tool's arguments, scope what it is allowed to touch,
and do not expose destructive operations unless your deployment
really needs them.

Decorate each tool with
`@synalinks.saving.register_synalinks_serializable()` so that it
survives `get_config` / `from_config` round-trips (saving and
loading the agent):

```python
@synalinks.saving.register_synalinks_serializable()
async def fetch_url(url: str) -> dict:
    \"\"\"Fetch a URL and return its body.

    Args:
        url (str): the URL to GET.
    \"\"\"
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return {"status": resp.status_code, "body": resp.text}

agent = synalinks.RLM(
    data_model=Answer,
    language_model=primary,
    sub_language_model=cheap,
    tools=[synalinks.Tool(fetch_url)],
)
```

Inside a single snippet the LM can now mix real-world side effects
with recursive sub-LM work: fetch a page, hand the body to
`llm_query` for classification, then `submit` the result. Treat
every tool the way you'd treat a public web endpoint — validate
inputs, restrict filesystem and network access, and don't expose
anything destructive unless the deployment really requires it.

## Choosing the Sub-LM

`sub_language_model` defaults to whatever `language_model` you
passed, so by default the **same** model plays both roles. That is
convenient for prototyping but almost always wasteful in
production, because the two jobs have very different shapes:

- The **primary** LM plans, writes code, and formats the final
  structured answer. It needs to be capable, but you only call it
  a handful of times per run.
- The **sub-LM** answers narrow, local questions — "summarize
  this paragraph", "is this paragraph about X?". A small, cheap
  model is usually plenty, and you call it many times per run.

Pass a separate `sub_language_model` to exploit the asymmetry:

```python
primary = synalinks.LanguageModel(model="ollama/qwen3:8b")
cheap   = synalinks.LanguageModel(model="ollama/qwen3:8b")

agent = synalinks.RLM(
    data_model=Answer,
    language_model=primary,
    sub_language_model=cheap,
)
```

A typical RLM run makes dozens of sub-LM calls and just one or
two primary-LM turns, so total cost is dominated by the per-call
cost of the sub-LM. In short: picking the sub-LM is where most of
your money is spent.

## Working Rules (in the system prompt)

The default instructions give the model a five-item checklist for
how to behave in the loop. Each rule corresponds to a real failure
mode this design has hit in practice:

1. **Explore before slicing.** On the very first turn, print
   sample values, lengths, types, and shapes of `inputs[field]`.
   A cheap probe (cost: zero sub-LM calls) prevents wasted sub-LM
   calls on the wrong field.
2. **Code for structure; `llm_query` for meaning.** Regex,
   slicing, and set operations answer "where" and "how many"
   deterministically and for free. Reserve the sub-LM for "what
   does this mean."
3. **Do not retype.** Always re-access values via
   `inputs[field]`, never by copying text out of the preview. The
   preview is truncated; copies drift away from the truth.
4. **Verify before submitting.** If results look empty, zero, or
   misshapen, spend an extra turn inspecting them before calling
   `submit`.
5. **`submit` is terminal.** A snippet runs to completion (so a
   `print` near `submit` does execute), but once `submit`
   succeeds the loop ends — that final captured print is never
   read by the model. Lesson: inspect on one turn, then submit on
   the next.

You can replace these defaults via the `instructions=` argument if
your task wants a different style, but the defaults are tuned for
the long-input, sparse-query case.

## Default settings

| Setting          | Default | Rationale                                                                |
|------------------|---------|--------------------------------------------------------------------------|
| `max_iterations` | 20      | Bounds the explore-carve-query-aggregate-submit loop.                    |
| `timeout`        | 60 s    | One `llm_query_batched(20)` snippet routinely needs more than 5 s.       |
| `max_llm_calls`  | 50      | Hard ceiling on sub-LM calls per run; surfaced to the LM via `error`.    |

Override any of them if the workload's measure is known.

## RLM versus `FunctionCallingAgent`

`FunctionCallingAgent` (FCA) is the more common pattern: the LM
emits JSON tool calls and gets back JSON results, with the full
input sitting in the prompt. The table below compares the two:

| Aspect              | FunctionCallingAgent              | RecursiveLanguageModelAgent                |
|---------------------|-----------------------------------|--------------------------------------------|
| Action shape        | JSON tool call                    | Python snippet executed in a sandbox       |
| Primary LM context  | Full input + tool catalog         | `InputsSummary` + tool catalog             |
| Long-input handling | Pass through prompt               | External environment via `inputs[field]`   |
| Recursive sub-LM    | None                              | `llm_query` / `llm_query_batched`          |
| Cost lever          | Choose one model                  | Choose a cheap `sub_language_model`        |
| Suited to           | Discrete, well-typed actions      | Long inputs, sparse / semantic queries     |

Prefer `RecursiveLanguageModelAgent` when the input dwarfs the query
and the relevant slice cannot be selected in advance: long documents,
large logs, large lists where a few items decide the answer.

## Common Traps

These are the same pitfalls you would worry about with ordinary
recursive functions in a CS101 course — base cases, termination,
type mismatches — but applied to an agent. (Recall: a **base
case** is the condition under which a recursive function stops
recursing.)

- **Infinite recursion in disguise.** The agent itself never
  recurses on its own inputs, but a *tool body* can — for
  example, a fetch tool whose next URL is computed from the
  previous sub-LM answer. Without an explicit decreasing measure
  (recursion depth, byte budget, set of already-visited URLs),
  such tools loop forever. Treat tool-induced recursion the same
  way you would treat factorial without a base case: insist on an
  argument that strictly shrinks.
- **Unbounded depth.** `max_iterations` bounds primary-LM turns
  and `max_llm_calls` bounds sub-LM calls — but neither bounds
  the depth of nested tool calls *inside one snippet*. Check that
  a snippet cannot build arbitrarily deep call chains.
- **Schema drift.** The sub-LM returns plain strings, not
  structured data. If you aggregate those strings without
  validating them, a malformed answer slips into the final
  `submit` payload and the payload then fails schema validation
  (validation = checking the output matches the declared
  `DataModel`). Always parse and filter sub-LM outputs before
  aggregating.
- **Preview retyping.** Copying values out of the `InputsSummary`
  preview into a sub-LM prompt silently smuggles truncated data
  past the model. The binding `inputs[field]` is the only correct
  read path.
- **Sandbox syntax errors.** The sandbox parses each snippet
  before running it. An invalid f-string or an unbalanced brace
  produces a `MontySyntaxError` observation, and the LM has to
  regenerate. Repeated parse failures eat into `max_iterations`
  without producing any answer.

## Complete example

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class Doc(synalinks.DataModel):
    text: str = synalinks.Field(description="The document to analyze")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer to the user")

async def main():
    load_dotenv()
    synalinks.clear_session()

    primary = synalinks.LanguageModel(model="ollama/qwen3:8b")
    cheap   = synalinks.LanguageModel(model="ollama/qwen3:8b")

    inputs = synalinks.Input(data_model=Doc)
    outputs = await synalinks.RLM(
        data_model=Answer,
        language_model=primary,
        sub_language_model=cheap,
        max_iterations=4,
        max_llm_calls=6,
    )(inputs)
    agent = synalinks.Program(inputs=inputs, outputs=outputs, name="rlm_qa")

    long_text = open("book.txt").read()
    result = await agent(Doc(text=long_text))
    print(result.prettify_json())

if __name__ == "__main__":
    asyncio.run(main())
```

Inside the `agent(...)` call, the primary LM sees only the preview
of `book.txt`. It writes Python that scans the full text inside the
sandbox, picks out the spans worth looking at, dispatches them to
the sub-LM, aggregates the answers, and finally calls `submit`.

## Expected output (representative)

```text
Haystack length: 8122 characters

============================================================
Example 1: needle-in-haystack via RLM
============================================================

Answer: The text is not truncated.

============================================================
Example 2: tight budget forces code-side aggregation
============================================================

Answer:

============================================================
Example 3: walk the trajectory
============================================================

Trajectory has 8 messages:
  [00] assistant ```python import asyncio inputs = {'fields': [{'name': 'text', 'type': 'str', 'size': 8122, 'preview': 'Paragraph 0: Lor...
  [01] tool      stdout: text Paragraph 0: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut l...
  [02] assistant ```python import asyncio inputs = {'fields': [{'name': 'text', 'type': 'str', 'size': 8122, 'preview': 'Paragraph 0: Lor...
  [03] tool      error: MontySyntaxError: unexpected EOF while parsing at byte range 1599..1599
  [04] assistant ```python import asyncio inputs = {'fields': [{'name': 'text', 'type': 'str', 'size': 8122, 'preview': 'Paragraph 0: Lor...
  [05] tool      error: MontySyntaxError: unexpected EOF while parsing at byte range 1599..1599
  [06] assistant ```python import asyncio inputs = {'fields': [{'name': 'text', 'type': 'str', 'size': 8122, 'preview': 'Paragraph 0: Lor...
  [07] tool      error: MontySyntaxError: Expected `}`, found newline at byte range 1533..1534
```

Output is *stochastic* (varies from run to run) and depends on which
models you picked. With a small local model
(`ollama/qwen3:8b`) you can see a realistic failure mode in
the trajectory above: the model repeatedly emits syntactically
invalid Python, which the sandbox rejects with `MontySyntaxError`.
Larger primary models almost always close the loop cleanly with
`submit`; this trap is genuine, not an artefact of the example.

## Take-Home Summary

- **Long input as environment.** The primary LM sees the summary;
  the full value is bound in the sandbox as `inputs[field]`.
- **Two recursive primitives.** `llm_query` and
  `llm_query_batched` each return a dict with a `result` key
  (and an `error` key when they short-circuit or when an element
  fails).
- **Single shared budget.** `max_llm_calls` bounds both
  primitives at once. Each `agent(...)` call gets a fresh
  counter; concurrent calls do not fight over one budget.
- **Choice of `sub_language_model` dominates cost.** Use a small,
  cheap model unless you have measured that you need more.
- **Working rules.** The default `instructions=` encodes five
  rules tuned for the long-input, sparse-query regime; override
  if needed.
- **Termination.** The agent stops when a snippet calls
  `submit(result={...})`; the payload is validated against your
  target `DataModel`. If `max_iterations` runs out without a
  `submit`, a final inference step formats whatever trajectory
  has accumulated into the schema. Each individual snippet is
  bounded by `timeout`.

## API References

- [RecursiveLanguageModelAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/RecursiveLanguageModelAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [Tool](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Tool%20module/)
- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Doc(synalinks.DataModel):
    """A long document to analyze."""

    text: str = synalinks.Field(description="The document text")


class Answer(synalinks.DataModel):
    """A final answer to the user query."""

    answer: str = synalinks.Field(description="The final answer to the user")


# =============================================================================
# Synthetic long input — a noisy haystack with one needle
#
# We build a long, repetitive document and hide one fact ("magic number")
# in the middle. The primary LM never sees this string; it only sees the
# InputsSummary preview. Finding the needle requires writing code that
# scans the full text in the sandbox and either uses a regex or batches
# sub-LM calls over candidate spans.
# =============================================================================


def build_haystack(needle: str, paragraphs: int = 200) -> str:
    """Return a long document with `needle` placed near the middle."""
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
    )
    body = [f"Paragraph {i}: {filler}" for i in range(paragraphs)]
    body[paragraphs // 2] = f"Paragraph {paragraphs // 2}: {needle}"
    return "\n\n".join(body)


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # A capable primary LM for orchestration; a cheap one for sub-queries.
    # Both default to the same model if you only pass `language_model=`.
    primary = synalinks.LanguageModel(model="ollama/qwen3:8b")
    cheap = synalinks.LanguageModel(model="ollama/qwen3:8b")

    # synalinks.enable_observability(
    #     project_name="recursive_language_model_agent_guide",
    # )

    haystack = build_haystack(
        needle="The magic number is 4242, please remember it.",
        paragraphs=40,
    )
    print(f"Haystack length: {len(haystack)} characters\n")

    # -------------------------------------------------------------------------
    # Example 1: Minimal RLM — primary LM never sees the full document
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: needle-in-haystack via RLM")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Doc)
    outputs = await synalinks.RLM(
        data_model=Answer,
        language_model=primary,
        sub_language_model=cheap,
        max_iterations=4,
        max_llm_calls=6,
    )(inputs)
    agent = synalinks.Program(inputs=inputs, outputs=outputs, name="rlm_needle")

    result = await agent(Doc(text=haystack))
    print(f"\nAnswer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Example 2: Same agent, tighter budgets — show the LM choosing
    # code-side aggregation when sub-LM calls are scarce
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: tight budget forces code-side aggregation")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Doc)
    outputs = await synalinks.RLM(
        data_model=Answer,
        language_model=primary,
        sub_language_model=cheap,
        max_iterations=4,
        max_llm_calls=2,  # almost no sub-LM budget — must use regex / slicing
    )(inputs)
    tight_agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="rlm_tight",
    )

    result = await tight_agent(Doc(text=haystack))
    print(f"\nAnswer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Example 3: Inspect the trajectory — every assistant code block and
    # every observation is recorded when return_inputs_with_trajectory=True
    # (the default). Useful for debugging which snippets the LM ran and
    # which sub-LM calls it dispatched.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: walk the trajectory")
    print("=" * 60)

    messages = result.get("messages") or []
    print(f"\nTrajectory has {len(messages)} messages:")
    for i, m in enumerate(messages):
        role = m.get("role")
        content = (m.get("content") or "").strip()
        head = content[:120].replace("\n", " ")
        print(f"  [{i:02d}] {role:9s} {head}{'…' if len(content) > 120 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
