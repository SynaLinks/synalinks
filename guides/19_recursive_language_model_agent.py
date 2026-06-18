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

primary = synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")

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
primary = synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")
cheap   = synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")

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
  produces a `SyntaxError` observation, and the LM has to
  regenerate. Repeated parse failures eat into `max_iterations`
  without producing any answer.

## Complete example

A real task: **incident triage over a long multi-service log**. The input
pairs a short `question` (fully visible to the primary LM) with a 2000-line
`log` (only previewed). The question states only the *goal*; the agent decides
*how*. Answering needs both halves of the design — code to find the first error
line deterministically (structure), and the sub-LM to explain it (meaning):

```python
import asyncio
from dotenv import load_dotenv
import synalinks

class LogReport(synalinks.DataModel):
    question: str = synalinks.Field(description="The triage question to answer")
    log: str = synalinks.Field(description="The full application log")

class Incident(synalinks.DataModel):
    first_error_line: str = synalinks.Field(
        description="The first line in the log containing 'ERROR', copied verbatim")
    explanation: str = synalinks.Field(
        description="One-sentence, plain-English explanation of that error")

async def main():
    load_dotenv()
    synalinks.clear_session()

    primary = synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")
    cheap   = synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")

    inputs = synalinks.Input(data_model=LogReport)
    outputs = await synalinks.RLM(
        data_model=Incident,
        language_model=primary,
        sub_language_model=cheap,
        max_iterations=5,
        max_llm_calls=8,
    )(inputs)
    agent = synalinks.Program(inputs=inputs, outputs=outputs, name="rlm_triage")

    log = open("app.log").read()
    question = (
        "An incident took place. Find its root cause — the first error in this "
        "chronological log — and explain in one sentence what went wrong."
    )
    result = await agent(LogReport(question=question, log=log))
    print(result.prettify_json())

if __name__ == "__main__":
    asyncio.run(main())
```

Inside the `agent(...)` call the primary LM sees only the *preview* of the
log. From the goal alone it works out the approach: scan the full log inside
the sandbox, keep the ERROR lines, take the first (the root cause — `db` here,
with `api` / `worker` cascading off it), hand that one line to the sub-LM for a
human-readable explanation, and `submit` the structured `Incident`. The
structural part of the answer (`first_error_line`) is copied verbatim by
deterministic code, not guessed — so it is correct regardless of model size,
and a cheap sub-LM only ever phrases a single short line.

## Expected output (representative)

```text
Log: 2000 lines, 116878 characters
(ground truth: first failure in 'db' at 2026-01-15T10:00:00)

============================================================
Example 1: incident triage via RLM
============================================================

Example 1 result:
  first_error_line: 2026-01-15T10:00:00 ERROR db: connection pool exhausted (max=20): all connections checked out, rejecting new checkouts
  explanation:      The database exhausted its connection pool (all 20 connections were in use), so it rejected new checkouts and dependent services began to fail.
  first ERROR line correct? ✅

============================================================
Example 2: walk the trajectory
============================================================

Trajectory has 4 messages:
  [00] assistant {"code": "errors = [l for l in inputs['log'].splitlines() if ' ERROR '...
  [01] tool      {'stdout': "2026-01-15T10:00:00 ERROR db: connection pool exhausted...
  [02] assistant {"code": "first = errors[0]; s = llm_query(prompt=f'Explain: {first}')...
  [03] tool      {'submit': 'accepted'}
```

Output is *stochastic* (varies from run to run) and depends heavily on the
models you pick. The **first_error_line is deterministic** — it is copied from
the log by code, not the model — so a model that actually runs the scan gets it
exactly right every time; only the phrasing of `explanation` varies. A capable
primary model closes the loop in two or three turns. A small local model
(`ollama/qwen3:8b`) is slower and less reliable here: it can over-think each
step or guess from the preview instead of scanning — which is exactly why the
deterministic `✅ / ❌` check above is worth printing. The lesson generalizes:
keep the *structural* part of the answer in code so it stays correct
independent of model strength, and spend the LM only on what needs language.

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
from datetime import datetime
from datetime import timedelta

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
#
# A real RLM task: incident triage over a long, multi-service application log.
# The input pairs a short `question` (the goal only) with a huge `log` (only
# previewed). The answer is *structured* and needs both halves of the RLM
# design:
#   - CODE (structure): scan the lines, keep the ERRORs, take the first one
#     (the log is chronological). Deterministic — never a sub-LM call.
#   - sub-LM (meaning): turn that one terse error line into a plain-English
#     explanation.
# =============================================================================


class LogReport(synalinks.DataModel):
    """An incident-triage request over a long application log."""

    question: str = synalinks.Field(description="The triage question to answer")
    log: str = synalinks.Field(
        description="The full application log, one timestamped event per line"
    )


class Incident(synalinks.DataModel):
    """The triaged root-cause incident."""

    first_error_line: str = synalinks.Field(
        description="The first line in the log containing 'ERROR', copied verbatim"
    )
    explanation: str = synalinks.Field(
        description="One-sentence, plain-English explanation of that error"
    )


# =============================================================================
# Synthetic long input — a realistic multi-service log with one buried incident
#
# Thousands of INFO/DEBUG lines across five services, then — ~60% in — a real
# incident: `db` fails FIRST (connection pool exhausted), and `api` / `worker`
# cascade off it a few seconds later. The earliest ERROR by timestamp is the
# db one, so the root cause is deterministic and checkable. The primary LM
# never sees this text; it only gets the InputsSummary preview.
# =============================================================================


def build_incident_log(total_lines: int = 2000):
    """Return ``(log_text, ground_truth)`` for a planted db-root-cause incident."""
    base = datetime(2026, 1, 15, 9, 0, 0)
    services = ["auth", "api", "cache", "worker", "db"]
    info = [
        "request handled in {ms}ms",
        "health check ok",
        "cache hit ratio {pct}%",
        "scheduled job completed",
        "connection established",
    ]

    def line(i, level, service, msg):
        # Clean, easy-to-parse format: `<iso-ts> <LEVEL> <service>: <msg>`.
        # ISO timestamps sort lexicographically, so "earliest" is just `min`.
        ts = (base + timedelta(seconds=3 * i)).strftime("%Y-%m-%dT%H:%M:%S")
        return f"{ts} {level} {service}: {msg}"

    lines = []
    for i in range(total_lines):
        service = services[i % len(services)]
        msg = info[i % len(info)].format(ms=10 + (i % 90), pct=80 + (i % 20))
        level = "DEBUG" if i % 7 == 0 else "INFO"
        lines.append(line(i, level, service, msg))

    # Plant the incident ~60% in: db fails first (the root cause), then api and
    # worker cascade off it over the next few seconds (downstream effects).
    at = total_lines * 6 // 10
    root_line = line(
        at,
        "ERROR",
        "db",
        "connection pool exhausted (max=20): all connections checked out, "
        "rejecting new checkouts",
    )
    lines[at] = root_line
    cascade = [
        ("api", "upstream query timed out after 5000ms waiting for a db connection"),
        ("worker", "job retry failed: could not acquire a db connection"),
        ("api", "returned 503 to client: database unavailable"),
    ]
    for k, (svc, msg) in enumerate(cascade, start=1):
        lines[at + k] = line(at + k, "ERROR", svc, msg)

    ground_truth = {
        "service": "db",
        "timestamp": (base + timedelta(seconds=3 * at)).strftime("%Y-%m-%dT%H:%M:%S"),
        "line": root_line,
    }
    return "\n".join(lines), ground_truth


def show_incident(result, truth):
    """Print the structured answer and check the deterministic part.

    ``first_error_line`` is copied verbatim from the log by code, so it must
    match the planted root-cause line exactly — a hard, model-independent check.
    """
    first_line = str(result.get("first_error_line", ""))
    correct = truth["line"] in first_line
    print(f"  first_error_line: {result.get('first_error_line')}")
    print(f"  explanation:      {result.get('explanation')}")
    print(f"  first ERROR line correct? {'✅' if correct else '❌'}")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()
    synalinks.enable_logging()

    # A capable primary LM for orchestration; a cheap one for sub-queries.
    # Both default to the same model if you only pass `language_model=`.
    primary = synalinks.LanguageModel(
        model="ollama/qwen3:8b", reasoning_effort="disable"
    )
    cheap = synalinks.LanguageModel(model="ollama/qwen3:8b", reasoning_effort="disable")

    # synalinks.enable_observability(
    #     project_name="recursive_language_model_agent_guide",
    # )

    log, truth = build_incident_log(total_lines=2000)
    question = (
        "An incident took place. Find its root cause — the first error in this "
        "chronological log — and explain in one sentence what went wrong."
    )
    print(f"Log: {len(log.splitlines())} lines, {len(log)} characters")
    print(
        f"(ground truth: first failure in '{truth['service']}' at {truth['timestamp']})\n"
    )

    # -------------------------------------------------------------------------
    # Example 1: Full triage — the primary LM never sees the 2000-line log,
    # only its preview. It writes code to find the earliest ERROR (structure)
    # and delegates the explanation to the sub-LM (meaning).
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: incident triage via RLM")
    print("=" * 60)

    inputs = synalinks.Input(data_model=LogReport)
    outputs = await synalinks.RLM(
        data_model=Incident,
        language_model=primary,
        sub_language_model=cheap,
        max_iterations=5,
        max_llm_calls=8,
    )(inputs)
    agent = synalinks.Program(inputs=inputs, outputs=outputs, name="rlm_triage")
    agent.summary()

    result = await agent(LogReport(question=question, log=log))
    print("\nExample 1 result:")
    show_incident(result, truth)

    # -------------------------------------------------------------------------
    # Example 2: Inspect the trajectory — every assistant code block and every
    # observation is recorded when return_inputs_with_trajectory=True (the
    # default). Useful for seeing which snippets the agent ran, which sub-LM
    # calls it dispatched, and how it converged on (or missed) the answer.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: walk the trajectory")
    print("=" * 60)

    messages = result.get("messages") or []
    print(f"\nTrajectory has {len(messages)} messages:")
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content") or ""
        content = (content if isinstance(content, str) else str(content)).strip()
        head = content[:120].replace("\n", " ")
        print(f"  [{i:02d}] {role:9s} {head}{'…' if len(content) > 120 else ''}")


if __name__ == "__main__":
    asyncio.run(main())
