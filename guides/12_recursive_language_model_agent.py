# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Recursive Language Model Agent

The `RecursiveLanguageModelAgent` (also exported as `synalinks.RLM`) is a
specialization of `CodeModeAgent` for tasks where the *input itself* is
too large or too noisy to feed straight into the language model. Instead
of packing a whole book, log dump, or scraped corpus into the primary
LM's context, the agent treats those inputs as an **external
environment**: it writes Python that programmatically slices, filters,
and aggregates the data, and recursively delegates semantic work to a
*sub-LM* on the snippets it actually cares about.

The pattern follows
[Recursive Language Models (Zhang, Kraska, Khattab — 2025)](https://arxiv.org/abs/2512.24601).

## Why Recursive?

A long context is expensive on three axes that compound:

- **Token cost** scales linearly with the prompt size; a single 200K
  prompt can outweigh a hundred small ones.
- **Latency** scales linearly too, and provider rate limits often
  bottleneck on input tokens before output tokens.
- **Accuracy regresses** past a model-specific knee — the
  "lost in the middle" effect — even when the relevant span fits.

Two patterns are common at this scale:

1. **Stuff it all in.** Hits the wall above. Easy to write, expensive
   to run, fragile on the recall side.
2. **Off-line preprocess + retrieval.** Build an index, retrieve top-k
   chunks per query, ground the LM on the chunks. Fast at query time,
   but you've committed to a chunking strategy, an embedding model, and
   a retrieval ceiling *before you know what the question is*.

RLM is the third option. The primary LM never sees the long input; it
sees only an `InputsSummary` (per-field previews and sizes). The full
value lives in the sandbox under `inputs[field]`. The LM writes code
that decides — *per query, per turn* — what slice matters, then calls a
sub-LM on that slice. No fixed chunking, no fixed retriever, no
context blowup on the primary LM.

```mermaid
flowchart TD
    A[Long input + Query] --> S[InputsSummary<br/>previews + sizes only]
    S --> P[Primary LM]
    P --> C[Python snippet]
    C --> X[Monty Sandbox<br/>inputs is dict, full value]
    X --> Q{semantic work?}
    Q -->|Yes| L[llm_query / llm_query_batched<br/>sub-LM on a snippet]
    Q -->|No| R[Pure code: regex, slicing, set ops]
    L --> O[Observation]
    R --> O
    O --> P
    P -->|done| SU[submit result]
```

The primary LM stays in a small, structured context (summary + tool
catalog + accumulated trajectory). Sub-LM calls each see only the
snippet that the code carved out. The framework caps how many sub-LM
calls one run can make, so a runaway loop can't blow your bill.

## Minimal Example

```python
import synalinks

class Doc(synalinks.DataModel):
    text: str

class Answer(synalinks.DataModel):
    answer: str

primary = synalinks.LanguageModel(model="openai/gpt-4o")

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

The whole `book.txt` lives in the sandbox; the primary LM only sees
its preview and length.

## Inputs as External Environment

`RecursiveLanguageModelAgent` inherits the `CodeModeAgent` rule that
the user input is bound as a dict named `inputs` inside the sandbox —
but it leans on it harder. The primary LM sees an `InputsSummary` in
its prompt instead of the raw values:

```text
fields:
  - name: text
    type: str
    size: 482917         # full character length
    preview: "Chapter 1...  (first 200 chars)"
    truncated: true
```

Two things follow from that:

- **Read full values through `inputs[field]`**, never re-type them
  from the preview. The preview is truncated; the sandbox dict is not.
- **Aggregate in code, query the sub-LM on the result.** A regex pulls
  every quoted string from a 500K document in one snippet; only the
  matching spans get passed to `llm_query` for analysis.

This is the inversion of the usual "give the LM the whole document
and let it figure it out" approach: the LM stays small and orchestrates,
the data stays big and external.

## The Two Recursive Helpers

The sandbox always exposes two extra async tools beyond `submit`:

### `llm_query(prompt)`

```python
async def main():
    out = await llm_query(prompt="Summarize this paragraph: ...")
    print(out["result"])

asyncio.run(main())
```

Sends one prompt to the *sub-LM* (configurable, see below). Returns
`{"result": <text>}`. Use it for semantic work on a snippet you've
already carved out with code: classification, summarization, entity
extraction, reformatting.

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

Same as `llm_query` but takes a list and dispatches them concurrently
via `asyncio.gather`. Returns `{"result": [<text-or-error>, ...]}`,
one entry per input prompt, in input order. Strongly preferred over a
Python loop of `llm_query` calls — sequential calls waste wall time
inside the per-snippet `timeout`.

Failed sub-prompts come back as strings prefixed with
`"[error] <ExceptionType>: <message>"`. Filter them before aggregating
so an exception doesn't get silently treated as data.

### Budget

Both helpers share one counter capped at `max_llm_calls` (default 50).
When exhausted, both short-circuit:

```text
{"result": <empty>, "error": "sub-LM call budget exhausted (50/50). ..."}
```

No quota is consumed by the short-circuit, and the error message
explicitly tells the LM to fall back to code-side aggregation. The
counter resets on every `agent(...)` invocation — concurrent calls
get independent budgets.

For `llm_query_batched`, the budget is checked all-or-nothing: a batch
of 10 with 5 calls remaining short-circuits with `{"result": [], ...}`
rather than partially fulfilling.

## Adding Your Own Tools

`RecursiveLanguageModelAgent` inherits the `tools=` argument from
`CodeModeAgent`. The names ``llm_query``, ``llm_query_batched``, and
``submit`` are reserved, but anything else you bind is exposed inside
the sandbox as a global async callable — alongside the recursive
helpers — and can be composed with them in the same snippet.

Because tool **bodies run on the host with full Python privileges**
(filesystem, network, third-party libraries), they're how the agent
reaches anything outside Monty's whitelist. Decorate each tool with
``@synalinks.saving.register_synalinks_serializable()`` so it
round-trips through ``get_config`` / ``from_config``:

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

Inside a snippet the LM can now interleave host effects with recursive
sub-LM work — fetch a page, hand its body to ``llm_query`` for
classification, then ``submit``. See *Binding Tools — The Bridge to
the Host* in the `CodeModeAgent` guide for the full contract,
naming gotchas, and the security implications of bound tools.

## Choosing a Sub-LM

By default `sub_language_model` defaults to `language_model` — the
primary LM is reused for sub-queries. This is fine for prototyping but
usually wasteful: the primary LM does the orchestration, planning, and
final answer formatting, where capability matters; the sub-LM handles
"summarize this 800-token snippet" or "is this paragraph about X?",
where a much smaller, cheaper model is enough.

Pass a separate `sub_language_model` to split the workload:

```python
primary = synalinks.LanguageModel(model="openai/gpt-4o")
cheap   = synalinks.LanguageModel(model="openai/gpt-4o-mini")

agent = synalinks.RLM(
    data_model=Answer,
    language_model=primary,
    sub_language_model=cheap,
)
```

A typical RLM run dispatches dozens of sub-LM calls and one or two
primary-LM turns, so the bill is dominated by the sub-LM's per-call
cost. Picking a cheap model here is the single biggest cost lever.

## Working Rules (Inside the Prompt)

The default instructions hand the LM a numbered checklist of how to
operate the recursive loop. They show up verbatim in the prompt, but
they're worth reading directly because they describe the failure modes
this agent design has hit in practice:

1. **EXPLORE FIRST** — print sample values, lengths, types, and shapes
   of `inputs[field]` before slicing or batching. A cheap probe turn
   prevents wasted sub-LM calls on the wrong field.
2. **CODE FOR STRUCTURE, `llm_query` FOR MEANING** — regex, slicing,
   and set ops find WHERE things are; the sub-LM understands WHAT they
   mean. Don't burn `llm_query` budget on aggregation a one-liner can
   do.
3. **MINIMIZE RETYPING** — when values are long, precise, or
   error-prone (IDs, numbers, quoted text, code), re-access them via
   `inputs[field]` and compute in Python. Never copy from the
   `InputsSummary` preview into a sub-LM prompt — the preview is
   truncated.
4. **VERIFY BEFORE SUBMITTING** — if results look wrong (empty, zeros,
   unexpected shape), inspect on a separate turn. Don't submit a guess.
5. **`submit` IS TERMINAL** — the snippet runs to completion (so a
   `print` next to `submit` is captured), but a successful submit
   ends the loop with no follow-up turn — you never get to read that
   print. Inspect on one turn, submit on the next.

You can replace these with your own `instructions=` if your task wants
a different operational style, but the defaults are tuned for the
"long input, sparse query" workload RLM is built for.

## Defaults Differ from `CodeModeAgent`

| Setting          | `CodeModeAgent` | `RecursiveLanguageModelAgent` | Why                                                                     |
|------------------|-----------------|-------------------------------|-------------------------------------------------------------------------|
| `max_iterations` | 5               | 20                            | RLM workflows explore → carve → batch-query → aggregate → submit.       |
| `timeout`        | 5s              | 60s                           | One `llm_query_batched(20 prompts)` snippet routinely needs >5s.        |
| `max_llm_calls`  | n/a             | 50                            | Hard ceiling; the LM is told about it and falls back to code on overrun.|

Override any of them if your workload knows better.

## RLM vs `CodeModeAgent` vs `FunctionCallingAgent`

| Aspect              | FunctionCallingAgent              | CodeModeAgent                        | RecursiveLanguageModelAgent              |
|---------------------|-----------------------------------|--------------------------------------|------------------------------------------|
| Action shape        | JSON tool call                    | Python snippet                       | Python snippet                           |
| Primary LM context  | Full input + tool catalog         | InputsSummary + tool catalog         | InputsSummary + tool catalog             |
| Long-input handling | Stuff into prompt                 | Stuff into prompt (read via `inputs`)| Treat as external environment            |
| Recursive sub-LM    | —                                 | —                                    | `llm_query` / `llm_query_batched` built-in |
| Cost lever          | Pick one model                    | Pick one model                       | Pick a cheap `sub_language_model`        |
| Best for            | Discrete, well-typed actions      | Composition / batching / control flow| Long inputs, sparse / semantic queries   |

Reach for `RecursiveLanguageModelAgent` when the input dwarfs the
question — long documents, large logs, big lists where only a few
items matter — and you want the LM to *decide what to look at* rather
than commit upfront to a chunking or retrieval strategy.

## Complete Example

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

    primary = synalinks.LanguageModel(model="openai/gpt-4o")
    cheap   = synalinks.LanguageModel(model="openai/gpt-4o-mini")

    inputs = synalinks.Input(data_model=Doc)
    outputs = await synalinks.RLM(
        data_model=Answer,
        language_model=primary,
        sub_language_model=cheap,
        max_iterations=10,
        max_llm_calls=20,
    )(inputs)
    agent = synalinks.Program(inputs=inputs, outputs=outputs, name="rlm_qa")

    long_text = open("book.txt").read()
    result = await agent(Doc(text=long_text))
    print(result.prettify_json())

if __name__ == "__main__":
    asyncio.run(main())
```

Inside that `agent(...)` call, the primary LM never sees more than the
preview of `book.txt`. It writes Python that scans the full text in
the sandbox, picks out spans worth reading, dispatches them to the
cheap sub-LM, aggregates the answers, and calls `submit`.

## Key Takeaways

- **Long inputs as external environment**: the primary LM sees a
  metadata summary; the full value lives in `inputs[field]` inside
  the sandbox.

- **Two recursive helpers**: `llm_query(prompt)` and
  `llm_query_batched(prompts)` send work to a sub-LM. Both return
  dicts with a `result` key (and an `error` key when the budget is
  exhausted or a batched prompt fails).

- **Shared budget**: `max_llm_calls` caps total sub-LM calls per run
  and is shared between the two helpers. Each `agent(...)` invocation
  gets a fresh counter; concurrent calls do not race.

- **Pick a cheap `sub_language_model`**: the bill is dominated by sub-LM
  calls. Splitting primary vs. sub-LM is the largest cost lever.

- **Working rules in the prompt**: the default instructions hand the
  LM a 5-item operating checklist tuned for the long-input, sparse-query
  workload. Override `instructions=` if your task wants a different
  style.

- **Same `submit` discipline as `CodeModeAgent`**: the LM ends a run by
  calling `submit(result={...})`. Schema validation, the
  `max_iterations` fallback, and the per-snippet `timeout` semantics
  all carry over unchanged from `CodeModeAgent`.

## API References

- [RecursiveLanguageModelAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/RecursiveLanguageModelAgent%20module/)
- [CodeModeAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/CodeModeAgent%20module/)
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
    primary = synalinks.LanguageModel(model="ollama/qwen3")
    cheap = synalinks.LanguageModel(model="ollama/qwen3")

    haystack = build_haystack(
        needle="The magic number is 4242, please remember it.",
        paragraphs=200,
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
        max_iterations=10,
        max_llm_calls=20,
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
        max_iterations=8,
        max_llm_calls=3,  # almost no sub-LM budget — must use regex / slicing
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
