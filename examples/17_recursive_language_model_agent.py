"""
# Recursive Language Model Agent

The `RecursiveLanguageModelAgent` (exported as `synalinks.RLM`) is built
for tasks where the *input itself* is too large or too noisy to feed
straight into the language model. Instead of packing a whole book, log
dump, or scraped corpus into the primary LM's context, the agent treats
those inputs as an **external environment**: the LM writes Python that
programmatically slices, filters, and aggregates the data inside a
persistent sandbox, and recursively delegates semantic work to a
*sub-LM* on the snippets it actually cares about.

The pattern follows
[Recursive Language Models (Zhang, Kraska, Khattab — 2025)](https://arxiv.org/abs/2512.24601).

## Why Recursive?

A long context is expensive on three compounding axes: token cost
scales linearly with the prompt size, latency scales linearly, and
accuracy regresses past a model-specific knee (the "lost in the middle"
effect). RLM avoids all three by keeping the primary LM in a small,
structured context (a metadata summary of the input plus the tool
catalog plus the accumulated trajectory). The full value lives in the
sandbox under `inputs[field]`, and the LM decides *per query, per turn*
which slice to look at.

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

## Needle in a Haystack

This example builds a long, repetitive document (~200 paragraphs of
filler text) and hides a single fact — "The magic number is 4242" —
near the middle. The primary LM never sees the full text; it only
sees an `InputsSummary` with a preview and a length. Finding the
needle requires writing code that scans the full text in the sandbox
and either uses a regex or batches sub-LM calls over candidate spans.

Define a `Doc` input and `Answer` output data model:

```python
class Doc(synalinks.DataModel):
    text: str = synalinks.Field(description="The document to analyze")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer to the user")
```

Wire up the RLM agent. The primary LM drives orchestration and final
formatting; the sub-LM (configurable via `sub_language_model=`) handles
per-snippet semantic work. Both default to the same model when only
`language_model=` is passed:

```python
inputs = synalinks.Input(data_model=Doc)
outputs = await synalinks.RLM(
    data_model=Answer,
    language_model=language_model,
    max_iterations=10,
    max_llm_calls=20,
)(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs, name="rlm_needle")
```

When the agent runs, the primary LM is given a **single** tool —
`run_python_code(code=...)` — and calls it with one Python
snippet per turn. The snippet runs in a Monty REPL sandbox and the call
returns `{"stdout": ..., "stderr": ..., "error": ...}`. State persists
across turns — variables, imports, and function definitions accumulate.
`submit` and two extra async helpers live **inside** the sandbox (not as
tools the LM can call) alongside any tools you bind:

- `llm_query(prompt)` — single sub-LM call, returns `{"result": <text>}`.
- `llm_query_batched(prompts)` — concurrent sub-LM calls, returns
  `{"result": [<text>, ...]}`, preserving input order.

A shared counter caps the two helpers at `max_llm_calls` per
`agent(...)` invocation; when exhausted they short-circuit with
`{"result": <empty>, "error": "..."}` and do not consume quota. The
counter resets on every invocation, so concurrent calls get independent
budgets.

Termination: the snippet calls the in-sandbox `submit(result={...})`,
which captures the final payload, validates it against the configured
output schema, and ends the run. Empty snippets are no-ops — the loop
reminds the LM to call `submit`. If `max_iterations` is reached without a
successful `submit`, a final LM inference step formats the accumulated
trajectory into the target schema.

### Key Takeaways

- **Long inputs as external environment**: the primary LM sees a
  metadata summary; the full value lives in `inputs[field]` inside
  the sandbox.
- **Two recursive helpers**: `llm_query` and `llm_query_batched` send
  work to a sub-LM and share one budget capped at `max_llm_calls`.
- **Pick a cheap `sub_language_model`** when you have one available:
  a typical RLM run is dominated by sub-LM calls, so splitting primary
  vs. sub-LM is the largest cost lever.
- **`submit` is the termination path**: schema validation errors come
  back as a retry observation on the next turn.

## Program Visualization

![rlm_needle](../assets/examples/rlm_needle.png)

## API References

- [RecursiveLanguageModelAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/RecursiveLanguageModelAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [DataModel](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20DataModel%20class/)
- [LanguageModel](https://synalinks.github.io/synalinks/Synalinks%20API/Language%20Models%20API/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

synalinks.enable_logging()


class Doc(synalinks.DataModel):
    text: str = synalinks.Field(
        description="The document to analyze",
    )


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(
        description="The final answer to the user",
    )


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


async def main():
    load_dotenv()

#     synalinks.enable_observability(
#         tracking_uri="http://localhost:5000",
#         experiment_name="rlm_needle",
#     )

    language_model = synalinks.LanguageModel(
        model="ollama/qwen3:8b",
    )

    haystack = build_haystack(
        needle="The magic number is 4242, please remember it.",
        paragraphs=200,
    )
    print(f"Haystack length: {len(haystack)} characters")

    # ==========================================================================
    # Build the RLM agent
    # ==========================================================================
    print("Creating the recursive language model agent...")

    inputs = synalinks.Input(data_model=Doc)
    outputs = await synalinks.RLM(
        data_model=Answer,
        language_model=language_model,
        max_iterations=10,
        max_llm_calls=20,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="rlm_needle",
        description="A recursive language model agent for needle-in-haystack QA",
    )

    synalinks.utils.plot_program(
        agent,
        to_folder="examples",
        show_module_names=True,
        show_trainable=True,
        show_schemas=True,
    )

    # ==========================================================================
    # Run the agent
    # ==========================================================================
    print("Running the agent...")

    response = await agent(Doc(text=haystack))

    print(response.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
