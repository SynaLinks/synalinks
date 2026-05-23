# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Vector RAG Agent

[Guide 6](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/) showed how to store documents in a
`KnowledgeBase` and query them with similarity / fulltext / hybrid
search. [Guide 5](https://synalinks.github.io/synalinks/guides/Agents/) introduced the agent loop. This guide combines
the two: a **Vector RAG Agent** is an agent whose tool set is
pre-wired for retrieval, so the *agent itself* decides whether to
search, what table to search, and how to phrase the query.

## Why Make Retrieval Agentic?

A textbook RAG pipeline is a fixed three-step program:

1. Embed the user's question.
2. Run a single similarity search.
3. Stuff the top-k results into a final prompt.

That works when retrieval is *always* needed, the right phrasing
*is* the user's question, and a single round of results *is* enough.
It struggles when:

- The user asks something the model already knows ("what day is
  today?") — the retrieval step is pure overhead.
- The user's phrasing doesn't match the corpus phrasing ("PTO
  policy" vs "vacation days") — one search misses; a reformulation
  might land.
- One topic is enough; another needs two searches against different
  tables.

An agent moves those decisions inside the loop. The model sees the
question, *decides* whether to retrieve, picks the table, writes a
query in its own words, reads the results, and may search again
before answering.

```mermaid
graph TD
    A["User question"] --> B["VectorRAGAgent"]
    B --> C{"Retrieval needed?"}
    C -->|"no"| H["Final answer"]
    C -->|"yes"| D["search_knowledge_base"]
    D --> E["Top-k hits"]
    E --> F{"Enough info?"}
    F -->|"yes"| H
    F -->|"no, refine"| D
    F -->|"need a record's full body"| G["get_record_by_id"]
    G --> H
```

## The Three Tools

`synalinks.VectorRAGAgent` wraps a `FunctionCallingAgent` and
pre-wires three retrieval tools bound to a `KnowledgeBase`:

| Tool | Purpose |
|------|---------|
| `get_knowledge_base_schema()` | List every table with its columns and descriptions. Used once to learn what's available. |
| `search_knowledge_base(table_name, query)` | Run the configured search (similarity / fulltext / hybrid_fts) against one table and return up to `k` hits. `k` is fixed per-agent at construction; the LM doesn't choose it per-call. |
| `get_record_by_id(table_name, record_id)` | After a search returns ids, fetch a full record. Useful when search snippets truncate long fields. |

The three tools cover the natural shape of retrieval: *what tables
exist → search one of them → optionally read a record in full*.

## Picking a Search Mode

`search_type` is a per-agent setting (the LM doesn't choose). It
shapes both the dispatch behaviour and the default instructions the
LM receives (so it phrases queries the right way).

| `search_type` | What it does | Use when |
|---------------|--------------|----------|
| `"similarity"` | Pure vector search over embeddings. Requires an embedding model. | The corpus and the question use *different* words for the same concept (paraphrase-heavy). |
| `"fulltext"` | BM25 keyword search. No embedding model needed. | The words in the question are likely to appear verbatim in the documents (technical terms, named entities). |
| `"hybrid_fts"` (default) | Vector + BM25 fused with Reciprocal Rank Fusion (RRF). Requires an embedding model. | You don't want to choose — RRF combines both signals and is the safest default. |

For each mode, the agent's default instructions tell the LM how to
phrase its queries: natural-language paraphrases for similarity,
keyword-rich strings for fulltext, both for hybrid.

## Output Format

Search results are returned as **CSV** by default (`output_format="csv"`),
which is dramatically more token-efficient than JSON for tabular
hits — no key names repeated per row. The LM reads CSV well, and on
modern providers it parses faster too. Switch to `"json"` if you
need list-of-dicts results for downstream code, but for purely
LM-facing flows CSV is the better default.

## Building the Agent

The constructor signature mirrors `FunctionCallingAgent` exactly —
every parameter on that class is accepted with identical semantics.
The additions are retrieval-specific:

| Param | Required | Default | Notes |
|-------|----------|---------|-------|
| `knowledge_base` | yes | — | The `KnowledgeBase` to retrieve from. |
| `search_type` | no | `"hybrid_fts"` | `"similarity"`, `"fulltext"`, or `"hybrid_fts"`. |
| `k` | no | `5` | Top-k for searches. Fixed per-agent — not exposed to the LM. |
| `similarity_threshold` | no | `None` | Max vector distance for similarity / hybrid modes. |
| `fulltext_threshold` | no | `None` | Min BM25 score for fulltext / hybrid modes. |
| `output_format` | no | `"csv"` | `"csv"` (compact) or `"json"` (list of dicts). |
| `tools` | no | `None` | Extra `Tool` instances or async functions to append to the three built-ins. Same name-collision and no-leading-underscore rules as `FunctionCallingAgent`. |

```python
import synalinks

embedding_model = synalinks.EmbeddingModel(
    model="gemini/gemini-embedding-001",
)
kb = synalinks.KnowledgeBase(
    uri="duckdb://docs.db",
    data_models=[Document],
    embedding_model=embedding_model,
)
# ... populate kb ...

lm = synalinks.LanguageModel(model="ollama/mistral")

inputs = synalinks.Input(data_model=synalinks.ChatMessages)
outputs = await synalinks.VectorRAGAgent(
    knowledge_base=kb,
    language_model=lm,
)(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs)
```

Note: `"similarity"` and `"hybrid_fts"` modes need the knowledge
base to have an `embedding_model` set. `"fulltext"` works without
one.

## Layering Extra Tools

The `tools=` slot lets you append non-retrieval tools to the built-
in three. Useful when the answer mixes retrieved text with another
capability:

```python
@synalinks.saving.register_synalinks_serializable()
async def calculate(expression: str):
    \"\"\"Evaluate an arithmetic expression.

    Args:
        expression (str): A safe expression like '100 * 0.85'.
    \"\"\"
    return {"result": eval(expression, {"__builtins__": {}}, {})}

agent_module = synalinks.VectorRAGAgent(
    knowledge_base=kb,
    language_model=lm,
    tools=[synalinks.Tool(calculate)],
)
```

Now the LM can retrieve a pricing policy from the kb and then
compute a discount in the same turn.

## A Worked Example

A small end-to-end task: a kb of HR / pricing policies, and the
agent answers a question that needs both retrieval and arithmetic.

```python
result = await agent(synalinks.ChatMessages(messages=[synalinks.ChatMessage(
    role="user",
    content=(
        "If I have 100 users on the Enterprise plan, what would the "
        "per-user monthly cost be after the volume discount?"
    ),
)]))
```

What the agent typically does:

1. `search_knowledge_base("Document", "enterprise plan pricing volume discount")`
   — pulls the pricing-policy document.
2. The LM reads the discount tiers from the snippet.
3. `calculate("99 * (1 - 0.15)")` — applies the 15% discount.
4. Stops, produces the final natural-language answer.

If the first search doesn't return what the LM expected, it
reformulates and searches again. Multiple search rounds are normal
when the question's phrasing diverges from the corpus's.

## Multi-Turn Conversations

`VectorRAGAgent` accepts `ChatMessages` as input and threads context
across turns. Each turn the agent sees the full conversation, so it
can ground a follow-up against retrieval from a previous turn
without re-asking:

```python
messages = []
for user_msg in [
    "What's the daily meal allowance?",
    "What about international?",
    "I'm going for 3 days — what's my total budget?",
]:
    messages.append(synalinks.ChatMessage(role="user", content=user_msg))
    chat = synalinks.ChatMessages(messages=messages)
    result = await agent(chat)

    last = next(m for m in reversed(result.get("messages", []))
                if m.get("role") == "assistant" and m.get("content"))
    print("Agent:", last["content"])
    messages.append(synalinks.ChatMessage(role="assistant", content=last["content"]))
```

## Compared to Other Agents

`VectorRAGAgent` is one of several specialized agents that wrap a
`FunctionCallingAgent` with a workload-specific tool set:

| Agent | Bound to | Tools |
|-------|----------|-------|
| `FunctionCallingAgent` | nothing | whatever you pass in |
| `SQLAgent` | a `KnowledgeBase` | schema discovery, table sample, read-only SQL |
| `VectorRAGAgent` | a `KnowledgeBase` | schema discovery, similarity / fulltext / hybrid search, get-by-id |
| `DeepAgent` | a workdir | list, search, read, write, edit, bash |

When to pick `VectorRAGAgent` over `SQLAgent`: the data is *text*
(unstructured documents), and the question is "find me what's
relevant to X" rather than "compute Y from rows of Z". When the
data is structured (typed columns) and the answer needs joins or
aggregations, reach for `SQLAgent`.

You can also use both — give an agent a `KnowledgeBase` with both
document tables and structured tables, layer SQL tools on top of
retrieval, and let the LM mix them. But for a single workload, the
specialized agent is the simpler call.

## API References

- [VectorRAGAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/VectorRAGAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [SQLAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/SQLAgent%20module/)
- [DeepAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/DeepAgent%20module/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/Knowledge%20Base/)
- [EmbeddingModel](https://synalinks.github.io/synalinks/Synalinks%20API/Embedding%20Models%20API/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks


class Document(synalinks.DataModel):
    """A document in the knowledge base."""

    id: str = synalinks.Field(description="Document id")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Body text")


SAMPLE_DOCUMENTS = [
    Document(
        id="pricing-enterprise",
        title="Enterprise Plan Pricing",
        content=(
            "Enterprise Plan: $99/user/month billed annually. "
            "Volume discounts: 10% off at 50 users, 15% off at 100, "
            "20% off at 250."
        ),
    ),
    Document(
        id="policy-pto",
        title="PTO Policy",
        content=(
            "Full-time employees receive 20 days of PTO per year. "
            "PTO accrues monthly. Up to 5 unused days carry over."
        ),
    ),
    Document(
        id="policy-remote",
        title="Remote Work Policy",
        content=(
            "Employees may work remotely up to 3 days per week with "
            "manager approval. Core hours are 10 AM to 4 PM."
        ),
    ),
]


async def main():
    load_dotenv()
    synalinks.clear_session()

    db_path = "./guides/vector_rag_agent_guide.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Build a kb with an embedding model so hybrid_fts can use both
    # vector and BM25 signals.
    embedding_model = synalinks.EmbeddingModel(model="ollama/all-minilm")
    kb = synalinks.KnowledgeBase(
        uri=f"duckdb://{db_path}",
        data_models=[Document],
        embedding_model=embedding_model,
        wipe_on_start=True,
        name="vector_rag_agent_guide_kb",
    )

    for doc in SAMPLE_DOCUMENTS:
        await kb.update(doc.to_json_data_model())

    language_model = synalinks.LanguageModel(model="ollama/mistral")

    # Build the agent with the default hybrid_fts mode.
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)
    outputs = await synalinks.VectorRAGAgent(
        knowledge_base=kb,
        language_model=language_model,
        max_iterations=5,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="vector_rag_agent",
        description="A RAG agent that retrieves and answers from documents.",
    )

    # Ask a question that needs retrieval to answer correctly.
    question = synalinks.ChatMessages(
        messages=[
            synalinks.ChatMessage(
                role="user",
                content="How much PTO do employees get per year?",
            )
        ]
    )
    result = await agent(question)

    # Print the final assistant message.
    for msg in reversed(result.get("messages", [])):
        if msg.get("role") == "assistant" and msg.get("content"):
            print("Agent:", msg["content"])
            break

    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
