# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# SQL Agent

[Guide 5](Agents.md) showed the agent loop in the abstract: a model that
decides, acts, observes, and repeats. [Guide 6](Knowledge%20Base.md) introduced
the `KnowledgeBase` — a typed, queryable store backed by DuckDB. This
guide combines the two. An **SQL Agent** is an agent whose tool set is
pre-wired for working with a knowledge base via SQL: it can discover
the schema, sample rows, and run read-only SELECT queries. The user
asks a question in natural language; the agent writes the SQL.

## When Do You Want One?

A plain `Generator` answers from prompt context alone. A `RAG` pipeline
retrieves text and pastes it into the prompt. Neither knows about
*structure*: the difference between a row, a column, a join. When the
answer requires:

- joining tables (top customers by order total),
- aggregations (count, sum, group-by),
- filters that need exact-typed comparisons (price < 50, status =
  "pending"),

an SQL agent is the right shape. The model writes one `SELECT`, the
engine computes the answer, and the model summarizes it. No
embedding similarity, no chunking, no recall problem — just the
right tuples.

## The Three Tools

`synalinks.SQLAgent` wraps a `FunctionCallingAgent` and pre-wires
three tools, all bound to a single `KnowledgeBase` you supply:

| Tool | Purpose |
|------|---------|
| `get_database_schema()` | Returns every table with its columns, types, and descriptions. |
| `get_table_sample(table_name, limit, offset)` | Fetch a few rows from a table to see what the values actually look like. Page-bounded by the agent's `k`. |
| `run_sql_query(sql_query)` | Execute a read-only `SELECT` query and return the rows. Wrapped server-side with an outer `LIMIT k` so even an unbounded query can't drain a giant table into the conversation. |

The agent's default instructions already include the list of
available tables, so a trivial question can skip `get_database_schema`
and jump straight to `run_sql_query`.

### Why these three, and not just one

You could imagine giving the agent a single "ask the DB" tool. But
the LM needs to know the schema before it can write SQL, and reading
the schema repeatedly into every prompt wastes tokens. The
three-tool split lets the LM:

1. Discover the schema once.
2. Optionally peek at sample values when a column's type isn't
   enough to know what's in it ("status" — is it `pending`/`paid`,
   or `0`/`1`?).
3. Then write its query.

Each tool returns just what's needed for the next step.

## Safety Model

Safety is the **knowledge base's** responsibility, not the agent's.
The `run_sql_query` tool always passes `read_only=True` to
`kb.query(...)`. The DuckDB adapter enforces that flag in two layers
— both using DuckDB's own machinery, so there are no hand-rolled
keyword blocklists (which leak false negatives through comments,
string literals, casing, and stacked statements like `SELECT 1;
DROP TABLE x`):

1. **Parser layer.** Each statement is parsed with DuckDB's parser
   and rejected unless it's a `SELECT`. This catches multi-statement
   injection, `COPY (SELECT ...) TO '/path'` filesystem exfiltration,
   `ATTACH`, `EXPORT`, and every other side-effecting statement.
2. **Connection sandbox.** The persistent connection had
   `enable_external_access=false` applied at construction time, so
   `SELECT` table functions that touch the host filesystem or
   network — `read_csv`, `read_parquet`, `read_json`, `read_blob`,
   `glob`, httpfs/S3 variants — return a permission error.

That means the `run_sql_query` tool body is trivial: pass the SQL
through and surface any error so the agent can read it and retry
with corrected SQL.

## The `k` Cap

Both `get_table_sample` and `run_sql_query` are capped server-side
at the agent's `k` setting (default 50):

- `get_table_sample`: the LM's `limit` argument is clamped to
  `min(limit, k)`.
- `run_sql_query`: the LM's SQL is wrapped in
  `SELECT * FROM ({sql}) LIMIT k`. A `LIMIT` inside the LM's own
  SQL still applies first, so `LIMIT 5` returns 5 rows even when
  the cap is higher.

Result dicts include `row_cap` and `may_have_more` so the LM can
detect truncation and refine the query (add filters, `ORDER BY ...
LIMIT n`) rather than asking for more rows.

## Building the Agent

The constructor signature mirrors `FunctionCallingAgent` exactly —
every parameter on that class is accepted with identical semantics.
The additions are SQL-specific:

| Param | Required | Default | Notes |
|-------|----------|---------|-------|
| `knowledge_base` | yes | — | The `KnowledgeBase` to query. |
| `k` | no | `50` | Max rows per call (samples and queries). |
| `output_format` | no | `"csv"` | How result sets render to the LM. `"csv"` is compact; `"json"` returns a list of dicts. |
| `tools` | no | `None` | Extra `Tool` instances or async functions to append to the three built-ins. Same name-collision and no-leading-underscore rules as `FunctionCallingAgent`. |

```python
import synalinks

lm = synalinks.LanguageModel(model="ollama/mistral")
kb = synalinks.KnowledgeBase(
    uri="duckdb://my_database.db",
    data_models=[Customer, Product, SalesOrder],
)

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="A natural-language question")

class SQLAnswer(synalinks.DataModel):
    answer: str = synalinks.Field(description="A natural-language answer")
    sql_query: str = synalinks.Field(description="The SQL that produced it")

inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.SQLAgent(
    knowledge_base=kb,
    language_model=lm,
    data_model=SQLAnswer,
)(inputs)
agent = synalinks.Program(inputs=inputs, outputs=outputs)
```

`data_model=SQLAnswer` is a common pattern: the agent's *final*
output is a structured object with both the natural-language answer
and the SQL that produced it. That gives downstream code something
to display and lets users verify the query.

## A Worked Example

A small end-to-end task: ask the agent to find the top customers by
order total against the demo data populated in this guide's `main()`
(three customers, four orders). The agent has to discover both
tables (`Customer` and `SalesOrder`), figure out the join key, write
a `GROUP BY` query, and summarize the result.

```python
result = await agent(Query(query="Who are the top 2 customers by total order amount?"))

print(result["answer"])
# "Alice and Carol are tied as the top customers, each with $1,500
#  in total order amount."

print(result["sql_query"])
# SELECT c.name, SUM(o.total_amount) AS total
# FROM Customer c JOIN SalesOrder o ON c.id = o.customer_id
# GROUP BY c.name
# ORDER BY total DESC
# LIMIT 2
```

What the agent typically does:

1. `get_database_schema()` if the table list in the instructions
   doesn't give it enough information about columns.
2. `get_table_sample("SalesOrder", limit=3, offset=0)` to see
   whether `customer_id` matches `Customer.id` and what
   `total_amount` looks like.
3. `run_sql_query(...)` with the JOIN + GROUP BY + ORDER BY.
4. Stop calling tools; produce the final `SQLAnswer`.

If the query has a typo or a non-existent column, the result dict
carries the error and the agent retries.

## Compared to Other Agents

`SQLAgent` is one of several specialized agents that wrap a
`FunctionCallingAgent` with a workload-specific tool set:

| Agent | Bound to | Tools |
|-------|----------|-------|
| `FunctionCallingAgent` | nothing | whatever you pass in |
| `SQLAgent` | a `KnowledgeBase` | schema discovery, table sample, read-only SQL |
| `VectorRAGAgent` | a `KnowledgeBase` | schema discovery, semantic / keyword / hybrid search, get-by-id |
| `DeepAgent` | a workdir | list, search, read, write, edit, bash |

When to pick `SQLAgent` over `VectorRAGAgent`: the data is
structured (rows and columns), and the question requires
computation (joins, aggregations, exact filters). When the data is
documents and the question is "find me the passage that talks
about X", reach for `VectorRAGAgent` instead.

## API References

- [SQLAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/SQLAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [VectorRAGAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/VectorRAGAgent%20module/)
- [DeepAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/DeepAgent%20module/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Bases%20API/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks


class Customer(synalinks.DataModel):
    """A customer."""

    id: str = synalinks.Field(description="Customer ID")
    name: str = synalinks.Field(description="Customer name")
    country: str = synalinks.Field(description="Customer country")


class SalesOrder(synalinks.DataModel):
    """An order."""

    id: str = synalinks.Field(description="Order ID")
    customer_id: str = synalinks.Field(description="Customer ID")
    total_amount: float = synalinks.Field(description="Order total")


class Query(synalinks.DataModel):
    """A natural-language question about the data."""

    query: str = synalinks.Field(description="The question")


class SQLAnswer(synalinks.DataModel):
    """The agent's structured answer."""

    answer: str = synalinks.Field(description="Natural-language answer")
    sql_query: str = synalinks.Field(description="SQL that produced the answer")


async def main():
    load_dotenv()
    synalinks.clear_session()

    db_path = "./guides/sql_agent_guide.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Build the knowledge base with two related tables.
    kb = synalinks.KnowledgeBase(
        uri=f"duckdb://{db_path}",
        data_models=[Customer, SalesOrder],
        wipe_on_start=True,
        name="sql_agent_guide_kb",
    )

    # Sample data — enough to make a top-N query non-trivial.
    customers = [
        Customer(id="C1", name="Alice", country="USA"),
        Customer(id="C2", name="Bob", country="UK"),
        Customer(id="C3", name="Carol", country="Spain"),
    ]
    orders = [
        SalesOrder(id="O1", customer_id="C1", total_amount=1200.0),
        SalesOrder(id="O2", customer_id="C1", total_amount=300.0),
        SalesOrder(id="O3", customer_id="C2", total_amount=150.0),
        SalesOrder(id="O4", customer_id="C3", total_amount=1500.0),
    ]
    for c in customers:
        await kb.update(c.to_json_data_model())
    for o in orders:
        await kb.update(o.to_json_data_model())

    language_model = synalinks.LanguageModel(model="ollama/mistral")

    # Build the agent. The data_model param sets the schema of the
    # final answer — the LM is required to produce both a natural-
    # language explanation and the SQL it used.
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.SQLAgent(
        knowledge_base=kb,
        language_model=language_model,
        data_model=SQLAnswer,
        max_iterations=10,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="sql_agent",
        description="A SQL agent that answers questions about the database.",
    )

    # Ask a non-trivial question that requires a JOIN + GROUP BY.
    result = await agent(
        Query(query="Who are the top 2 customers by total order amount?")
    )
    print("Answer:", result["answer"])
    print("SQL:", result["sql_query"])

    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
