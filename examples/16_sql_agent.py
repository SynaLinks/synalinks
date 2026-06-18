"""
# SQL Agent

An **SQL Agent** combines the reasoning capabilities of LLMs with structured
database access. Instead of writing SQL manually, users ask questions in
natural language and the agent autonomously discovers the schema, constructs
queries, and returns answers.

## Why SQL Agents Matter

Traditional database access requires SQL knowledge. An SQL Agent bridges
this gap:

```mermaid
graph LR
    subgraph Traditional
        A[User] --> B[Write SQL]
        B --> C[Execute Query]
        C --> D[Interpret Results]
    end
    subgraph SQL Agent
        E[User Question] --> F[Agent Reasons]
        F --> G[Auto-generates SQL]
        G --> H[Natural Language Answer]
    end
```

SQL Agents provide:

1. **Natural Language Interface**: Ask questions without knowing SQL
2. **Schema Discovery**: Agent explores database structure automatically
3. **Query Generation**: Constructs correct SQL based on user intent
4. **Safe Execution**: Read-only queries prevent accidental modifications

## Architecture

`synalinks.SQLAgent` wires a `FunctionCallingAgent` with three pre-built
tools bound to a `KnowledgeBase`:

```mermaid
flowchart TD
    A[User Question] --> B[SQLAgent]
    B --> C{Select Tool}
    C -->|Discover structure| D[get_database_schema]
    C -->|View sample data| E[get_table_sample]
    C -->|Execute query| F[run_sql_query]
    D --> G[Schema Info]
    E --> H[Sample Rows]
    F --> I[Query Results]
    G --> B
    H --> B
    I --> B
    B --> J[Natural Language Answer + SQL]
```

The agent follows an autonomous loop: it first discovers the schema (or
reads the tables from the system instructions), optionally samples data to
understand the format, then constructs and executes SQL queries until it
has enough information to answer.

## Available Tools

| Tool | Description |
|------|-------------|
| `get_database_schema` | Returns all tables and their columns with types |
| `get_table_sample` | Fetches sample rows with `LIMIT`/`OFFSET` pagination |
| `run_sql_query` | Executes read-only `SELECT` queries |

Result sets are rendered as **CSV** by default so the LM spends fewer
input tokens reading tabular data. Set `output_format="json"` on the
`SQLAgent` if you want list-of-dicts instead.

## Defining Input and Output Models

```python
import synalinks

class Query(synalinks.DataModel):
    \"\"\"A natural language query about the database.\"\"\"
    query: str = synalinks.Field(
        description="The user's question about the data in natural language"
    )

class SQLResult(synalinks.DataModel):
    \"\"\"The result of the SQL agent's analysis.\"\"\"
    answer: str = synalinks.Field(
        description="A clear, natural language answer to the user's question"
    )
    sql_query: str = synalinks.Field(
        description="The SQL query that was executed to get the answer"
    )
```

The `Query` model captures the user's natural language question. The
`SQLResult` model structures the output to include both a human-readable
answer and the SQL used, providing transparency into the agent's reasoning.

## Building the Agent

`SQLAgent` takes a `KnowledgeBase` + `LanguageModel` + output data model
and handles tool wiring internally — no manual `Tool` definitions needed.

```python
# Create Knowledge Base with your data models
kb = synalinks.KnowledgeBase(
    uri="duckdb://my_database.db",
    data_models=[Customer, Product, SalesOrder],
)

# Configure language model
lm = synalinks.LanguageModel(model="ollama/qwen3:8b")

# Build the agent via the Functional API
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.SQLAgent(
    knowledge_base=kb,
    language_model=lm,
    data_model=SQLResult,
    max_iterations=10,
)(inputs)

sql_agent = synalinks.Program(
    inputs=inputs,
    outputs=outputs,
    name="sql_agent",
)
```

## Safety Considerations

Safety is the **Knowledge Base's** responsibility, not the agent's. The
DuckDB adapter treats `read_only=True` (which `run_sql_query` passes by
default) as the whole safety contract, enforced at two layers — both
using DuckDB's own machinery, so there are no hand-rolled keyword
blocklists (which leak false negatives via comments, string literals,
casing, or stacked statements like `SELECT 1; DROP TABLE x`):

1. **Parser check (blocks writes).** Each statement is parsed with
   DuckDB's own parser and rejected unless it's a `SELECT`. This catches
   multi-statement injection, `COPY (SELECT ...) TO 'file'` exfiltration,
   `ATTACH`, `EXPORT`, and every other side-effecting statement.
2. **Sandbox (blocks external I/O).** The persistent connection has
   `enable_external_access=false` applied at construction time, so
   `SELECT` table functions that touch the host filesystem or network —
   `read_csv`, `read_parquet`, `read_json`, `read_blob`, `read_text`,
   `glob`, the httpfs/S3 variants — return a permission error.

## Example Usage

```python
# Ask a natural language question
result = await sql_agent(Query(query="Who are the top 3 customers by orders?"))

print(result["answer"])
# Output: "The top 3 customers are: Carlos Garcia ($1539.96),
#          Alice Johnson ($1489.96), and Diana Chen ($379.94)"

print(result["sql_query"])
# Output: "SELECT c.name, SUM(o.total_amount) as total
#          FROM Customer c JOIN SalesOrder o ON c.id = o.customer_id
#          GROUP BY c.name ORDER BY total DESC LIMIT 3"
```

The agent automatically:
1. Discovered the Customer and SalesOrder tables
2. Understood the relationship via customer_id
3. Wrote a proper JOIN with aggregation
4. Returned both the answer and the SQL for transparency

## Key Takeaways

- **One module, three tools**: `synalinks.SQLAgent` bundles schema discovery,
  table sampling, and read-only SQL execution into a single ready-to-use
  module. No manual tool wiring.

- **Token-efficient by default**: `output_format="csv"` minimizes LM input
  tokens on wide result sets. Switch to `"json"` only if downstream code
  needs typed structured data.

- **Pagination**: `get_table_sample` accepts `limit` / `offset`; for
  arbitrary queries the LM is instructed to include `LIMIT` clauses.

- **Safety First**: The `run_sql_query` tool always passes
  `read_only=True` to `kb.sql(...)`. Non-`SELECT` statements are
  rejected at the parser; the sandbox blocks `read_csv` / httpfs.

- **Transparent Outputs**: Include the generated SQL in the output schema
  so users can verify and learn from the agent's reasoning.

## API References

- [SQLAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/SQLAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/)
"""

# --8<-- [start:source]
import asyncio
import json
import os

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class Customer(synalinks.DataModel):
    """A customer in the database."""

    id: str = synalinks.Field(description="Customer ID")
    name: str = synalinks.Field(description="Customer name")
    email: str = synalinks.Field(description="Customer email")
    country: str = synalinks.Field(description="Customer country")


class Product(synalinks.DataModel):
    """A product in the database."""

    id: str = synalinks.Field(description="Product ID")
    name: str = synalinks.Field(description="Product name")
    category: str = synalinks.Field(description="Product category")
    price: float = synalinks.Field(description="Product price")
    stock: int = synalinks.Field(description="Stock quantity")


class SalesOrder(synalinks.DataModel):
    """An order in the database."""

    id: str = synalinks.Field(description="Order ID")
    customer_id: str = synalinks.Field(description="Customer ID")
    product_id: str = synalinks.Field(description="Product ID")
    quantity: int = synalinks.Field(description="Quantity ordered")
    total_amount: float = synalinks.Field(description="Total order amount")
    status: str = synalinks.Field(description="Order status")


class Query(synalinks.DataModel):
    """A natural language query about the database."""

    query: str = synalinks.Field(
        description="The user's question about the data in natural language"
    )


class SQLResult(synalinks.DataModel):
    """The result of the SQL agent's analysis."""

    answer: str = synalinks.Field(
        description="A clear, natural language answer to the user's question"
    )
    sql_query: str = synalinks.Field(
        description="The SQL query that was executed to get the answer"
    )


# =============================================================================
# Database Setup
# =============================================================================


async def populate_knowledge_base(kb):
    """Populate the knowledge base with sample data."""

    customers = [
        Customer(
            id="C001", name="Alice Johnson", email="alice@example.com", country="USA"
        ),
        Customer(id="C002", name="Bob Smith", email="bob@example.com", country="UK"),
        Customer(
            id="C003",
            name="Carlos Garcia",
            email="carlos@example.com",
            country="Spain",
        ),
        Customer(
            id="C004", name="Diana Chen", email="diana@example.com", country="China"
        ),
        Customer(
            id="C005", name="Emma Wilson", email="emma@example.com", country="Canada"
        ),
    ]

    products = [
        Product(
            id="P001",
            name="Laptop Pro",
            category="Electronics",
            price=1299.99,
            stock=50,
        ),
        Product(
            id="P002",
            name="Wireless Mouse",
            category="Electronics",
            price=49.99,
            stock=200,
        ),
        Product(
            id="P003",
            name="Mechanical Keyboard",
            category="Electronics",
            price=149.99,
            stock=100,
        ),
        Product(
            id="P004",
            name="USB-C Hub",
            category="Accessories",
            price=79.99,
            stock=150,
        ),
        Product(
            id="P005",
            name="Monitor Stand",
            category="Accessories",
            price=89.99,
            stock=75,
        ),
        Product(
            id="P006",
            name="Webcam HD",
            category="Electronics",
            price=129.99,
            stock=80,
        ),
        Product(id="P007", name="Desk Lamp", category="Office", price=45.99, stock=120),
        Product(
            id="P008",
            name="Notebook Set",
            category="Office",
            price=12.99,
            stock=500,
        ),
    ]

    orders = [
        SalesOrder(
            id="O001",
            customer_id="C001",
            product_id="P001",
            quantity=1,
            total_amount=1299.99,
            status="completed",
        ),
        SalesOrder(
            id="O002",
            customer_id="C001",
            product_id="P002",
            quantity=2,
            total_amount=99.98,
            status="completed",
        ),
        SalesOrder(
            id="O003",
            customer_id="C002",
            product_id="P003",
            quantity=1,
            total_amount=149.99,
            status="completed",
        ),
        SalesOrder(
            id="O004",
            customer_id="C003",
            product_id="P001",
            quantity=1,
            total_amount=1299.99,
            status="shipped",
        ),
        SalesOrder(
            id="O005",
            customer_id="C003",
            product_id="P004",
            quantity=3,
            total_amount=239.97,
            status="shipped",
        ),
        SalesOrder(
            id="O006",
            customer_id="C004",
            product_id="P002",
            quantity=5,
            total_amount=249.95,
            status="pending",
        ),
        SalesOrder(
            id="O007",
            customer_id="C004",
            product_id="P006",
            quantity=1,
            total_amount=129.99,
            status="completed",
        ),
        SalesOrder(
            id="O008",
            customer_id="C005",
            product_id="P007",
            quantity=2,
            total_amount=91.98,
            status="completed",
        ),
        SalesOrder(
            id="O009",
            customer_id="C005",
            product_id="P008",
            quantity=10,
            total_amount=129.90,
            status="completed",
        ),
        SalesOrder(
            id="O010",
            customer_id="C001",
            product_id="P005",
            quantity=1,
            total_amount=89.99,
            status="pending",
        ),
    ]

    for customer in customers:
        await kb.update(customer.to_json_data_model())
    for product in products:
        await kb.update(product.to_json_data_model())
    for order in orders:
        await kb.update(order.to_json_data_model())

    print(f"  Stored {len(customers)} customers")
    print(f"  Stored {len(products)} products")
    print(f"  Stored {len(orders)} orders")


# =============================================================================
# Main Example
# =============================================================================


async def main():
    """Demonstrate the SQL agent with natural language queries."""
    load_dotenv()
    synalinks.clear_session()

    print("=" * 60)
    print("SQL Agent with Knowledge Base")
    print("=" * 60)

    db_path = "guides/sql_agent.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\nCreating knowledge base...")
    kb = synalinks.KnowledgeBase(
        uri=f"duckdb://{db_path}",
        data_models=[Customer, Product, SalesOrder],
        wipe_on_start=True,
        name="sql_agent_kb",
    )

    print("Populating with sample data...")
    await populate_knowledge_base(kb)

    print("\nBuilding SQL agent...")

    lm = synalinks.LanguageModel(model="ollama/qwen3:8b")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.SQLAgent(
        knowledge_base=kb,
        language_model=lm,
        data_model=SQLResult,
        max_iterations=10,
    )(inputs)

    sql_agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="sql_agent",
        description="An agent that answers questions about data using SQL queries",
    )

    sql_agent.summary()

    example_queries = [
        "What tables are available in the database?",
        "Who are the top 3 customers by total order amount?",
        "What is the most popular product category?",
        "Show me all pending orders with customer names",
    ]

    print("\n" + "=" * 60)
    print("SQL Agent Demo")
    print("=" * 60)

    for query_text in example_queries:
        print(f"\nQuestion: {query_text}")
        print("-" * 40)

        try:
            result = await sql_agent(Query(query=query_text))

            # Show trajectory (tool calls and results)
            messages = result.get("messages", [])
            tool_calls_count = 0
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        tool_calls_count += 1
                        # name/arguments nest under "function" (the
                        # chat-completion shape); fall back to a flat dict.
                        fn = tool_call.get("function", tool_call)
                        name = fn.get("name", "?")
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (ValueError, TypeError):
                                args = {"_raw": args}
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                        print(
                            f"Tool Call {tool_calls_count}: {name}({args_str})"
                        )
                elif msg.get("role") == "tool":
                    content = msg.get("content", "")
                    # Truncate long results for readability
                    if isinstance(content, dict):
                        content = str(content)
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"Tool Result: {content}")

            print(f"\nAnswer: {result['answer']}")
            print(f"SQL: {result['sql_query']}")
        except Exception as e:
            print(f"Error: {e}")

        print()

    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
