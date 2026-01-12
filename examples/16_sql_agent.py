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

The SQL Agent uses three specialized tools to interact with the Knowledge Base:

```mermaid
flowchart TD
    A[User Question] --> B[FunctionCallingAgent]
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

The agent follows an autonomous loop: it first discovers the schema, optionally
samples data to understand the format, then constructs and executes SQL queries.
This process repeats until the agent has enough information to answer.

## Available Tools

| Tool | Description |
|------|-------------|
| `get_database_schema` | Returns all tables and their columns with types |
| `get_table_sample` | Fetches sample rows from a table with pagination |
| `run_sql_query` | Executes read-only SELECT queries safely |

## Defining Input and Output Models

First, define DataModels for the agent's input (user query) and output
(answer with SQL):

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

The `Query` model captures the user's natural language question. The `SQLResult`
model structures the output to include both a human-readable answer and the
SQL query used, providing transparency into the agent's reasoning.

## Tool Design for Database Access

Each tool uses `kb.get_symbolic_data_models()` to dynamically discover
available tables. This makes the agent adaptable to any database schema
without hardcoding table names.

**Important Tool Constraints:**

- **No Optional Parameters**: All tool parameters must be required. OpenAI
  and other LLM providers require all parameters in their JSON schemas. Do
  not use default values for parameters.

- **Complete Docstring Required**: Every parameter must be documented in the
  `Args:` section. The Tool extracts descriptions from the docstring to build
  the JSON schema sent to the LLM. Missing descriptions raise a ValueError.

Example tool definition:

```python
from synalinks.src.saving.object_registration import register_synalinks_serializable

@register_synalinks_serializable()
async def get_database_schema():
    \"\"\"Get the complete database schema including all tables and columns.\"\"\"
    kb = get_knowledge_base()
    symbolic_models = kb.get_symbolic_data_models()

    schema_info = []
    for model in symbolic_models:
        schema = model.get_schema()
        table_name = schema.get("title", "Unknown")
        properties = schema.get("properties", {})

        columns = []
        for col_name, col_info in properties.items():
            col_type = col_info.get("type", "unknown")
            columns.append(f"  - {col_name} ({col_type})")

        schema_info.append(f"Table: {table_name}\\n" + "\\n".join(columns))

    return {"schema": "\\n\\n".join(schema_info), "table_count": len(symbolic_models)}
```

The `@register_synalinks_serializable()` decorator enables the tool to be
saved and loaded with the program. The tool extracts table names and column
information from the JSON schema of each symbolic data model.

## Building the SQL Agent

Wrap the tool functions with `synalinks.Tool()` and create the agent using
`FunctionCallingAgent`:

```python
# Create Knowledge Base with your data models
kb = synalinks.KnowledgeBase(
    uri="duckdb://my_database.db",
    data_models=[Customer, Product, SalesOrder],
)

# Configure language model
lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

# Wrap async functions as Tool objects
schema_tool = synalinks.Tool(get_database_schema)
sample_tool = synalinks.Tool(get_table_sample)
query_tool = synalinks.Tool(run_sql_query)

# Build the agent using Functional API
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.FunctionCallingAgent(
    data_model=SQLResult,
    language_model=lm,
    tools=[schema_tool, sample_tool, query_tool],
    autonomous=True,      # Run until agent decides it's done
    max_iterations=10,    # Safety limit to prevent infinite loops
)(inputs)

sql_agent = synalinks.Program(
    inputs=inputs,
    outputs=outputs,
    name="sql_agent",
)
```

The `autonomous=True` setting allows the agent to make multiple tool calls
until it has gathered enough information. The `max_iterations` parameter
prevents runaway execution.

## Safety Considerations

The `run_sql_query` tool enforces read-only access through multiple layers:

1. **SELECT Only**: Rejects queries not starting with SELECT
2. **Keyword Filtering**: Blocks DROP, DELETE, INSERT, UPDATE, ALTER, etc.
3. **Read-Only Mode**: Uses `kb.query(sql, read_only=True)`

```python
@register_synalinks_serializable()
async def run_sql_query(sql_query: str):
    \"\"\"Execute a read-only SQL query.\"\"\"
    kb = get_knowledge_base()

    # Validate query is read-only
    query_upper = sql_query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed.", "success": False}

    # Check for dangerous keywords
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
    for keyword in dangerous:
        if keyword in query_upper:
            return {"error": f"Forbidden keyword: {keyword}", "success": False}

    results = await kb.query(sql_query, read_only=True)
    return {"success": True, "results": results, "row_count": len(results)}
```

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

- **Dynamic Schema Discovery**: Use `kb.get_symbolic_data_models()` to make
  tools adaptable to any database structure without hardcoding.

- **Autonomous Reasoning**: The `FunctionCallingAgent` with `autonomous=True`
  iteratively calls tools until it can answer the question.

- **Safety First**: Always validate SQL queries and use read-only mode to
  prevent accidental data modifications.

- **Transparent Outputs**: Include the generated SQL in the output so users
  can verify and learn from the agent's reasoning.

- **Tool Serialization**: Use `@register_synalinks_serializable()` on tool
  functions to enable program saving and loading.

## API References

- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agent%20Modules/FunctionCallingAgent%20module/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/)
- [Tool](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Tool%20module/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks
from synalinks.src.saving.object_registration import register_synalinks_serializable

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
# Global Knowledge Base (set in main)
# =============================================================================

_knowledge_base = None


def get_knowledge_base():
    """Get the global knowledge base instance."""
    global _knowledge_base
    return _knowledge_base


# =============================================================================
# SQL Agent Tools
# =============================================================================


@register_synalinks_serializable()
async def get_database_schema():
    """Get the complete database schema including all tables and their columns.

    Returns a list of all tables with their column names and types.
    Use this tool first to understand what data is available before writing queries.
    """
    kb = get_knowledge_base()
    symbolic_models = kb.get_symbolic_data_models()

    schema_info = []
    for model in symbolic_models:
        schema = model.get_schema()
        table_name = schema.get("title", "Unknown")
        properties = schema.get("properties", {})

        columns = []
        for col_name, col_info in properties.items():
            col_type = col_info.get("type", "unknown")
            col_desc = col_info.get("description", "")
            columns.append(f"  - {col_name} ({col_type}): {col_desc}")

        schema_info.append(f"Table: {table_name}\n" + "\n".join(columns))

    return {
        "schema": "\n\n".join(schema_info),
        "table_count": len(symbolic_models),
    }


@register_synalinks_serializable()
async def get_table_sample(table_name: str, limit: int, offset: int):
    """Get a sample of rows from a specific table to understand the data format.

    Args:
        table_name (str): The name of the table to sample.
        limit (int): Number of sample rows to return (recommended: 3-5).
        offset (int): Number of rows to skip before returning results (use 0 for start).
    """
    kb = get_knowledge_base()
    symbolic_models = kb.get_symbolic_data_models()

    # Find the matching symbolic model by table name
    target_model = None
    available_tables = []
    for model in symbolic_models:
        schema = model.get_schema()
        name = schema.get("title", "Unknown")
        available_tables.append(name)
        if name == table_name:
            target_model = model
            break

    if target_model is None:
        return {"error": f"Table '{table_name}' not found. Available: {available_tables}"}

    try:
        results = await kb.getall(target_model, limit=limit, offset=offset)
        return {
            "table": table_name,
            "sample_data": [dict(r) for r in results],
            "row_count": len(results),
        }
    except Exception as e:
        return {"error": str(e)}


@register_synalinks_serializable()
async def run_sql_query(sql_query: str):
    """Execute a read-only SQL query and return the results.

    Args:
        sql_query (str): A SELECT SQL query to execute. Only SELECT queries are allowed.

    Important:
        - Only SELECT queries are permitted for safety
        - Use get_database_schema first to discover available tables
        - Use proper JOIN syntax for multi-table queries
        - Include LIMIT clause for large result sets
    """
    kb = get_knowledge_base()

    # Validate query is read-only
    query_upper = sql_query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return {
            "error": "Only SELECT queries are allowed.",
            "success": False,
        }

    # Check for dangerous patterns
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
    for keyword in dangerous:
        if keyword in query_upper:
            return {"error": f"Forbidden keyword: {keyword}", "success": False}

    try:
        results = await kb.query(sql_query, read_only=True)
        return {
            "success": True,
            "query": sql_query,
            "row_count": len(results),
            "results": results,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "query": sql_query}


# =============================================================================
# Database Setup
# =============================================================================


async def populate_knowledge_base(kb):
    """Populate the knowledge base with sample data."""

    # Sample customers
    customers = [
        Customer(
            id="C001",
            name="Alice Johnson",
            email="alice@example.com",
            country="USA"
        ),
        Customer(
            id="C002",
            name="Bob Smith",
            email="bob@example.com",
            country="UK",
        ),
        Customer(
            id="C003",
            name="Carlos Garcia",
            email="carlos@example.com",
            country="Spain",
        ),
        Customer(
            id="C004",
            name="Diana Chen",
            email="diana@example.com",
            country="China",
        ),
        Customer(
            id="C005",
            name="Emma Wilson",
            email="emma@example.com",
            country="Canada",
        ),
    ]

    # Sample products
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
        Product(
            id="P007",
            name="Desk Lamp",
            category="Office",
            price=45.99,
            stock=120,
        ),
        Product(
            id="P008", 
            name="Notebook Set",
            category="Office",
            price=12.99,
            stock=500,
        ),
    ]

    # Sample orders
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

    # Store all data
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
    global _knowledge_base

    load_dotenv()
    synalinks.clear_session()

    # -------------------------------------------------------------------------
    # Create Knowledge Base
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("SQL Agent with Knowledge Base")
    print("=" * 60)

    db_path = "guides/sql_agent.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\nCreating knowledge base...")
    _knowledge_base = synalinks.KnowledgeBase(
        uri=f"duckdb://{db_path}",
        data_models=[Customer, Product, SalesOrder],
        wipe_on_start=True,
        name="sql_agent_kb",
    )

    print("Populating with sample data...")
    await populate_knowledge_base(_knowledge_base)

    # -------------------------------------------------------------------------
    # Create SQL Agent
    # -------------------------------------------------------------------------
    print("\nBuilding SQL agent...")

    lm = synalinks.LanguageModel(model="openai/gpt-4.1-mini")

    # Wrap tools
    schema_tool = synalinks.Tool(get_database_schema)
    sample_tool = synalinks.Tool(get_table_sample)
    query_tool = synalinks.Tool(run_sql_query)

    # Build the SQL agent
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=SQLResult,
        language_model=lm,
        tools=[schema_tool, sample_tool, query_tool],
        autonomous=True,
        max_iterations=10,
    )(inputs)

    sql_agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="sql_agent",
        description="An agent that answers questions about data using SQL queries",
    )

    sql_agent.summary()

    # -------------------------------------------------------------------------
    # Demo Queries
    # -------------------------------------------------------------------------
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
            print(f"Query: {result['query']}")
            print(f"Answer: {result['answer']}")
            print(f"SQL: {result['sql_query']}")
        except Exception as e:
            print(f"Error: {e}")

        print()

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
