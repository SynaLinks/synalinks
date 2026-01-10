"""
# RAG Agent

This example demonstrates how to build a RAG Agent - an autonomous agent that
can search a knowledge base and use other tools to answer complex questions.
Unlike a simple RAG pipeline, a RAG agent can decide when to search, what to
search for, and can combine multiple searches with other tools.

## Why RAG Agents?

```mermaid
graph TD
    A[Question] --> B[Agent]
    B --> C{Need to Search?}
    C -->|Yes| D[search_documents]
    D --> E[Results]
    E --> B
    C -->|No| F{Use Calculator?}
    F -->|Yes| G[calculate]
    G --> B
    F -->|No| H[Final Answer]
```

Traditional RAG pipelines always retrieve documents, even for simple questions.
A RAG agent is smarter:

- Decides IF retrieval is needed
- Can reformulate queries for better search results
- Can perform multiple searches and combine results
- Can use other tools (calculator, web search, etc.) alongside retrieval

## Building a RAG Agent

```python
# Define the search tool
@synalinks.saving.register_synalinks_serializable()
async def search_knowledge_base(query: str):
    \"\"\"Search the knowledge base for documents relevant to the query.

    Use this tool to find information about company policies, procedures,
    products, or any documented knowledge.

    Args:
        query (str): The search query describing what information you need.
    \"\"\"
    results = await knowledge_base.hybrid_search(query, k=3)
    return {"documents": results}

# Create the agent
inputs = synalinks.Input(data_model=synalinks.ChatMessages)
outputs = await synalinks.FunctionCallingAgent(
    tools=[synalinks.Tool(search_knowledge_base)],
    language_model=language_model,
    autonomous=True,
    max_iterations=5,
)(inputs)
```

### Key Takeaways

- **Autonomous Decision Making**: The agent decides when to search and what
    queries to use.
- **Multi-Tool Support**: Combine document search with other tools like
    calculators or APIs.
- **Iterative Reasoning**: The agent can search multiple times and refine
    its understanding.
- **Conversational**: Maintains context across multiple turns.

## Program Visualization

![rag_agent](../assets/examples/rag_agent.png)

## API References

- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Bases%20API/)
- [ChatMessages (Base DataModels)](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/The%20Base%20DataModels/)
- [EmbeddingModel](https://synalinks.github.io/synalinks/Synalinks%20API/Embedding%20Models%20API/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Define the data models
# =============================================================================


class Document(synalinks.DataModel):
    """A document stored in the knowledge base."""

    id: str = synalinks.Field(
        description="Unique document identifier",
    )
    title: str = synalinks.Field(
        description="Document title",
    )
    content: str = synalinks.Field(
        description="The main text content of the document",
    )
    category: str = synalinks.Field(
        description="Document category",
    )


# Global knowledge base reference (will be set in main)
_knowledge_base = None


# =============================================================================
# Define the tools for the agent
# =============================================================================


@synalinks.saving.register_synalinks_serializable()
async def search_knowledge_base(query: str):
    """Search the knowledge base for documents relevant to the query.

    Use this tool to find information about company policies, procedures,
    products, or any documented knowledge. Returns the most relevant documents.

    Args:
        query (str): The search query describing what information you need.
            Be specific and use relevant keywords.
    """
    global _knowledge_base

    if _knowledge_base is None:
        return {"error": "Knowledge base not initialized"}

    results = await _knowledge_base.hybrid_search(query, k=30)
    return {"documents": results}


@synalinks.saving.register_synalinks_serializable()
async def calculate(expression: str):
    """Perform mathematical calculations.

    Use this for any math operations like addition, multiplication,
    percentages, or complex expressions.

    Args:
        expression (str): A mathematical expression to evaluate,
            e.g., '100 * 0.15' or '(50 + 30) / 2'.
    """
    # Validate expression
    allowed_chars = "0123456789+-*/().% "
    if not all(c in allowed_chars for c in expression):
        return {
            "status": "error",
            "message": "Invalid characters in expression",
            "result": None,
        }

    try:
        # Handle percentage
        expr = expression.replace("%", "/100")
        result = eval(expr, {"__builtins__": {}}, {})
        return {
            "status": "success",
            "expression": expression,
            "result": round(float(result), 2),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "result": None,
        }


@synalinks.saving.register_synalinks_serializable()
async def get_current_date():
    """Get the current date and time.

    Use this when you need to know today's date or the current time.
    This function takes no arguments.
    """
    from datetime import datetime

    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="rag_business_agent",
    )

    global _knowledge_base

    # Initialize models
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    embedding_model = synalinks.EmbeddingModel(
        model="openai/text-embedding-3-small",
    )

    # Clean up any existing database
    db_path = "./examples/rag_agent.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # ==========================================================================
    # Step 1: Create and Populate Knowledge Base
    # ==========================================================================
    print("Step 1: Creating Knowledge Base")
    print("=" * 60)

    _knowledge_base = synalinks.KnowledgeBase(
        uri=f"duckdb://{db_path}",
        data_models=[Document],
        embedding_model=embedding_model,
        metric="cosine",
    )

    # Sample company documents
    documents = [
        Document(
            id="policy-001",
            title="Remote Work Policy",
            content="""
            Employees may work remotely up to 3 days per week with manager approval.
            Remote work requests must be submitted at least 48 hours in advance.
            Employees must be available during core hours (10 AM - 4 PM).
            A reliable internet connection and dedicated workspace are required.
            Equipment allowance of $500 is provided for home office setup.
            """,
            category="HR Policy",
        ),
        Document(
            id="policy-002",
            title="Expense Reimbursement Policy",
            content="""
            Business expenses must be submitted within 30 days of purchase.
            Receipts are required for all expenses over $25.
            Travel expenses include flights, hotels, meals, and transportation.
            Daily meal allowance: $75 for domestic travel, $100 for international.
            Hotel booking limit: $200/night domestic, $300/night international.
            Manager approval required for expenses over $500.
            """,
            category="Finance Policy",
        ),
        Document(
            id="policy-003",
            title="Vacation and PTO Policy",
            content="""
            Full-time employees receive 20 days of PTO per year.
            PTO accrues at 1.67 days per month.
            Unused PTO can be carried over (maximum 5 days).
            Vacation requests should be submitted 2 weeks in advance.
            Holiday schedule includes 10 paid company holidays.
            Sick leave is separate from PTO: 10 days per year.
            """,
            category="HR Policy",
        ),
        Document(
            id="product-001",
            title="Product Pricing - Enterprise Plan",
            content="""
            Enterprise Plan: $99/user/month (billed annually).
            Minimum 50 users required for Enterprise Plan.
            Volume discounts available:
            - 50-99 users: 10% discount
            - 100-249 users: 15% discount
            - 250+ users: 20% discount
            Enterprise features include: SSO, advanced analytics, dedicated support,
            custom integrations, and 99.9% SLA.
            """,
            category="Product",
        ),
        Document(
            id="product-002",
            title="Product Pricing - Startup Plan",
            content="""
            Startup Plan: $29/user/month (billed monthly) or $25/user/month (annual).
            Available for teams of 5-49 users.
            Includes: Basic features, email support, 99% uptime SLA.
            Free trial: 14 days with full feature access.
            No setup fees or long-term contracts required.
            """,
            category="Product",
        ),
        Document(
            id="support-001",
            title="Customer Support Hours",
            content="""
            Standard Support: Monday-Friday, 9 AM - 6 PM (local time).
            Enterprise Support: 24/7 with dedicated account manager.
            Average response times:
            - Critical issues: 1 hour (Enterprise), 4 hours (Standard)
            - High priority: 4 hours (Enterprise), 24 hours (Standard)
            - Normal priority: 24 hours (Enterprise), 72 hours (Standard)
            Support channels: Email, chat, phone (Enterprise only).
            """,
            category="Support",
        ),
    ]

    print("Storing documents...")
    for doc in documents:
        await _knowledge_base.update(doc.to_json_data_model())
        print(f"  - {doc.title}")

    # ==========================================================================
    # Step 2: Create RAG Agent
    # ==========================================================================
    print("\nStep 2: Creating RAG Agent")
    print("=" * 60)

    # Define tools
    tools = [
        synalinks.Tool(search_knowledge_base),
        synalinks.Tool(calculate),
        synalinks.Tool(get_current_date),
    ]

    # Create the agent
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)
    outputs = await synalinks.FunctionCallingAgent(
        tools=tools,
        language_model=language_model,
        autonomous=True,
        max_iterations=5,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="rag_agent",
        description="An agent that can search documents and perform calculations",
    )

    # Plot the agent
    synalinks.utils.plot_program(
        agent,
        to_folder="examples",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    print("RAG Agent created!")

    # ==========================================================================
    # Step 3: Test the Agent
    # ==========================================================================
    print("\nStep 3: Testing RAG Agent")
    print("=" * 60)

    test_questions = [
        "What is the remote work policy?",
        "How much PTO do employees get per year?",
        "If I have 100 users on the Enterprise plan, what would be the monthly cost per user after discount?",
        "What's the meal allowance for a 5-day international trip?",
        "What day is it today?",
        "Compare the Startup and Enterprise plans.",
    ]

    for question in test_questions:
        print(f"\n{'─' * 60}")
        print(f"User: {question}")
        print(f"{'─' * 60}")

        # Create message
        messages = synalinks.ChatMessages(
            messages=[synalinks.ChatMessage(role="user", content=question)]
        )

        # Get response
        result = await agent(messages)

        # Extract assistant's final response
        response_messages = result.get("messages", [])
        if response_messages:
            # Get the last assistant message
            for msg in reversed(response_messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    print(f"\nAgent: {msg.get('content')}")
                    break

    # ==========================================================================
    # Step 4: Multi-turn Conversation
    # ==========================================================================
    print("\n\nStep 4: Multi-turn Conversation")
    print("=" * 60)

    conversation = [
        "What is the daily meal allowance for travel?",
        "What about for international trips?",
        "If I'm traveling for 3 days internationally, what's my total meal budget?",
    ]

    messages = []
    for user_msg in conversation:
        print(f"\n{'─' * 60}")
        print(f"User: {user_msg}")

        # Add user message to history
        messages.append(synalinks.ChatMessage(role="user", content=user_msg))

        # Get response
        chat_messages = synalinks.ChatMessages(messages=messages)
        result = await agent(chat_messages)

        # Extract and display assistant response
        response_messages = result.get("messages", [])
        assistant_response = None
        for msg in reversed(response_messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                assistant_response = msg.get("content")
                print(f"\nAgent: {assistant_response}")
                break

        # Add assistant response to history for next turn
        if assistant_response:
            messages.append(
                synalinks.ChatMessage(
                    role="assistant",
                    content=assistant_response,
                ),
            )

    print("\n\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
