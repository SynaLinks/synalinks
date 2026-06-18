"""
# Cypher Agent

The `SQLAgent` ([Code Example](https://synalinks.github.io/synalinks/Code Examples/SQL%20Agent/)) answers questions over a
*tabular* knowledge base. The **Cypher Agent** is its graph counterpart: it
answers questions over a **knowledge graph** (nodes and edges) by writing
read-only [Cypher](https://opencypher.org/) queries.

`synalinks.CypherAgent` is a thin specialization of
[`FunctionCallingAgent`](https://synalinks.github.io/synalinks/Code Examples/Autonomous%20Agent/) that pre-wires three tools,
all bound to a `KnowledgeBase` with a graph adapter:

| Tool | Purpose |
|------|---------|
| `get_graph_schema()` | List every node and relation label with their properties. |
| `get_node_sample(label, limit, offset)` | Fetch a few nodes of a label to see the data shape. Page-bounded by the agent's `k`. |
| `run_cypher_query(cypher_query)` | Execute a read-only `MATCH ... RETURN` query. Result sets are capped at `k` rows. |

The user asks a question in natural language; the agent discovers the schema,
writes Cypher, runs it, and summarizes the rows.

## When do you want one?

Graphs shine when the answer follows **relationships** — friend-of-friend,
shortest path, "which X connects to Y through Z". Those are multi-hop
traversals that an SQL agent would express as a chain of self-joins. With a
graph, the same question is a single `MATCH` pattern:

```cypher
MATCH (a:Person {name: 'Alice'})-[:Knows]->(p:Person)-[:LivesIn]->(c:City {name: 'Paris'})
RETURN p.name
```

## Safety

Only **read-only** Cypher runs. The graph adapter scans the query and rejects
any write/admin keyword (`CREATE`, `MERGE`, `SET`, `DELETE`, `DETACH`,
`REMOVE`, `DROP`, `ALTER`, `COPY`, `INSTALL`, `LOAD`) — so an LM can explore
the graph but never mutate it.

## API References

- [CypherAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/CypherAgent%20module/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/Knowledge%20Base/)
- [Program](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
"""

# --8<-- [start:source]
import asyncio

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Graph schema: two node labels and two relation labels.
# Each Entity/Relation declares a default `label` so instances can be built
# without repeating it; the value matches the class name (PascalCase).
# =============================================================================
class Person(synalinks.Entity):
    label: str = synalinks.Field(default="Person", description="The entity label")
    name: str = synalinks.Field(description="Person name")


class City(synalinks.Entity):
    label: str = synalinks.Field(default="City", description="The entity label")
    name: str = synalinks.Field(description="City name")


class LivesIn(synalinks.Relation):
    label: str = synalinks.Field(default="LivesIn", description="The relation label")
    subj: Person
    obj: City


class Knows(synalinks.Relation):
    label: str = synalinks.Field(default="Knows", description="The relation label")
    subj: Person
    obj: Person


class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="A natural-language question")


class CypherAnswer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer in natural language")
    cypher_query: str = synalinks.Field(description="The Cypher that produced it")


async def main():
    load_dotenv()
    synalinks.clear_session()

#     synalinks.enable_observability(
#         tracking_uri="http://localhost:5000",
#         experiment_name="cypher_agent",
#     )

    # A graph KnowledgeBase needs `graph_uri=` (Ladybug here) plus the entity
    # and relation models. `:memory:` keeps it ephemeral; swap in a file path
    # like "ladybug://./cypher_agent.lb" to persist.
    knowledge_base = synalinks.KnowledgeBase(
        graph_uri="ladybug://:memory:",
        entity_models=[Person, City],
        relation_models=[LivesIn, Knows],
        # The graph store dedups entities by embedding on insert; the choice is
        # incidental to Cypher querying itself.
        embedding_model=synalinks.EmbeddingModel(model="ollama/all-minilm"),
    )

    # Populate a tiny social graph.
    await knowledge_base.update_relations(
        [
            LivesIn(subj=Person(name="Alice"), obj=City(name="Paris")),
            LivesIn(subj=Person(name="Bob"), obj=City(name="Paris")),
            LivesIn(subj=Person(name="Carol"), obj=City(name="London")),
            Knows(subj=Person(name="Alice"), obj=Person(name="Bob")),
            Knows(subj=Person(name="Alice"), obj=Person(name="Carol")),
        ]
    )

    language_model = synalinks.LanguageModel(
        model="ollama/qwen3:8b",
    )

    # `data_model=CypherAnswer` forces the agent to return both a natural-
    # language answer and the Cypher it used.
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.CypherAgent(
        knowledge_base=knowledge_base,
        language_model=language_model,
        data_model=CypherAnswer,
        max_iterations=8,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="cypher_agent",
        description="A Cypher agent that answers questions about a knowledge graph.",
    )

    # A two-hop question: traverse Knows, then LivesIn.
    result = await agent(Query(query="Which people that Alice knows live in Paris?"))
    print("Answer:", result.get("answer"))
    print("Cypher:", result.get("cypher_query"))


if __name__ == "__main__":
    asyncio.run(main())
