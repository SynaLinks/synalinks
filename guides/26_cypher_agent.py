# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Cypher Agent

[Guide 23](https://synalinks.github.io/synalinks/guides/SQL%20Agent/) built an **SQL Agent** — an agent whose tools let
it discover a schema, sample rows, and run read-only `SELECT` queries over a
tabular `KnowledgeBase`. This guide builds the **graph** counterpart. A
**Cypher Agent** answers questions over a *knowledge graph* — nodes connected
by typed edges — by writing read-only [Cypher](https://opencypher.org/)
queries.

## Tables vs. graphs

A tabular store answers questions about *rows and columns*: filters,
aggregations, joins between a handful of tables. A graph store answers
questions about *connections*: who is linked to whom, and through what path.

The classic example is the **friend-of-friend** query. In SQL that is a chain
of self-joins on a `friendships` table; the more hops, the more joins, and the
SQL grows with the question. In a graph it is a single pattern, and the number
of hops is just the length of the pattern:

```cypher
MATCH (a:Person {name: 'Alice'})-[:Knows]->(f:Person)-[:LivesIn]->(c:City {name: 'Paris'})
RETURN f.name
```

When your questions are about *relationships and paths*, reach for a Cypher
agent; when they are about *typed tabular aggregates*, reach for an SQL agent.

## The three tools

`synalinks.CypherAgent` wraps a `FunctionCallingAgent` and pre-wires three
tools, all bound to a single graph `KnowledgeBase`:

| Tool | Purpose |
|------|---------|
| `get_graph_schema()` | List every node and relation label, with their properties. |
| `get_node_sample(label, limit, offset)` | Fetch a few nodes of a label to see the data shape. Page-bounded by the agent's `k`. |
| `run_cypher_query(cypher_query)` | Execute a read-only `MATCH ... RETURN` query. Result sets are capped at `k` rows. |

The default instructions are seeded with the graph's node and relation labels,
so the agent knows what is available without spending a turn on
`get_graph_schema` for trivial questions.

## Defining the graph

A graph `KnowledgeBase` is constructed with `graph_uri=` (a
[Ladybug](https://github.com/SynaLinks/ladybug) store here) plus `entity_models`
(the node types) and `relation_models` (the edge types). Entities subclass
`synalinks.Entity` and relations subclass `synalinks.Relation`, which carries a
typed `subj` (source node) and `obj` (target node).

Give each model a default `label` (matching its class name) so you can build
instances without repeating it:

```python
class Person(synalinks.Entity):
    label: str = synalinks.Field(default="Person", description="The entity label")
    name: str = synalinks.Field(description="Person name")

class LivesIn(synalinks.Relation):
    label: str = synalinks.Field(default="LivesIn", description="The relation label")
    subj: Person
    obj: City
```

## Safety

Only **read-only** Cypher runs. The adapter scans the query (after stripping
comments and string literals) and rejects any write/admin keyword — `CREATE`,
`MERGE`, `SET`, `DELETE`, `DETACH`, `REMOVE`, `DROP`, `ALTER`, `COPY`,
`INSTALL`, `LOAD` — so the agent can explore the graph but never mutate it or
read files.

## A note on the model

Function-calling agents need a model that emits **native tool calls**. Capable
local models such as `qwen3` do; some smaller ones do not and will describe a
tool call in prose instead of emitting one. Prefer a tool-capable model (or a
hosted one) for the Cypher agent.

### Key Takeaways

- **Graphs for relationships**: pick a Cypher agent when answers follow edges
  and paths; pick an SQL agent for typed tabular aggregates.
- **Three tools**: `get_graph_schema`, `get_node_sample`, `run_cypher_query`,
  all bound to one graph `KnowledgeBase`.
- **Read-only**: writes/admin Cypher is rejected by the engine.
- **`data_model=`** makes the agent return a structured answer (here, the
  prose answer plus the Cypher it ran).

## API References

- [CypherAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/CypherAgent%20module/)
- [FunctionCallingAgent](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Agents%20Modules/FunctionCallingAgent%20module/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/Knowledge%20Base/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


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

    # `graph_uri=` selects the graph adapter; `:memory:` is ephemeral.
    knowledge_base = synalinks.KnowledgeBase(
        graph_uri="ladybug://:memory:",
        entity_models=[Person, City],
        relation_models=[LivesIn, Knows],
        embedding_model=synalinks.EmbeddingModel(model="ollama/all-minilm"),
    )

    await knowledge_base.update_relations(
        [
            LivesIn(subj=Person(name="Alice"), obj=City(name="Paris")),
            LivesIn(subj=Person(name="Bob"), obj=City(name="Paris")),
            LivesIn(subj=Person(name="Carol"), obj=City(name="London")),
            Knows(subj=Person(name="Alice"), obj=Person(name="Bob")),
        ]
    )

    # Function calling needs a tool-capable model; qwen3 emits native tool calls.
    language_model = synalinks.LanguageModel(model="ollama/qwen3:8b")

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

    result = await agent(Query(query="Who lives in Paris?"))
    print("Answer:", result.get("answer"))
    print("Cypher:", result.get("cypher_query"))


if __name__ == "__main__":
    asyncio.run(main())
