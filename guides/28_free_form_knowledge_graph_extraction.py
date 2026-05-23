"""
# Free-form Knowledge Graph Extraction

The [previous guide](https://synalinks.github.io/synalinks/guides/Knowledge%20Graph%20Extraction/) built graphs
against a *fixed* graph schema: every node type was a subclass with a
`Literal` label (`label: Literal["City"]`), so the model could only emit
the entity and relation types you declared. That is the right tool when
you know the shape of your domain up front and want to *store and query*
it in a typed graph.

But sometimes you **don't** know the shape in advance. You're pointed at
an arbitrary corpus — a pile of articles, a research paper, a wiki — and
you want the model to surface *whatever* entities and relations it finds,
inventing the vocabulary as it goes. That is **free-form extraction**:
the schema constrains the *structure* (there are entities, there are
relations, each has a label) but leaves the *labels open*.

The mechanism is one small change. Instead of pinning `label` to a
`Literal`, you leave it as the plain `str` the base classes already
declare. The discriminated union of fixed types becomes a single
generic node type and a single generic edge type; the `label` field
carries the type the model discovered, as data.

```mermaid
graph LR
    A["Document"] --> B["Generator<br/>(generic schema)"]
    B --> C["KnowledgeGraph<br/>open labels"]
    C --> D["EmbedKnowledge"]
    D --> E["UpdateKnowledge"]
    E --> F[("Graph store<br/>tables created on the fly")]
```

## Constrained vs. Free-form

The two approaches share every module — the only difference is the
schema you hand the `Generator`.

| | Constrained (Guide 27) | Free-form (this guide) |
|---|---|---|
| Label | `Literal["City"]` (fixed) | `str` (open) |
| Node types | One subclass per type | One generic node |
| Graph schema | You define it | The model discovers it |
| Decoding | Picks among known types | Emits any label string |
| Best for | Known domain, typed queries | Exploration, unknown corpora |
| Risk | Misses types you didn't model | Label drift / inconsistency |

Neither is "better" — they're the two ends of a dial. Free-form
maximizes *coverage* (nothing is excluded by your schema) at the cost of
*consistency* (the model might say `"Person"` in one document and
`"Human"` in the next). Constrained is the reverse. A common pattern is
to start free-form to *learn* what's in a corpus, then promote the
labels you care about into a constrained schema.

## The Generic Schema

The base `Entity`, `Relation`, and `KnowledgeGraph` already carry the
right shape. You subclass `Entity` only to add the properties you want
on every node (here `name` and `description`), leaving `label` as the
inherited open string:

```python
from typing import List
import synalinks

class Node(synalinks.Entity):
    # `label` is inherited as a plain `str` — the model fills it with
    # the type it discovers ("Person", "Organization", "Concept", ...).
    name: str = synalinks.Field(description="The entity's name / identifier.")
    description: str = synalinks.Field(description="A short description.")

class Edge(synalinks.Relation):
    # `label` inherited as `str` (the relation type). Endpoints are
    # generic Nodes so they carry the same open shape.
    subj: Node = synalinks.Field(description="Source entity.")
    obj: Node = synalinks.Field(description="Target entity.")

class Graph(synalinks.KnowledgeGraph):
    entities: List[Node] = synalinks.Field(description="Entities found in the text.")
    relations: List[Edge] = synalinks.Field(description="Relations found in the text.")
```

Compared with Guide 27, there is no `Union` of concrete types and no
`Literal` anywhere. `name` is still the first content field, so it is
still the primary key — two mentions of `"Marie Curie"` collapse onto
one node regardless of the label the model chose.

## Why Storage Still Works Without Declaring Models

Here is the part that makes free-form practical: you store the result
**without telling the knowledge base which labels exist**. The graph
store creates a node table the first time it sees a label, and an edge
table the first time it sees a relation type — inferring each table's
columns from the data itself.

```python
knowledge_base = synalinks.KnowledgeBase(
    graph_uri="ladybug://discovered.lb",
    embedding_model=embedding_model,
    # No entity_models / relation_models: the labels aren't known yet.
)
```

This is the division of labor to keep in mind: **data models constrain
the schema at the generator** (constrained decoding), while the **store
materializes whatever arrives**. In Guide 27 the two happened to line up
(you declared `City`, the store made a `City` table). Here they
deliberately don't — the generator is unconstrained, and the store
follows the data.

A practical consequence: a graph store's edge table fixes its endpoint
types when first created, so a relation label that connects *consistent*
types ("`BORN_IN`" always Person→Place) stores cleanly, while one that
connects wildly different type pairs under the same label is the case to
watch. In practice relation types are consistent, so this is rarely a
problem — but it's the reason a little label hygiene in your prompt pays
off.

## Extracting, Storing, and Reading Back

Extraction is one `Generator` call against the generic `Graph`, then the
same `EmbedKnowledge` → `UpdateKnowledge` pair from Guide 27:

```python
inputs = synalinks.Input(data_model=Document)
graph = await synalinks.Generator(
    data_model=Graph,
    language_model=language_model,
    instructions=(
        "Extract every entity and the relations between them. Choose a "
        "concise UPPER_SNAKE_CASE label for each relation and a "
        "PascalCase label for each entity type."
    ),
)(inputs)
embedded = await synalinks.EmbedKnowledge(
    embedding_model=embedding_model,
    in_mask=["name", "description"],
)(graph)
stored = await synalinks.UpdateKnowledge(knowledge_base=knowledge_base)(embedded)
```

Because the labels are discovered, **introspection matters more** than
in the constrained case — you often don't know the node/edge tables
until after the run. `kb.cypher(...)` is the natural lens:

```python
# What entity types did the model actually invent?
labels = await knowledge_base.cypher(
    "MATCH (n) RETURN DISTINCT label(n) AS type ORDER BY type"
)

# The neighbourhood around a specific entity (read-back keeps the
# `name` / `description` properties even though no model was declared).
subgraph = await knowledge_base.local_graph_search(
    "Marie Curie", label="Person", max_hops=2, k=1,
)
```

The same multi-stage strategies and orphan-node reasoning from Guide 27
apply unchanged — they're about *how many calls* you spend, orthogonal
to whether the labels are fixed or open.

## Key Takeaways

- **Free-form extraction** keeps the structure (entities, relations,
  labels) but leaves the **labels open** — drop the `Literal`, use the
  inherited `label: str`, and collapse the type union into one generic
  `Node` and one generic `Edge`.
- The model **discovers the graph schema**: maximal coverage, at the cost of
  label consistency. Start free-form to learn a corpus; promote the
  labels worth keeping into a constrained schema (Guide 27) later.
- **No `entity_models` / `relation_models` needed.** The store creates
  node and relation tables on demand, inferring columns from the data —
  the data models are where *decoding* is constrained, not where storage
  is.
- The **first content field is still the primary key**, so dedup works
  the same; read-back via `cypher` or `local_graph_search` preserves the
  discovered properties.
- Lean on **introspection** (`MATCH (n) RETURN DISTINCT label(n)`) since
  you don't know the labels up front, and nudge **label hygiene** in the
  prompt to keep a relation type connecting consistent endpoint types.

## API References

- [Entity / Relation / KnowledgeGraph](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/Knowledge%20Data%20Models/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/Knowledge%20Base/)
- [EmbedKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/EmbedKnowledge%20module/)
- [UpdateKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/UpdateKnowledge%20module/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
"""

# --8<-- [start:source]
import asyncio
from typing import List

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Generic (free-form) schema: open labels, no Literal discriminators.
# =============================================================================


class Document(synalinks.DataModel):
    """A piece of unstructured text to extract a graph from."""

    text: str = synalinks.Field(description="The raw document text")


class Node(synalinks.Entity):
    # `label` is inherited from Entity as a plain `str`. The model fills
    # it with the entity TYPE it discovers ("Person", "City", ...).
    name: str = synalinks.Field(description="The entity's name / identifier.")
    description: str = synalinks.Field(description="A short description of the entity.")


class Edge(synalinks.Relation):
    # `label` inherited as `str` (the relation type); generic endpoints.
    subj: Node = synalinks.Field(description="The source entity.")
    obj: Node = synalinks.Field(description="The target entity.")


class Graph(synalinks.KnowledgeGraph):
    entities: List[Node] = synalinks.Field(
        description="Every entity mentioned in the text, with a discovered label.",
    )
    relations: List[Edge] = synalinks.Field(
        description="Every relation between the entities, with a discovered label.",
    )


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    language_model = synalinks.LanguageModel(model="ollama/llama3.2:latest")
    embedding_model = synalinks.EmbeddingModel(model="ollama/mxbai-embed-large")

    document = Document(
        text=(
            "Marie Curie was a physicist and chemist born in Warsaw. She "
            "conducted pioneering research on radioactivity at the University "
            "of Paris, and won the Nobel Prize in Physics in 1903 together "
            "with Pierre Curie and Henri Becquerel."
        ),
    )

    # The knowledge base declares NO entity_models / relation_models — the
    # labels aren't known until the model discovers them, and the store
    # creates the tables on demand.
    knowledge_base = synalinks.KnowledgeBase(
        graph_uri="ladybug://:memory:",
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=True,
    )

    # -------------------------------------------------------------------------
    # Extract a free-form graph, embed it, and store it
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Free-form extraction")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Document)
    graph = await synalinks.Generator(
        data_model=Graph,
        language_model=language_model,
        instructions=(
            "Extract every entity and every relation between them from the "
            "text. Pick a PascalCase label for each entity type (e.g. "
            "'Person', 'City', 'Award') and a concise PascalCase label "
            "for each relation (e.g. 'BornIn', 'Won')."
        ),
    )(inputs)
    embedded = await synalinks.EmbedKnowledge(
        embedding_model=embedding_model,
        in_mask=["name", "description"],
    )(graph)
    stored = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded)

    program = synalinks.Program(
        inputs=inputs,
        outputs=stored,
        name="free_form_kg_extraction",
        description="Extract an open-label knowledge graph and store it.",
    )

    await program(document)

    # -------------------------------------------------------------------------
    # Introspect: which labels did the model actually invent?
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Discovered entity types")
    print("=" * 60)

    types = await knowledge_base.cypher(
        "MATCH (n) RETURN DISTINCT label(n) AS type ORDER BY type"
    )
    for row in types:
        print(f"  - {row['type']}")

    print("\n" + "=" * 60)
    print("Discovered relations")
    print("=" * 60)

    edges = await knowledge_base.cypher(
        "MATCH (a)-[r]->(b) RETURN a.name AS subj, label(r) AS rel, b.name AS obj"
    )
    for row in edges:
        print(f"  - ({row['subj']}) -[{row['rel']}]-> ({row['obj']})")

    # -------------------------------------------------------------------------
    # Query: the neighbourhood around an entity (properties preserved)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Local graph search around 'Marie Curie'")
    print("=" * 60)

    subgraph = await knowledge_base.local_graph_search(
        "Marie Curie",
        label="Person",
        max_hops=2,
        k=1,
    )
    for entity in subgraph.entities:
        print(f"  - {entity.label}: {entity.name}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
