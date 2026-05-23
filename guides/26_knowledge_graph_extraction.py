"""
# Knowledge Graph Extraction

[Guide 6](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/) stored *flat* records — one table per
`DataModel`, retrieved by full-text or vector search. That is enough
when the answer lives inside a single record ("what is the total of
invoice INV-2024-002?"). It is *not* enough when the answer lives in
the **connections between** records ("which landmarks are in the
capital of France?"). For that you need a **knowledge graph**: typed
**entities** (nodes) joined by typed **relations** (edges).

This guide is about getting that graph *out of unstructured text*.
The model reads a document and emits a graph whose shape you fixed in
advance — every node and edge validated against a schema by
**constrained JSON decoding**, so the LM cannot invent a label or drop
a required field. Once extracted, the graph is embedded and stored in a
graph-backed `KnowledgeBase`, ready for the graph retrieval covered at
the end.

```mermaid
graph LR
    A["Document"] --> B["Generator(s)<br/>constrained decoding"]
    B --> C["KnowledgeGraph<br/>(entities + relations)"]
    C --> D["EmbedKnowledge"]
    D --> E["UpdateKnowledge"]
    E --> F[("Graph store")]
```

The single most important idea: **extraction is not one fixed
pipeline.** A frontier model can read a paragraph and emit the whole
graph in one call; a small local model does far better when you split
the job into narrow sub-tasks and recombine the pieces. Synalinks lets
you dial that granularity up or down without changing the schema — the
same `City` / `IsCapitalOf` definitions back a one-call extractor and a
ten-call one. We build up from the simplest strategy to the most
robust, and end with how to pick.

## The Graph Schema

Two base classes describe a property graph:

- **`Entity`** — a node. It carries a `label` (the node *type*) plus
  whatever fields that type needs.
- **`Relation`** — a directed edge. It carries a `label`, a `subj`
  (source entity), and an `obj` (target entity).

You subclass them, pinning `label` to a `Literal` so it becomes a
**discriminator** the decoder must match exactly, and typing each
relation's `subj` / `obj` to the concrete entity classes it connects:

```python
from typing import Literal
import synalinks

class Country(synalinks.Entity):
    label: Literal["Country"]
    name: str = synalinks.Field(description="Country name, e.g. 'France'.")

class City(synalinks.Entity):
    label: Literal["City"]
    name: str = synalinks.Field(description="City name, e.g. 'Paris'.")

class Landmark(synalinks.Entity):
    label: Literal["Landmark"]
    name: str = synalinks.Field(description="A notable place, e.g. 'Eiffel Tower'.")

class IsCapitalOf(synalinks.Relation):
    label: Literal["IsCapitalOf"]
    subj: City        # a City ...
    obj: Country      # ... is the capital of a Country

class IsLocatedIn(synalinks.Relation):
    label: Literal["IsLocatedIn"]
    subj: Landmark    # a Landmark ...
    obj: City         # ... sits in a City
```

Two conventions worth burning in, both inherited from the KB:

- **The first content field is the primary key.** Here that is `name`
  — two extractions of `"Paris"` collapse onto one node instead of
  duplicating. Keep the identifying field first.
- **You do not declare an `embedding` field.** When the graph store has
  an embedding model, it adds the vector column (and its index)
  automatically; declaring your own would only get in the way.

Note that a `Relation` embeds the *full* `subj` and `obj` entities, not
just their ids. That single design choice is what makes the
relations-only strategy at the end possible.

## The Graph Knowledge Base

A graph-backed `KnowledgeBase` is opened with `graph_uri=` (instead of,
or alongside, the SQL `uri=`). The default backend is an embedded graph
store, so — like DuckDB in Guide 6 — there is no server to run.

```python
knowledge_base = synalinks.KnowledgeBase(
    graph_uri="ladybug://geography.lb",
    entity_models=[Country, City, Landmark],
    relation_models=[IsCapitalOf, IsLocatedIn],
    embedding_model=embedding_model,   # enables vector search + dedup
    metric="cosine",
)
```

`entity_models` declares the node tables, `relation_models` the edge
tables — the graph counterpart of Guide 6's `data_models`. The
`embedding_model` is what lets the store deduplicate near-identical
nodes and, later, answer similarity queries.

## Embedding and Storing a Graph

Two modules move an extracted graph into the store:

- **`EmbedKnowledge`** walks the graph and embeds the field(s) named in
  `in_mask`, attaching a vector to every entity. Embed the field that
  *identifies* the node (its `name`), not every field — that keeps the
  vector focused and cheap.
- **`UpdateKnowledge`** writes the embedded graph to the store,
  upserting nodes by primary key and creating the edges.

```python
embedded = await synalinks.EmbedKnowledge(
    embedding_model=embedding_model,
    in_mask=["name"],
)(knowledge_graph)

stored = await synalinks.UpdateKnowledge(
    knowledge_base=knowledge_base,
)(embedded)
```

Everything before these two steps is *how you produce the graph*. That
is where the strategies diverge.

## Strategy 1 — One-Stage Extraction

Ask for the entire graph in a single `Generator` call. Define a
`KnowledgeGraph` subclass whose `entities` and `relations` are `Union`s
over your concrete types, and the constrained decoder fills both lists
at once:

```python
from typing import List, Union

class GeographyGraph(synalinks.KnowledgeGraph):
    entities: List[Union[Country, City, Landmark]] = synalinks.Field(
        description="Every country, city, and landmark mentioned.",
    )
    relations: List[Union[IsCapitalOf, IsLocatedIn]] = synalinks.Field(
        description="Every capital-of and located-in relation.",
    )

inputs = synalinks.Input(data_model=Document)
knowledge_graph = await synalinks.Generator(
    data_model=GeographyGraph,
    language_model=language_model,
)(inputs)
```

One call, minimal latency, simplest wiring. The cost: the model must
hold the *whole* extraction task in its head at once — identify every
entity type, infer every relation, and stay self-consistent. Frontier
models handle this well; smaller models start dropping entities and
hallucinating edges as the schema grows.

## Strategy 2 — Two-Stage Extraction

Split entity-finding from relation-finding. First extract the entities,
feed them *back in* alongside the document so the second call has the
node list to connect, then extract the relations:

```python
class GeographyEntities(synalinks.Entities):
    entities: List[Union[Country, City, Landmark]] = synalinks.Field(
        description="Every country, city, and landmark mentioned.",
    )

class GeographyRelations(synalinks.Relations):
    relations: List[Union[IsCapitalOf, IsLocatedIn]] = synalinks.Field(
        description="Every relation between the entities.",
    )

inputs = synalinks.Input(data_model=Document)

entities = await synalinks.Generator(
    data_model=GeographyEntities,
    language_model=language_model,
)(inputs)

# `inputs & entities` is a logical AND: it merges the document and the
# extracted entities into one data model (see the Data Model Operators
# example), so the relation pass sees both.
inputs_and_entities = inputs & entities
relations = await synalinks.Generator(
    data_model=GeographyRelations,
    language_model=language_model,
)(inputs_and_entities)

# Merge the two halves back into a single graph-shaped data model.
knowledge_graph = entities & relations
```

Each call now reasons about *one* thing, which a mid-sized model does
more reliably than the all-at-once version. The `&` operator
(logical AND) is what stitches the stages together — the
[Data Model Operators example](https://synalinks.github.io/synalinks/Code%20Examples/Data%20Model%20Operators/)
covers it and its siblings in depth.

## Strategy 3 — Multi-Stage Extraction

For small local models, or wildly heterogeneous schemas, go further:
**one `Generator` per type.** Each call extracts a single entity or
relation kind, shrinking the task to its smallest unit, then you fuse
the results:

```python
class Cities(synalinks.Entities):
    entities: List[City] = synalinks.Field(description="Only cities.")

class Countries(synalinks.Entities):
    entities: List[Country] = synalinks.Field(description="Only countries.")

# ... one per entity type, and one per relation type ...

cities = await synalinks.Generator(data_model=Cities, language_model=lm)(inputs)
countries = await synalinks.Generator(data_model=Countries, language_model=lm)(inputs)
# ... etc.

# Fuse with logical OR so a single failed call doesn't sink the batch,
# then `.factorize()` to collapse the per-call lists into one.
entities = await synalinks.Or()([cities, countries, places])
entities = entities.factorize()
```

The choice between `synalinks.And()` and `synalinks.Or()` here is a
choice about **failure semantics**: `And` requires every branch to
succeed (all-or-nothing), while `Or` keeps whatever branches *did*
succeed (robust to a flaky call). `.factorize()` then merges the
several `Entities` results into a single deduplicated list. Maximum
accuracy per call and maximum resilience — at the price of many LM
round-trips.

## Strategy 4 — Relations-Only (Avoiding Orphan Nodes)

An **orphan node** is an entity connected to nothing. Graph retrieval —
the whole point of building a graph — works by *traversing edges*, so
orphans are dead weight: they can never be reached from a neighbour.

Because a `Relation` carries its `subj` and `obj` entities in full,
there is an elegant fix: **extract only the relations.** Every entity
then arrives already attached to at least one edge, so orphans are
impossible by construction:

```python
relations = await synalinks.Or()(
    [is_capital_of_relations, is_located_in_relations]
)
relations = relations.factorize()

embedded = await synalinks.EmbedKnowledge(
    embedding_model=embedding_model,
    in_mask=["name"],
)(relations)

stored = await synalinks.UpdateKnowledge(
    knowledge_base=knowledge_base,
)(embedded)
```

`UpdateKnowledge` unpacks each relation into its two endpoint nodes plus
the edge, so the graph is fully populated — just guaranteed
connected. Reach for this whenever you intend to *query* the graph by
traversal rather than look entities up one by one.

## Querying the Extracted Graph

Once stored, the graph answers connection-shaped questions the flat KB
could not. The retrieval surface is covered in the
[Knowledge Base guide](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/); the two graph-native entry
points are:

- **`kb.local_graph_search(query, label=..., max_hops=N)`** — vector-match
  seed entities, then return their `N`-hop neighbourhood as a subgraph.
  Entity-centric: *"what does the graph say around here?"*
- **`kb.cypher(query)`** — a read-only Cypher escape hatch for exact,
  hand-written traversals.

```python
# The local neighbourhood around the best match for "Paris".
subgraph = await knowledge_base.local_graph_search(
    "Paris", label="City", max_hops=2, k=1,
)

# Or an exact traversal: capitals and the country they head.
rows = await knowledge_base.cypher(
    "MATCH (c:City)-[:IsCapitalOf]->(n:Country) RETURN c.name, n.name"
)
```

## Choosing a Strategy

| Strategy | LM calls / doc | Best when | Watch out for |
|---|---|---|---|
| One-stage | 1 | Frontier model; small schema | Drops entities as schema grows |
| Two-stage | 2 | Mid-sized model; want entity/relation separation | Relation pass depends on entity pass |
| Multi-stage | many | Small local models; heterogeneous types | Cost and latency of many calls |
| Relations-only | per relation type | You will *query by traversal* | Entities never mentioned in a relation are skipped |

Start at the top. Move down only when evaluation shows the model
dropping or hallucinating parts of the graph — the schema never
changes, only how many calls you spend filling it. And remember the
generators are **trainable**: before adding stages, you can often close
the gap by optimizing the prompts of a simpler pipeline (see the
[Training guide](https://synalinks.github.io/synalinks/guides/Training/)).

## Key Takeaways

- A **knowledge graph** captures the *connections* flat records can't:
  typed `Entity` nodes joined by typed `Relation` edges, both fixed by
  schema and enforced through constrained JSON decoding.
- **Subclass `Entity` / `Relation`**, pinning `label` to a `Literal`
  and typing each relation's `subj` / `obj`. First content field is the
  primary key; the embedding column is added for you.
- Open a graph store with **`graph_uri=`**, declaring `entity_models`
  and `relation_models`; **`EmbedKnowledge`** then **`UpdateKnowledge`**
  embed and persist the extracted graph.
- Extraction granularity is a **dial**, not a fixed pipeline:
  **one-stage** (1 call) → **two-stage** (entities then relations,
  joined with `&`) → **multi-stage** (one call per type, fused with
  `And`/`Or` + `.factorize()`). Same schema throughout.
- **Relations carry their endpoints in full**, so extracting
  *relations only* guarantees a connected graph with no orphan nodes —
  the right default when you'll query by traversal.
- Query the result with **`local_graph_search`** (neighbourhood) or
  **`cypher`** (exact traversal).

## API References

- [Entity / Relation / KnowledgeGraph](https://synalinks.github.io/synalinks/Synalinks%20API/Data%20Models%20API/Knowledge%20Data%20Models/)
- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/Knowledge%20Base/)
- [EmbedKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/EmbedKnowledge%20module/)
- [UpdateKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/UpdateKnowledge%20module/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
"""

# --8<-- [start:source]
import asyncio
import os
from typing import List
from typing import Literal
from typing import Union

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Input + Graph Schema
# =============================================================================


class Document(synalinks.DataModel):
    """A piece of unstructured text to extract a graph from."""

    text: str = synalinks.Field(description="The raw document text")


class Country(synalinks.Entity):
    label: Literal["Country"]
    name: str = synalinks.Field(description="Country name, e.g. 'France'.")


class City(synalinks.Entity):
    label: Literal["City"]
    name: str = synalinks.Field(description="City name, e.g. 'Paris'.")


class Landmark(synalinks.Entity):
    label: Literal["Landmark"]
    name: str = synalinks.Field(description="A notable place, e.g. 'Eiffel Tower'.")


class IsCapitalOf(synalinks.Relation):
    label: Literal["IsCapitalOf"]
    subj: City
    obj: Country


class IsLocatedIn(synalinks.Relation):
    label: Literal["IsLocatedIn"]
    subj: Landmark
    obj: City


# One-stage: entities AND relations in a single schema.
class GeographyGraph(synalinks.KnowledgeGraph):
    entities: List[Union[Country, City, Landmark]] = synalinks.Field(
        description="Every country, city, and landmark mentioned in the text.",
    )
    relations: List[Union[IsCapitalOf, IsLocatedIn]] = synalinks.Field(
        description="Every capital-of and located-in relation between them.",
    )


# Two-stage: the entity and relation halves as separate schemas.
class GeographyEntities(synalinks.Entities):
    entities: List[Union[Country, City, Landmark]] = synalinks.Field(
        description="Every country, city, and landmark mentioned in the text.",
    )


class GeographyRelations(synalinks.Relations):
    relations: List[Union[IsCapitalOf, IsLocatedIn]] = synalinks.Field(
        description="Every relation between the entities in the text.",
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
            "France is a country in Western Europe. Its capital is Paris, "
            "a city on the Seine. The Eiffel Tower, a famous landmark, is "
            "located in Paris."
        ),
    )

    # -------------------------------------------------------------------------
    # One-stage extraction → embed → store
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("One-stage extraction")
    print("=" * 60)

    knowledge_base = synalinks.KnowledgeBase(
        graph_uri="ladybug://:memory:",
        entity_models=[Country, City, Landmark],
        relation_models=[IsCapitalOf, IsLocatedIn],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=True,
    )

    inputs = synalinks.Input(data_model=Document)
    knowledge_graph = await synalinks.Generator(
        data_model=GeographyGraph,
        language_model=language_model,
        instructions=(
            "Extract every country, city, and landmark, and the relations "
            "between them, from the document text."
        ),
    )(inputs)
    embedded = await synalinks.EmbedKnowledge(
        embedding_model=embedding_model,
        in_mask=["name"],
    )(knowledge_graph)
    stored = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded)

    one_stage = synalinks.Program(
        inputs=inputs,
        outputs=stored,
        name="one_stage_kg_extraction",
        description="Extract a geography knowledge graph in a single call.",
    )

    await one_stage(document)

    # -------------------------------------------------------------------------
    # Two-stage extraction (entities, then relations, merged with `&`)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Two-stage extraction")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Document)
    entities = await synalinks.Generator(
        data_model=GeographyEntities,
        language_model=language_model,
        instructions="Extract every country, city, and landmark from the text.",
    )(inputs)
    inputs_and_entities = inputs & entities
    relations = await synalinks.Generator(
        data_model=GeographyRelations,
        language_model=language_model,
        instructions="Extract the relations between the given entities.",
    )(inputs_and_entities)
    knowledge_graph = entities & relations
    embedded = await synalinks.EmbedKnowledge(
        embedding_model=embedding_model,
        in_mask=["name"],
    )(knowledge_graph)
    stored = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded)

    two_stage = synalinks.Program(
        inputs=inputs,
        outputs=stored,
        name="two_stage_kg_extraction",
        description="Extract entities, then relations, then store the graph.",
    )

    await two_stage(document)

    # -------------------------------------------------------------------------
    # Inspect the stored graph
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Stored graph")
    print("=" * 60)

    nodes = await knowledge_base.cypher(
        "MATCH (n:Country|City|Landmark) RETURN n.name AS name"
    )
    print("\nNodes:")
    for row in nodes:
        print(f"  - {row['name']}")

    edges = await knowledge_base.cypher(
        "MATCH (a)-[r]->(b) RETURN a.name AS subj, label(r) AS rel, b.name AS obj"
    )
    print("\nEdges:")
    for row in edges:
        print(f"  - ({row['subj']}) -[{row['rel']}]-> ({row['obj']})")

    # -------------------------------------------------------------------------
    # Query: the local neighbourhood around "Paris"
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Local graph search around 'Paris'")
    print("=" * 60)

    subgraph = await knowledge_base.local_graph_search(
        "Paris",
        label="City",
        max_hops=2,
        k=1,
    )
    print("\nNeighbourhood entities:")
    for entity in subgraph.entities:
        print(f"  - {entity.label}: {entity.name}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
