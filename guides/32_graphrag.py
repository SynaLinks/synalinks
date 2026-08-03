"""
# GraphRAG

[Guide 27](https://synalinks.github.io/synalinks/guides/Knowledge%20Graph%20Extraction/) ended with
`local_graph_search`: vector-match a seed entity, walk its
neighbourhood, hand the subgraph to a generator. That answers
**entity-centric** questions (*"who does Alice collaborate with?"*)
because everything relevant sits within a few hops of one node.

It cannot answer **corpus-centric** questions: *"what are the main
research areas across the whole lab?"*. No single neighbourhood
contains that answer; it is spread over the *entire* graph. Retrieving
more hops doesn't help: you'd end up stuffing the whole graph into the
prompt.

**GraphRAG** ([Edge et al., 2024](https://arxiv.org/abs/2404.16130))
solves this with a divide-and-conquer scheme borrowed from MapReduce:

1. **Index time**: cluster the graph into **communities** (groups of
   densely-connected entities, found with the Louvain algorithm) and
   score every node's importance with PageRank. Done once, persisted
   on the nodes.
2. **Query time (map)**: answer the question against *each* community
   subgraph independently, in parallel, scoring how relevant each
   community is.
3. **Query time (reduce)**: combine the partial answers, best first,
   into one final answer.

```mermaid
graph LR
    subgraph "Index time (once)"
        A[("Graph store")] --> B["build_communities()<br/>Louvain + PageRank"]
    end
    subgraph "Query time"
        Q["Question"] --> G["GlobalGraphSearch<br/>top-k communities"]
        G --> M1["Map: answer vs community 1"]
        G --> M2["Map: answer vs community 2"]
        G --> M3["Map: answer vs community k"]
        M1 --> R["Reduce: combine<br/>partial answers"]
        M2 --> R
        M3 --> R
    end
```

Synalinks packages this as three modules, from lowest to highest level:

- **`LocalGraphSearch`**: the entity-centric side, as a trainable
  module: an LM turns your inputs into seed queries, the store expands
  their neighbourhood, the subgraph flows out.
- **`GlobalGraphSearch`**: retrieval only: the top-`k` communities as
  subgraphs, no LM involved.
- **`GlobalGraphMapReduce`**: the full query-time pipeline:
  `GlobalGraphSearch` + a parallel *map* generator + a *reduce*
  generator.

This guide assumes a populated graph store; extraction is
[Guide 27](https://synalinks.github.io/synalinks/guides/Knowledge%20Graph%20Extraction/)'s job. To keep this guide
deterministic we hand-author a small collaboration graph instead of
extracting one.

## The Graph: a Research Lab

Six researchers, two tightly-knit teams, one cross-team collaboration:

- **Vision team**: Alice, Bob, Carol, collaborating with each other.
- **Language team**: Dan, Eve, Frank, collaborating with each other.
- **One bridge**: Alice also collaborates with Dan.

Louvain will recover the two teams as communities. Note one backend
constraint worth knowing up front: **Louvain runs on a single node
label at a time** (here everything is a `Researcher`). Graphs with
several entity types can either pass `node_labels=[...]` to cluster
one type, or use `algorithm="weakly_connected_components"` which
supports any number of labels.

Because a `Relation` carries its endpoint entities in full, seeding
the store takes only the edge list (the relations-only strategy from
Guide 27): `UpdateKnowledge` unpacks each edge into its two nodes plus
the link.

```python
class Researcher(synalinks.Entity):
    label: Literal["Researcher"]
    name: str = synalinks.Field(description="The researcher's name.")
    focus: str = synalinks.Field(description="Their research focus.")

class CollaboratesWith(synalinks.Relation):
    label: Literal["CollaboratesWith"]
    subj: Researcher
    obj: Researcher
```

## Index Time: `build_communities`

After the graph is stored (and embedded: the local side needs the
vectors), one call materializes the GraphRAG index:

```python
stamped = await knowledge_base.build_communities(algorithm="louvain")
```

This clusters the graph with Louvain, scores every node with PageRank,
and **persists** both as reserved `community` / `rank` properties on
the nodes. It is idempotent: re-run it whenever the graph has changed
enough to matter, exactly like rebuilding any index. Everything at
query time just *reads* these properties; no clustering happens per
query.

## Local Search: `LocalGraphSearch`

The entity-centric retriever, now as a module you can drop into a
program (rather than the raw `kb.local_graph_search()` call from
Guide 27). Its embedded generator reads your inputs and emits the seed
queries; the store vector-matches `k` seeds of `label`, expands
`max_hops` around them, and returns the deduped union as one subgraph:

```python
inputs = synalinks.Input(data_model=Query)
subgraph = await synalinks.LocalGraphSearch(
    knowledge_base=knowledge_base,
    language_model=language_model,
    label="Researcher",
    max_hops=2,
    k=1,
)(inputs)
```

`label` is optional: when omitted, the LM *infers* it per call,
constrained to an enum of the labels actually present in the store:
it cannot pick a nonexistent one. The generated queries (and the
inputs) are concatenated onto the output by default
(`return_query` / `return_inputs`), so a downstream generator sees
question + queries + subgraph together.

## Global Search: `GlobalGraphSearch`

The retrieval-only half of the global side. It reads the communities
`build_communities` stamped and returns the `k` most important ones,
each rebuilt as a full subgraph (member entities plus the relations
*internal* to the community; edges that straddle two communities
belong to neither), ordered by aggregate PageRank:

```python
communities = await synalinks.GlobalGraphSearch(
    knowledge_base=knowledge_base,
    k=3,
)(inputs)
```

Two things distinguish it from every other retriever: it runs **no
language model**, and it **ignores the content of its inputs**:
global search is whole-graph by definition, so there is no query to
build. Relevance to a specific question is decided downstream, which
is exactly the map step's job.

## Map-Reduce: `GlobalGraphMapReduce`

The full GraphRAG query-time pipeline in one module:

```python
answer = await synalinks.GlobalGraphMapReduce(
    knowledge_base=knowledge_base,
    language_model=language_model,
    k=3,
)(inputs)
```

Under the hood it:

1. fetches the top-`k` communities via an embedded `GlobalGraphSearch`;
2. **maps**: one generator call *per community, in parallel*, each
   answering the inputs against that community's subgraph alone and
   scoring the community's relevance 0-100;
3. filters (`score_threshold`, default 0 = keep everything; raise it
   to drop irrelevant communities like the original GraphRAG) and
   sorts the partial answers best-first;
4. **reduces**: one final generator call that combines them.

By default the reduce step returns a plain `ChatMessage`, the same
convention as a bare `Generator`. Pass `schema=` or `data_model=` to
get a structured final answer instead, and
`return_community_answers=True` to keep the scored partial answers in
the output (useful for debugging, or as a reward signal). The map and
reduce prompts are regular trainable generators: `map_instructions` /
`reduce_instructions` set their starting instructions, and the whole
module optimizes like any other (see the
[Training guide](https://synalinks.github.io/synalinks/guides/Training/)).

## Choosing Local vs Global

| Question shape | Example | Module |
|---|---|---|
| Entity-centric ("around here") | "Who does Alice collaborate with?" | `LocalGraphSearch` |
| Corpus-centric ("across everything") | "What are the lab's main research areas?" | `GlobalGraphMapReduce` |
| Need the raw communities | Custom summarization pipeline | `GlobalGraphSearch` |
| Exact traversal | "Capitals and their countries" | `kb.cypher(...)` |

The two sides compose: nothing stops a program from running both
retrievers and letting a final generator weigh the neighbourhood
against the big picture.

## Key Takeaways

- **Local search answers entity-centric questions; global search
  answers corpus-centric ones.** Neighbourhood expansion cannot
  summarize a whole graph; that is what communities are for.
- **`build_communities` is the index-time half**: Louvain communities
  + PageRank ranks, persisted on the nodes, rebuilt on demand.
  Louvain needs a single node label; WCC handles many.
- **`LocalGraphSearch`** wraps seed-and-expand as a trainable module:
  the LM writes the seed queries and can even infer the entity label,
  constrained to labels that exist.
- **`GlobalGraphSearch`** returns the top-`k` communities as
  subgraphs. No LM, input content ignored; relevance is the map
  step's job.
- **`GlobalGraphMapReduce`** = search + parallel scored map + best-first
  reduce. Chat-message output by default, structured via
  `schema`/`data_model`, filterable via `score_threshold`.

## API References

- [KnowledgeBase](https://synalinks.github.io/synalinks/Synalinks%20API/Knowledge%20Base%20API/Knowledge%20Base/)
- [LocalGraphSearch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Retrievers%20Modules/LocalGraphSearch%20module/)
- [GlobalGraphSearch](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Retrievers%20Modules/GlobalGraphSearch%20module/)
- [GlobalGraphMapReduce](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Retrievers%20Modules/GlobalGraphMapReduce%20module/)
- [EmbedKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/EmbedKnowledge%20module/)
- [UpdateKnowledge](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Knowledge%20Modules/UpdateKnowledge%20module/)
"""

# --8<-- [start:source]
import asyncio
from typing import List
from typing import Literal

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Graph Schema + Input
# =============================================================================


class Researcher(synalinks.Entity):
    label: Literal["Researcher"]
    name: str = synalinks.Field(description="The researcher's name, e.g. 'Alice'.")
    focus: str = synalinks.Field(description="Their research focus.")


class CollaboratesWith(synalinks.Relation):
    label: Literal["CollaboratesWith"]
    subj: Researcher
    obj: Researcher


# Relations-only seeding (Guide 27, strategy 4): each edge carries its two
# endpoint entities in full, so the edge list alone populates the graph.
class LabCollaborations(synalinks.Relations):
    relations: List[CollaboratesWith] = synalinks.Field(
        description="Every collaboration in the lab.",
    )


class Query(synalinks.DataModel):
    question: str = synalinks.Field(description="The user question")


# =============================================================================
# The Lab Graph: two teams + one bridge
# =============================================================================

ALICE = Researcher(
    label="Researcher", name="Alice", focus="diffusion models for image generation"
)
BOB = Researcher(label="Researcher", name="Bob", focus="real-time image segmentation")
CAROL = Researcher(label="Researcher", name="Carol", focus="video understanding")
DAN = Researcher(label="Researcher", name="Dan", focus="low-resource machine translation")
EVE = Researcher(label="Researcher", name="Eve", focus="speech recognition")
FRANK = Researcher(
    label="Researcher", name="Frank", focus="task-oriented dialogue systems"
)

COLLABORATIONS = LabCollaborations(
    relations=[
        # Vision team: a triangle.
        CollaboratesWith(label="CollaboratesWith", subj=ALICE, obj=BOB),
        CollaboratesWith(label="CollaboratesWith", subj=BOB, obj=CAROL),
        CollaboratesWith(label="CollaboratesWith", subj=CAROL, obj=ALICE),
        # Language team: a triangle.
        CollaboratesWith(label="CollaboratesWith", subj=DAN, obj=EVE),
        CollaboratesWith(label="CollaboratesWith", subj=EVE, obj=FRANK),
        CollaboratesWith(label="CollaboratesWith", subj=FRANK, obj=DAN),
        # The bridge between the two teams.
        CollaboratesWith(label="CollaboratesWith", subj=ALICE, obj=DAN),
    ],
)


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()
    synalinks.enable_logging(log_level="info")

    language_model = synalinks.LanguageModel(model="ollama/mistral:latest")
    embedding_model = synalinks.EmbeddingModel(model="ollama/mxbai-embed-large")

    knowledge_base = synalinks.KnowledgeBase(
        graph_uri="ladybug://:memory:",
        entity_models=[Researcher],
        relation_models=[CollaboratesWith],
        embedding_model=embedding_model,
        metric="cosine",
        wipe_on_start=True,
    )

    # -------------------------------------------------------------------------
    # Index time 1/2: embed + store the graph
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Index time: store the graph")
    print("=" * 60)

    inputs = synalinks.Input(data_model=LabCollaborations)
    embedded = await synalinks.EmbedKnowledge(
        embedding_model=embedding_model,
        in_mask=["name"],
    )(inputs)
    stored = await synalinks.UpdateKnowledge(
        knowledge_base=knowledge_base,
    )(embedded)

    loader = synalinks.Program(
        inputs=inputs,
        outputs=stored,
        name="lab_graph_loader",
        description="Embed and store the lab collaboration graph.",
    )
    await loader(COLLABORATIONS)

    # -------------------------------------------------------------------------
    # Index time 2/2: build the communities (the GraphRAG index)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Index time: build_communities")
    print("=" * 60)

    stamped = await knowledge_base.build_communities(algorithm="louvain")
    print(f"\nNodes stamped with community + rank: {stamped}")

    # -------------------------------------------------------------------------
    # Local search: entity-centric questions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LocalGraphSearch: around Alice")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    subgraph = await synalinks.LocalGraphSearch(
        knowledge_base=knowledge_base,
        language_model=language_model,
        label="Researcher",
        max_hops=2,
        k=1,
    )(inputs)

    local_search = synalinks.Program(
        inputs=inputs,
        outputs=subgraph,
        name="local_graph_search",
        description="Retrieve the neighbourhood around the entities in question.",
    )

    result = await local_search(
        Query(question="Who does Alice collaborate with, and what do they work on?")
    )
    print("\nNeighbourhood entities:")
    for entity in result.get("entities"):
        print(f"  - {entity['name']}: {entity['focus']}")
    print("\nNeighbourhood relations:")
    for relation in result.get("relations"):
        print(f"  - {relation['subj']['name']} -> {relation['obj']['name']}")

    # -------------------------------------------------------------------------
    # Global search (retrieval only): the top communities
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GlobalGraphSearch: the communities")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    communities = await synalinks.GlobalGraphSearch(
        knowledge_base=knowledge_base,
        k=3,
        return_inputs=False,
    )(inputs)

    global_search = synalinks.Program(
        inputs=inputs,
        outputs=communities,
        name="global_graph_search",
        description="Retrieve the graph's most important communities.",
    )

    result = await global_search(Query(question="ignored: global search is whole-graph"))
    for i, community in enumerate(result.get("knowledge_graphs")):
        members = ", ".join(entity["name"] for entity in community["entities"])
        print(f"\nCommunity #{i} ({len(community['relations'])} internal edges):")
        print(f"  {members}")

    # -------------------------------------------------------------------------
    # Map-reduce: corpus-centric questions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GlobalGraphMapReduce: the whole-lab question")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)
    answer = await synalinks.GlobalGraphMapReduce(
        knowledge_base=knowledge_base,
        language_model=language_model,
        k=3,
        return_community_answers=True,
    )(inputs)

    global_qa = synalinks.Program(
        inputs=inputs,
        outputs=answer,
        name="global_graph_qa",
        description="Answer corpus-level questions by community map-reduce.",
    )
    global_qa.summary()

    result = await global_qa(
        Query(question="What are the main research areas in the lab?")
    )
    print("\nScored community answers (map step):")
    for community_answer in result.get("community_answers"):
        score = community_answer["relevance_score"]
        print(f"  [{score:5.1f}] {community_answer['partial_answer']}")
    print("\nFinal answer (reduce step):")
    print(result.get("content"))

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
