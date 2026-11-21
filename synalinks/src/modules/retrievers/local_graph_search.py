# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import KnowledgeGraph
from synalinks.src.modules.retrievers.infer_helpers import kb_entity_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class LocalGraphSearchInput(DataModel):
    """Input shape for `LocalGraphSearch`."""

    local_graph_search: List[str] = Field(
        description="Natural-language queries naming the entities to search around",
    )


@synalinks_export(
    [
        "synalinks.modules.LocalGraphSearch",
        "synalinks.LocalGraphSearch",
    ]
)
class LocalGraphSearch(Retriever):
    """GraphRAG-style *local* search: seed by vector, expand, return a subgraph.

    Entity-centric graph retrieval. An embedded `Generator` turns the
    inputs into query text; the knowledge base vector-matches ``k`` seed
    entities of ``label``, expands their ``max_hops`` undirected
    neighbourhood, and the deduped union comes back as a
    `KnowledgeGraph`: the local context subgraph you hand a generator
    to answer questions like *"what does the graph say around **these**
    entities"*. Thin wrapper around `KnowledgeBase.local_graph_search`.

    The theme-centric counterpart is `GlobalGraphSearch`, which returns
    whole communities rather than a seed neighbourhood.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        language_model (LanguageModel): The language model that builds
            the query (and infers ``label`` when it is not given).
        schema (dict): JSON schema of the seed entity. Used to infer
            ``label`` from its ``title`` when not given explicitly.
            Mutually inferrable with ``entity_model``.
        entity_model (Entity | SymbolicDataModel): Entity model
            providing ``schema`` via ``.get_schema()`` when ``schema``
            is not given.
        label (str): Seed entity label whose vector index the search
            starts from. Defaults to the schema's ``title``. **Optional**:
            when neither ``label`` nor a schema to derive it from is
            given, the language model infers the label per call
            (constrained to the knowledge base's actual entity labels).
        max_hops (int): Neighbourhood radius in edges (>= 1). Defaults
            to 2.
        k (int): Number of seed entities per query text. Defaults to 10.
        threshold (float): Optional maximum seed vector-distance
            threshold. Lower distance = better match.
        rel_label (str): Optional relation-label constraint applied to
            every hop. ``None`` (default) traverses any edge type.
        ef_search (int): HNSW search-time candidate-list depth for the
            seed lookup.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = LocalGraphSearchInput
    result_model = KnowledgeGraph
    data_model_args = ("entity_model",)

    def __init__(
        self,
        *,
        schema=None,
        entity_model=None,
        label: Optional[str] = None,
        max_hops: int = 2,
        threshold: Optional[float] = None,
        rel_label: Optional[str] = None,
        ef_search: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.label = resolve_target(schema, entity_model, label)
        self.entity_model = entity_model

        if max_hops < 1:
            raise ValueError(f"`max_hops` must be >= 1, got {max_hops!r}")
        self.max_hops = max_hops
        self.threshold = threshold
        self.rel_label = rel_label
        self.ef_search = ef_search

        self.build_query_generator(
            [
                (
                    "entity_label",
                    "The seed entity label to search around, chosen to "
                    "best answer the inputs.",
                    kb_entity_labels(self.knowledge_base),
                )
            ]
            if self.label is None
            else []
        )

    async def search(self, query):
        query_json = query.get_json()
        queries = query_json.get("local_graph_search", [])
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("entity_label")
        if not queries or not label:
            return None

        graph = await self.knowledge_base.local_graph_search(
            queries,
            label=label,
            max_hops=self.max_hops,
            k=self.k,
            threshold=self.threshold,
            rel_label=self.rel_label,
            ef_search=self.ef_search,
        )
        return graph.get_json()

    def get_config(self):
        return {
            "schema": self.schema,
            "label": self.label,
            "max_hops": self.max_hops,
            "k": self.k,
            "threshold": self.threshold,
            "rel_label": self.rel_label,
            "ef_search": self.ef_search,
            **super().get_config(),
        }
