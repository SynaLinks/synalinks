# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_relation_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class RelationSimilaritySearchInput(DataModel):
    """Input shape for `RelationSimilaritySearch`."""

    similarity_search: List[str] = Field(
        description="Natural-language queries for vector similarity",
    )


@synalinks_export(
    [
        "synalinks.modules.RelationSimilaritySearch",
        "synalinks.RelationSimilaritySearch",
    ]
)
class RelationSimilaritySearch(Retriever):
    """Vector similarity search over relations of a single label.

    Graph-side counterpart of `SimilaritySearch`, but for
    edges. LM-driven wrapper around
    `KnowledgeBase.relation_similarity_search`. The query text
    matches against BOTH endpoints (subject and object); the adapter
    returns one row per matched edge with its best (lowest) distance
    and a ``matched_on`` tag (``"subj"``, ``"obj"``, or ``"both"``).

    Single-label only: to retrieve relations of multiple labels,
    compose several `RelationSimilaritySearch` modules in the
    program DAG and merge their outputs explicitly.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        schema (dict): JSON schema of the relation. Used to infer
            ``label`` from its ``title`` when not given explicitly.
            Mutually inferrable with ``relation_model``.
        relation_model (Relation | SymbolicDataModel): Relation model
            providing ``schema`` via ``.get_schema()`` when ``schema``
            is not given.
        label (str): Target relation label. Defaults to the schema's
            ``title``. **Optional**: when neither ``label`` nor a
            schema to derive it from is given, the language model infers
            the target relation label per call (constrained to the
            knowledge base's actual relation labels).
        k (int): Maximum number of results. Defaults to 10.
        threshold (float): Optional maximum vector-distance threshold
            applied per endpoint.
        ef_search (int): HNSW search-time candidate-list depth,
            applied to both endpoint vector searches.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = RelationSimilaritySearchInput
    data_model_args = ("relation_model",)

    def __init__(
        self,
        *,
        schema=None,
        relation_model=None,
        label: Optional[str] = None,
        threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.label = resolve_target(schema, relation_model, label)
        self.relation_model = relation_model
        self.threshold = threshold
        self.ef_search = ef_search

        self.build_query_generator(
            [
                (
                    "relation_label",
                    "The relation label to search, chosen to best answer the inputs.",
                    kb_relation_labels(self.knowledge_base),
                )
            ]
            if self.label is None
            else []
        )

    async def search(self, query):
        query_json = query.get_json()
        queries = query_json.get("similarity_search", [])
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("relation_label")
        if not queries or not label:
            return None

        rows = await self.knowledge_base.relation_similarity_search(
            queries,
            label=label,
            k=self.k,
            threshold=self.threshold,
            ef_search=self.ef_search,
            output_format=self.output_format,
        )
        return {"result": rows}

    def get_config(self):
        return {
            "schema": self.schema,
            "label": self.label,
            "k": self.k,
            "threshold": self.threshold,
            "ef_search": self.ef_search,
            "output_format": self.output_format,
            **super().get_config(),
        }
