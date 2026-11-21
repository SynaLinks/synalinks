# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_relation_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class RelationHybridFTSSearchInput(DataModel):
    """Input shape for `RelationHybridFTSSearch`.

    The ``keywords`` list is optional: when omitted, the adapter
    re-uses the vector side's text for BM25 scoring as well.
    """

    similarity_search: List[str] = Field(
        description="Natural-language queries for the vector branch",
    )
    keywords: Optional[List[str]] = Field(
        description="Optional keyword queries for the BM25 branch",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.modules.RelationHybridFTSSearch",
        "synalinks.RelationHybridFTSSearch",
    ]
)
class RelationHybridFTSSearch(Retriever):
    """RRF of vector similarity + BM25 fulltext over relations.

    Graph-side counterpart of `HybridFTSSearch`, but for
    edges. LM-driven wrapper around
    `KnowledgeBase.relation_hybrid_fts_search`. Either-endpoint
    union: per matched edge, the final ``rrf_score`` is the sum of
    the subject-side and object-side hybrid scores, equivalent to a
    4-source RRF. Falls back to fulltext-only when no embedding model
    is configured.

    Single-label only: to retrieve relations of multiple labels,
    compose several `RelationHybridFTSSearch` modules in the
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
        k_rank (int): RRF smoothing constant. Defaults to 60.
        similarity_threshold (float): Optional vector-distance
            threshold for the vector branch.
        fulltext_threshold (float): Optional BM25 threshold for the
            fulltext branch.
        ef_search (int): HNSW search-time candidate-list depth.
        conjunctive (bool): When ``True``, BM25 requires every term to
            match (AND-mode). Defaults to ``False``.
        bm25_b (float): Optional override for BM25's ``b`` parameter.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = RelationHybridFTSSearchInput
    data_model_args = ("relation_model",)

    def __init__(
        self,
        *,
        schema=None,
        relation_model=None,
        label: Optional[str] = None,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.label = resolve_target(schema, relation_model, label)
        self.relation_model = relation_model
        self.k_rank = k_rank
        self.similarity_threshold = similarity_threshold
        self.fulltext_threshold = fulltext_threshold
        self.ef_search = ef_search
        self.conjunctive = conjunctive
        self.bm25_b = bm25_b

        self.build_query_generator(
            [
                (
                    "relation_label",
                    "The relation label to search, chosen to best " "answer the inputs.",
                    kb_relation_labels(self.knowledge_base),
                )
            ]
            if self.label is None
            else []
        )

    async def search(self, query):
        query_json = query.get_json()
        queries = query_json.get("similarity_search", [])
        keywords = query_json.get("keywords")
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("relation_label")
        if not queries or not label:
            return None

        rows = await self.knowledge_base.relation_hybrid_fts_search(
            text_or_texts=queries,
            keywords=keywords,
            label=label,
            k=self.k,
            k_rank=self.k_rank,
            similarity_threshold=self.similarity_threshold,
            fulltext_threshold=self.fulltext_threshold,
            ef_search=self.ef_search,
            conjunctive=self.conjunctive,
            bm25_b=self.bm25_b,
            output_format=self.output_format,
        )
        return {"result": rows}

    def get_config(self):
        return {
            "schema": self.schema,
            "label": self.label,
            "k": self.k,
            "k_rank": self.k_rank,
            "similarity_threshold": self.similarity_threshold,
            "fulltext_threshold": self.fulltext_threshold,
            "ef_search": self.ef_search,
            "conjunctive": self.conjunctive,
            "bm25_b": self.bm25_b,
            "output_format": self.output_format,
            **super().get_config(),
        }
