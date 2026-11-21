# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_relation_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class RelationFullTextSearchInput(DataModel):
    """Input shape for `RelationFullTextSearch`."""

    fulltext_search: List[str] = Field(
        description="Keyword queries for BM25 full-text search",
    )


@synalinks_export(
    [
        "synalinks.modules.RelationFullTextSearch",
        "synalinks.RelationFullTextSearch",
    ]
)
class RelationFullTextSearch(Retriever):
    """BM25 full-text search over relations of a single label.

    Graph-side counterpart of `FullTextSearch`, but for edges.
    LM-driven wrapper around
    `KnowledgeBase.relation_fulltext_search`. Per matched edge,
    the final ``score`` is the sum of the subject-side and object-side
    BM25 scores (either-endpoint union).

    Single-label only: to retrieve relations of multiple labels,
    compose several `RelationFullTextSearch` modules in the
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
        threshold (float): Optional minimum BM25 threshold applied
            per endpoint.
        conjunctive (bool): When ``True``, BM25 requires every term to
            match (AND-mode). Defaults to ``False``.
        bm25_b (float): Optional override for BM25's ``b`` parameter.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = RelationFullTextSearchInput
    data_model_args = ("relation_model",)

    def __init__(
        self,
        *,
        schema=None,
        relation_model=None,
        label: Optional[str] = None,
        threshold: Optional[float] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.label = resolve_target(schema, relation_model, label)
        self.relation_model = relation_model
        self.threshold = threshold
        self.conjunctive = conjunctive
        self.bm25_b = bm25_b

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
        queries = query_json.get("fulltext_search", [])
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("relation_label")
        if not queries or not label:
            return None

        rows = await self.knowledge_base.relation_fulltext_search(
            queries,
            label=label,
            k=self.k,
            threshold=self.threshold,
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
            "threshold": self.threshold,
            "conjunctive": self.conjunctive,
            "bm25_b": self.bm25_b,
            "output_format": self.output_format,
            **super().get_config(),
        }
