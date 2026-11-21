# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_entity_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class PathSimilaritySearchInput(DataModel):
    """Input shape for `PathSimilaritySearch`."""

    subj_similarity_search: List[str] = Field(
        description="Natural-language queries for the subject endpoint",
    )
    obj_similarity_search: List[str] = Field(
        description="Natural-language queries for the object endpoint",
    )


@synalinks_export(
    [
        "synalinks.modules.PathSimilaritySearch",
        "synalinks.PathSimilaritySearch",
    ]
)
class PathSimilaritySearch(Retriever):
    """Vector variable-length path search where BOTH endpoints match.

    LM-driven wrapper around
    `KnowledgeBase.path_similarity_search`. Returns paths of
    ``min_hops..max_hops`` edges whose subject endpoint is
    vector-close to ``subj_similarity_search`` AND whose object
    endpoint is vector-close to ``obj_similarity_search``.

    ``rel_label`` is an optional constraint applied to every hop;
    when omitted, any edge type is allowed.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        subj_schema (dict): JSON schema of the subject entity. Used
            to infer ``subj_label`` from its ``title`` when not given
            explicitly. Mutually inferrable with ``subj_entity_model``.
        subj_entity_model (Entity | SymbolicDataModel): Subject entity
            model (provides ``subj_schema`` / ``subj_label``).
        subj_label (str): Subject entity label. **Optional**: when
            neither it nor a schema to derive it from is given, the
            language model infers it per call (constrained to the
            knowledge base's entity labels).
        obj_schema (dict): JSON schema of the object entity.
        obj_entity_model (Entity | SymbolicDataModel): Object entity
            model (provides ``obj_schema`` / ``obj_label``).
        obj_label (str): Object entity label. **Optional**: inferred
            per call like ``subj_label`` when not given.
        rel_label (str): Optional rel-label constraint applied to
            every hop in the path. ``None`` (default) accepts any
            edge type.
        min_hops (int): Minimum hop count, inclusive. Defaults to 1.
        max_hops (int): Maximum hop count, inclusive. Defaults to 3.
        k (int): Maximum number of results. Defaults to 10.
        subj_threshold (float): Optional subject-side distance ceiling.
        obj_threshold (float): Optional object-side distance ceiling.
        ef_search (int): HNSW search-time candidate-list depth.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = PathSimilaritySearchInput
    data_model_args = (
        "subj_entity_model",
        "obj_entity_model",
    )

    def __init__(
        self,
        *,
        subj_schema=None,
        subj_entity_model=None,
        subj_label: Optional[str] = None,
        obj_schema=None,
        obj_entity_model=None,
        obj_label: Optional[str] = None,
        rel_label: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        subj_threshold: Optional[float] = None,
        obj_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.subj_schema, self.subj_label = resolve_target(
            subj_schema, subj_entity_model, subj_label
        )
        self.subj_entity_model = subj_entity_model
        self.obj_schema, self.obj_label = resolve_target(
            obj_schema, obj_entity_model, obj_label
        )
        self.obj_entity_model = obj_entity_model
        self.rel_label = rel_label

        if min_hops < 1 or max_hops < min_hops:
            raise ValueError(
                f"Invalid hop range: min_hops={min_hops}, "
                f"max_hops={max_hops}. Require 1 <= min_hops <= max_hops."
            )
        self.min_hops = min_hops
        self.max_hops = max_hops
        self.subj_threshold = subj_threshold
        self.obj_threshold = obj_threshold
        self.ef_search = ef_search

        # Either endpoint label may be unset; when so the LM infers it per call,
        # constrained to the KB's entity labels (concatenated onto the query).
        infer_specs = []
        if self.subj_label is None:
            infer_specs.append(
                (
                    "subj_label",
                    "The subject entity label for the path, chosen to best "
                    "answer the inputs.",
                    kb_entity_labels(self.knowledge_base),
                )
            )
        if self.obj_label is None:
            infer_specs.append(
                (
                    "obj_label",
                    "The object entity label for the path, chosen to best "
                    "answer the inputs.",
                    kb_entity_labels(self.knowledge_base),
                )
            )

        self.build_query_generator(infer_specs)

    async def search(self, query):
        payload = query.get_json()
        subj_queries = payload.get("subj_similarity_search", [])
        obj_queries = payload.get("obj_similarity_search", [])
        # Fixed endpoint labels, or the ones the LM inferred this call.
        subj_label = self.subj_label or payload.get("subj_label")
        obj_label = self.obj_label or payload.get("obj_label")
        if not subj_queries or not obj_queries or not subj_label or not obj_label:
            return None

        rows = await self.knowledge_base.path_similarity_search(
            subj_queries,
            obj_queries,
            subj_label=subj_label,
            obj_label=obj_label,
            label=self.rel_label,
            min_hops=self.min_hops,
            max_hops=self.max_hops,
            k=self.k,
            subj_threshold=self.subj_threshold,
            obj_threshold=self.obj_threshold,
            ef_search=self.ef_search,
            output_format=self.output_format,
        )
        return {"result": rows}

    def get_config(self):
        return {
            "subj_schema": self.subj_schema,
            "subj_label": self.subj_label,
            "obj_schema": self.obj_schema,
            "obj_label": self.obj_label,
            "rel_label": self.rel_label,
            "min_hops": self.min_hops,
            "max_hops": self.max_hops,
            "k": self.k,
            "subj_threshold": self.subj_threshold,
            "obj_threshold": self.obj_threshold,
            "ef_search": self.ef_search,
            "output_format": self.output_format,
            **super().get_config(),
        }
