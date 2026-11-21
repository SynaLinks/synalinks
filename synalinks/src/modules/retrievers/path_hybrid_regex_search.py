# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_entity_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class PathHybridRegexSearchInput(DataModel):
    """Input shape for `PathHybridRegexSearch`.

    The ``*_regex_patterns`` lists are optional: when both are
    omitted, the module degenerates to plain `PathSimilaritySearch`.
    """

    subj_similarity_search: List[str] = Field(
        description="Natural-language queries for the subject vector branch",
    )
    obj_similarity_search: List[str] = Field(
        description="Natural-language queries for the object vector branch",
    )
    subj_regex_patterns: Optional[List[str]] = Field(
        description="Regex patterns (RE2) for the subject regex branch",
        default=None,
    )
    obj_regex_patterns: Optional[List[str]] = Field(
        description="Regex patterns (RE2) for the object regex branch",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.modules.PathHybridRegexSearch",
        "synalinks.PathHybridRegexSearch",
    ]
)
class PathHybridRegexSearch(Retriever):
    """RRF of vector + regex variable-length path search, AND semantics.

    LM-driven wrapper around
    `KnowledgeBase.path_hybrid_regex_search`. Each side is
    hybrid-searched (vec + regex) independently; the path's combined
    ``rrf_score`` is the sum of the two endpoint hybrid scores,
    the 4-source RRF identity. Falls through to plain
    `PathSimilaritySearch` when no patterns are supplied.

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
            every hop.
        min_hops (int): Minimum hop count, inclusive. Defaults to 1.
        max_hops (int): Maximum hop count, inclusive. Defaults to 3.
        k (int): Maximum number of results. Defaults to 10.
        k_rank (int): RRF smoothing constant. Defaults to 60.
        similarity_threshold (float): Optional vector-distance threshold.
        fields (list): Field names to match against. Applied to both
            endpoints.
        case_sensitive (bool): When ``False``, regex matches are
            case-insensitive. Defaults to ``True``.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = PathHybridRegexSearchInput
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
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
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
        self.k_rank = k_rank
        self.similarity_threshold = similarity_threshold
        self.fields = fields
        self.case_sensitive = case_sensitive

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
        subj_patterns = payload.get("subj_regex_patterns")
        obj_patterns = payload.get("obj_regex_patterns")
        # Fixed endpoint labels, or the ones the LM inferred this call.
        subj_label = self.subj_label or payload.get("subj_label")
        obj_label = self.obj_label or payload.get("obj_label")
        if not subj_queries or not obj_queries or not subj_label or not obj_label:
            return None

        rows = await self.knowledge_base.path_hybrid_regex_search(
            subj_text_or_texts=subj_queries,
            obj_text_or_texts=obj_queries,
            subj_pattern_or_patterns=subj_patterns or None,
            obj_pattern_or_patterns=obj_patterns or None,
            subj_label=subj_label,
            obj_label=obj_label,
            label=self.rel_label,
            min_hops=self.min_hops,
            max_hops=self.max_hops,
            k=self.k,
            k_rank=self.k_rank,
            similarity_threshold=self.similarity_threshold,
            fields=self.fields,
            case_sensitive=self.case_sensitive,
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
            "k_rank": self.k_rank,
            "similarity_threshold": self.similarity_threshold,
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
            "output_format": self.output_format,
            **super().get_config(),
        }
