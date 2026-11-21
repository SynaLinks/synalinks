# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_entity_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class EntityHybridRegexSearchInput(DataModel):
    """Input shape for `EntityHybridRegexSearch`.

    The ``regex_patterns`` list is optional: when omitted, the
    adapter falls back to plain vector similarity over
    ``similarity_search``.
    """

    similarity_search: List[str] = Field(
        description="Natural-language queries for the vector branch",
    )
    regex_patterns: Optional[List[str]] = Field(
        description="Regex patterns (RE2 syntax) for the regex branch",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.modules.EntityHybridRegexSearch",
        "synalinks.EntityHybridRegexSearch",
    ]
)
class EntityHybridRegexSearch(Retriever):
    """RRF fusion of vector similarity + regex matching over entities.

    Graph-side counterpart of `HybridRegexSearch`. Thin
    deterministic wrapper around
    `KnowledgeBase.entity_hybrid_regex_search`. The vector
    side's text comes from the input's ``similarity_search`` field;
    the regex side's patterns come from ``regex_patterns``. When
    ``regex_patterns`` is empty the adapter falls back to plain
    vector similarity.

    Regex uses RE2 (DuckDB's engine), so patterns are linear-time
    and not vulnerable to catastrophic backtracking.

    Single-label only: to retrieve entities of multiple labels,
    compose several `EntityHybridRegexSearch` modules in the
    program DAG and merge their outputs explicitly.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        schema (dict): JSON schema of the entity. Used to infer
            ``label`` from its ``title`` when not given explicitly.
            Mutually inferrable with ``entity_model``.
        entity_model (Entity | SymbolicDataModel): Entity model
            providing ``schema`` via ``.get_schema()`` when ``schema``
            is not given.
        label (str): Target entity label. Defaults to the schema's
            ``title``. **Optional**: when neither ``label`` nor a
            schema to derive it from is given, the language model infers
            the target entity label per call (constrained to the
            knowledge base's actual entity labels).
        k (int): Maximum number of results. Defaults to 10.
        k_rank (int): RRF smoothing constant. Defaults to 60.
        similarity_threshold (float): Optional vector-distance
            threshold for the vector branch.
        fields (list): Field names to match against in the regex
            branch. Defaults to every string field on the schema.
        case_sensitive (bool): When ``False``, regex matches are
            case-insensitive. Defaults to ``True``.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = EntityHybridRegexSearchInput
    data_model_args = ("entity_model",)

    def __init__(
        self,
        *,
        schema=None,
        entity_model=None,
        label: Optional[str] = None,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.label = resolve_target(schema, entity_model, label)
        self.entity_model = entity_model
        self.k_rank = k_rank
        self.similarity_threshold = similarity_threshold
        self.fields = fields
        self.case_sensitive = case_sensitive

        self.build_query_generator(
            [
                (
                    "entity_label",
                    "The entity label to search, chosen to best answer " "the inputs.",
                    kb_entity_labels(self.knowledge_base),
                )
            ]
            if self.label is None
            else []
        )

    async def search(self, query):
        query_json = query.get_json()
        queries = query_json.get("similarity_search", [])
        patterns = query_json.get("regex_patterns")
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("entity_label")
        # Need at least one signal (vector or regex) plus a label to look up.
        if (not queries and not patterns) or not label:
            return None

        rows = await self.knowledge_base.entity_hybrid_regex_search(
            text_or_texts=queries,
            pattern_or_patterns=patterns or None,
            label=label,
            fields=self.fields,
            case_sensitive=self.case_sensitive,
            k=self.k,
            k_rank=self.k_rank,
            similarity_threshold=self.similarity_threshold,
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
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
            "output_format": self.output_format,
            **super().get_config(),
        }
