# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_relation_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class RelationRegexSearchInput(DataModel):
    """Input shape for `RelationRegexSearch`."""

    regex_search: str = Field(
        description=(
            "Regex pattern (RE2 syntax) to match against the endpoints' string fields"
        ),
    )


@synalinks_export(
    [
        "synalinks.modules.RelationRegexSearch",
        "synalinks.RelationRegexSearch",
    ]
)
class RelationRegexSearch(Retriever):
    """Regex matching over relations of a single label.

    Graph-side counterpart of `RegexSearch`, but for edges.
    LM-driven wrapper around
    `KnowledgeBase.relation_regex_search`. The pattern is
    matched against the string fields of both endpoints; an edge
    surfaces if either endpoint matched. The row's ``score`` is 2.0
    when both endpoints matched and 1.0 when only one did, with
    ``matched_on`` indicating the side(s).

    Regex uses RE2, so patterns are linear-time and not vulnerable to
    catastrophic backtracking.

    Single-label only: to retrieve relations of multiple labels,
    compose several `RelationRegexSearch` modules in the
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
        fields (list): Field names to match against. Applied to both
            endpoints. Defaults to every string field.
        case_sensitive (bool): When ``False``, regex matches are
            case-insensitive. Defaults to ``True``.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = RelationRegexSearchInput
    data_model_args = ("relation_model",)

    def __init__(
        self,
        *,
        schema=None,
        relation_model=None,
        label: Optional[str] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.label = resolve_target(schema, relation_model, label)
        self.relation_model = relation_model
        self.fields = fields
        self.case_sensitive = case_sensitive

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
        pattern = query_json.get("regex_search")
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("relation_label")
        if not pattern or not label:
            return None

        rows = await self.knowledge_base.relation_regex_search(
            pattern,
            label=label,
            fields=self.fields,
            case_sensitive=self.case_sensitive,
            k=self.k,
            output_format=self.output_format,
        )
        return {"result": rows}

    def get_config(self):
        return {
            "schema": self.schema,
            "label": self.label,
            "k": self.k,
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
            "output_format": self.output_format,
            **super().get_config(),
        }
