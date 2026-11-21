# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_entity_labels
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class EntitySimilaritySearchInput(DataModel):
    """Input shape for `EntitySimilaritySearch`."""

    similarity_search: List[str] = Field(
        description="Natural-language queries for vector similarity",
    )


@synalinks_export(
    [
        "synalinks.modules.EntitySimilaritySearch",
        "synalinks.EntitySimilaritySearch",
    ]
)
class EntitySimilaritySearch(Retriever):
    """Vector similarity search over entities of a single label.

    Graph-side counterpart of `SimilaritySearch`. Thin
    deterministic wrapper around
    `KnowledgeBase.entity_similarity_search`. The query text
    comes straight from the input data model's ``similarity_search``
    field; an embedded Generator builds the query from the inputs.

    Single-label only: to retrieve entities of multiple labels,
    compose several `EntitySimilaritySearch` modules in the
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
        threshold (float): Optional maximum vector-distance threshold.
            Lower distance = better match.
        ef_search (int): HNSW search-time candidate-list depth.
        output_format (str): ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = EntitySimilaritySearchInput
    data_model_args = ("entity_model",)

    def __init__(
        self,
        *,
        schema=None,
        entity_model=None,
        label: Optional[str] = None,
        threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.label = resolve_target(schema, entity_model, label)
        self.entity_model = entity_model
        self.threshold = threshold
        self.ef_search = ef_search

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
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("entity_label")
        if not queries or not label:
            return None

        rows = await self.knowledge_base.entity_similarity_search(
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
