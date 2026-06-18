# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import GenericResult
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.modules.retrievers.infer_helpers import concat_infer_fields
from synalinks.src.modules.retrievers.infer_helpers import kb_entity_labels
from synalinks.src.saving import serialization_lib


class EntityHybridFTSSearchInput(DataModel):
    """Input shape for `EntityHybridFTSSearch`.

    The ``keywords`` list is optional — when omitted, the adapter
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
        "synalinks.modules.EntityHybridFTSSearch",
        "synalinks.EntityHybridFTSSearch",
    ]
)
class EntityHybridFTSSearch(Module):
    """RRF fusion of vector similarity + BM25 fulltext over entities.

    Graph-side counterpart of `HybridFTSSearch`. Thin
    deterministic wrapper around
    `KnowledgeBase.entity_hybrid_fts_search`. The vector side's
    text comes from the input's ``similarity_search`` field; the BM25
    side's text comes from ``keywords`` if present, otherwise the
    adapter falls back to ``similarity_search``.

    Single-label only: to retrieve entities of multiple labels,
    compose several `EntityHybridFTSSearch` modules in the
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
            ``title``. **Optional** — when neither ``label`` nor a
            schema to derive it from is given, the language model infers
            the target entity label per call (constrained to the
            knowledge base's actual entity labels).
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

    def __init__(
        self,
        *,
        knowledge_base=None,
        language_model=None,
        schema=None,
        entity_model=None,
        label: Optional[str] = None,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        output_format: str = "json",
        prompt_template: Optional[str] = None,
        examples: Optional[list] = None,
        instructions: Optional[str] = None,
        seed_instructions: Optional[str] = None,
        temperature: float = 0.0,
        use_inputs_schema: bool = False,
        use_outputs_schema: bool = False,
        return_inputs: bool = True,
        return_query: bool = True,
        name: Optional[str] = None,
        description: Optional[str] = None,
        trainable: bool = True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.knowledge_base = _get_kb(knowledge_base)
        self.language_model = _get_lm(language_model)

        if schema is None and entity_model is not None:
            schema = entity_model.get_schema()
        self.schema = schema
        self.entity_model = entity_model
        # `label` is optional: when it (and a schema to infer it from) is absent,
        # the LM picks the entity label per call (see query_generator).
        if label is None and schema is not None:
            label = schema.get("title") or None
        self.label = label

        if output_format not in ("json", "csv"):
            raise ValueError(
                f"`output_format` must be 'json' or 'csv', got {output_format!r}"
            )
        self.output_format = output_format

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}")
        self.k = k
        self.k_rank = k_rank
        self.similarity_threshold = similarity_threshold
        self.fulltext_threshold = fulltext_threshold
        self.ef_search = ef_search
        self.conjunctive = conjunctive
        self.bm25_b = bm25_b

        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query

        if self.label is None:
            gen_target = {
                "schema": concat_infer_fields(
                    EntityHybridFTSSearchInput.get_schema(),
                    [
                        (
                            "entity_label",
                            "The entity label to search, chosen to best answer "
                            "the inputs.",
                            kb_entity_labels(self.knowledge_base),
                        )
                    ],
                )
            }
        else:
            gen_target = {"data_model": EntityHybridFTSSearchInput}

        self.query_generator = Generator(
            **gen_target,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="entity_hybrid_fts_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
        query_json = query.get_json()
        queries = query_json.get("similarity_search", [])
        keywords = query_json.get("keywords")
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("entity_label")
        if not queries or not label:
            return None

        rows = await self.knowledge_base.entity_hybrid_fts_search(
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
        results = JsonDataModel(
            json={"result": rows},
            schema=GenericResult.get_schema(),
            name=self.name,
        )
        if self.return_query:
            results = await ops.logical_and(
                query,
                results,
                name="results_with_query_" + self.name,
            )
        if self.return_inputs:
            results = await ops.logical_and(
                inputs,
                results,
                name="results_with_inputs_" + self.name,
            )
        return results

    async def compute_output_spec(self, inputs, training=False):
        query = await self.query_generator(inputs, training=training)
        results = SymbolicDataModel(
            schema=GenericResult.get_schema(),
            name=self.name,
        )
        if self.return_query:
            results = await ops.logical_and(
                query,
                results,
                name="results_with_query_" + self.name,
            )
        if self.return_inputs:
            results = await ops.logical_and(
                inputs,
                results,
                name="results_with_inputs_" + self.name,
            )
        return results

    def get_config(self):
        config = {
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
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs": self.return_inputs,
            "return_query": self.return_query,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        knowledge_base_config = {
            "knowledge_base": serialization_lib.serialize_synalinks_object(
                self.knowledge_base,
            )
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        em = self.entity_model
        if em is not None and not is_symbolic_data_model(em):
            em = em.to_symbolic_data_model(name="entity_model_" + self.name)
        entity_model_config = {
            "entity_model": (
                serialization_lib.serialize_synalinks_object(em)
                if em is not None
                else None
            ),
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **entity_model_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        entity_model_serialized = config.pop("entity_model", None)
        entity_model = (
            serialization_lib.deserialize_synalinks_object(entity_model_serialized)
            if entity_model_serialized is not None
            else None
        )
        return cls(
            knowledge_base=knowledge_base,
            language_model=language_model,
            entity_model=entity_model,
            **config,
        )
