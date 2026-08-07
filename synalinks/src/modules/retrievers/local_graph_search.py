# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import KnowledgeGraph
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.modules.retrievers.infer_helpers import concat_infer_fields
from synalinks.src.modules.retrievers.infer_helpers import kb_entity_labels
from synalinks.src.saving import serialization_lib


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
class LocalGraphSearch(Module):
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

    def __init__(
        self,
        *,
        knowledge_base=None,
        language_model=None,
        schema=None,
        entity_model=None,
        label: Optional[str] = None,
        max_hops: int = 2,
        k: int = 10,
        threshold: Optional[float] = None,
        rel_label: Optional[str] = None,
        ef_search: Optional[int] = None,
        prompt_template: Optional[str] = None,
        examples: Optional[list] = None,
        instructions: Optional[str] = None,
        seed_instructions: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
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
        # the LM picks the seed entity label per call (see query_generator).
        if label is None and schema is not None:
            label = schema.get("title") or None
        self.label = label

        if max_hops < 1:
            raise ValueError(f"`max_hops` must be >= 1, got {max_hops!r}")
        self.max_hops = max_hops

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}")
        self.k = k
        self.threshold = threshold
        self.rel_label = rel_label
        self.ef_search = ef_search

        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query

        if self.label is None:
            gen_target = {
                "schema": concat_infer_fields(
                    LocalGraphSearchInput.get_schema(),
                    [
                        (
                            "entity_label",
                            "The seed entity label to search around, chosen to "
                            "best answer the inputs.",
                            kb_entity_labels(self.knowledge_base),
                        )
                    ],
                )
            }
        else:
            gen_target = {"data_model": LocalGraphSearchInput}

        self.query_generator = Generator(
            **gen_target,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="local_graph_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
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
        results = JsonDataModel(
            json=graph.get_json(),
            schema=KnowledgeGraph.get_schema(),
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
            schema=KnowledgeGraph.get_schema(),
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
            "max_hops": self.max_hops,
            "k": self.k,
            "threshold": self.threshold,
            "rel_label": self.rel_label,
            "ef_search": self.ef_search,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
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
