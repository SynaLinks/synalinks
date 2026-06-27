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
from synalinks.src.modules.retrievers.infer_helpers import kb_relation_labels
from synalinks.src.saving import serialization_lib


class RelationHybridRegexSearchInput(DataModel):
    """Input shape for `RelationHybridRegexSearch`.

    The ``regex_patterns`` list is optional — when omitted, the
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
        "synalinks.modules.RelationHybridRegexSearch",
        "synalinks.RelationHybridRegexSearch",
    ]
)
class RelationHybridRegexSearch(Module):
    """RRF of vector similarity + regex match over relations.

    Graph-side counterpart of `HybridRegexSearch`, but for
    edges. LM-driven wrapper around
    `KnowledgeBase.relation_hybrid_regex_search`. Per matched
    edge, the final ``rrf_score`` is the sum of the subject's and
    the object's hybrid scores — same 4-source-RRF reduction as
    `RelationHybridFTSSearch`. Falls back to plain vector
    similarity when ``regex_patterns`` is empty.

    Regex uses RE2, so patterns are linear-time and not vulnerable
    to catastrophic backtracking.

    Single-label only: to retrieve relations of multiple labels,
    compose several `RelationHybridRegexSearch` modules in the
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
            ``title``. **Optional** — when neither ``label`` nor a
            schema to derive it from is given, the language model infers
            the target relation label per call (constrained to the
            knowledge base's actual relation labels).
        k (int): Maximum number of results. Defaults to 10.
        k_rank (int): RRF smoothing constant. Defaults to 60.
        similarity_threshold (float): Optional vector-distance
            threshold for the vector branch.
        fields (list): Field names to match against in the regex
            branch. Applied to both endpoints.
        case_sensitive (bool): When ``False``, regex matches are
            case-insensitive. Defaults to ``True``.
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
        relation_model=None,
        label: Optional[str] = None,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        output_format: str = "json",
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

        if schema is None and relation_model is not None:
            schema = relation_model.get_schema()
        self.schema = schema
        self.relation_model = relation_model

        # `label` is optional: when it (and a schema to infer it from) is absent,
        # the LM picks the relation label per call (see query_generator).
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
        self.fields = fields
        self.case_sensitive = case_sensitive

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

        # When the target label is fixed, the generator only produces the search
        # queries. When it is not, concatenate an enum field (the KB's actual
        # relation labels) onto the query schema so the LM also infers `label`.
        if self.label is None:
            gen_target = {
                "schema": concat_infer_fields(
                    RelationHybridRegexSearchInput.get_schema(),
                    [
                        (
                            "relation_label",
                            "The relation label to search, chosen to best "
                            "answer the inputs.",
                            kb_relation_labels(self.knowledge_base),
                        )
                    ],
                )
            }
        else:
            gen_target = {"data_model": RelationHybridRegexSearchInput}

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
            name="relation_hybrid_regex_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
        query_json = query.get_json()
        queries = query_json.get("similarity_search", [])
        patterns = query_json.get("regex_patterns")
        # Fixed label, or the one the LM inferred this call.
        label = self.label or query_json.get("relation_label")
        # Need at least one signal — vector or regex — and a label to look up.
        if (not queries and not patterns) or not label:
            return None

        rows = await self.knowledge_base.relation_hybrid_regex_search(
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
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
            "output_format": self.output_format,
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
        rm = self.relation_model
        if rm is not None and not is_symbolic_data_model(rm):
            rm = rm.to_symbolic_data_model(name="relation_model_" + self.name)
        relation_model_config = {
            "relation_model": (
                serialization_lib.serialize_synalinks_object(rm)
                if rm is not None
                else None
            ),
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **relation_model_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        relation_model_serialized = config.pop("relation_model", None)
        relation_model = (
            serialization_lib.deserialize_synalinks_object(relation_model_serialized)
            if relation_model_serialized is not None
            else None
        )
        return cls(
            knowledge_base=knowledge_base,
            language_model=language_model,
            relation_model=relation_model,
            **config,
        )
