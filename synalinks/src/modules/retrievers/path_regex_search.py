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
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.modules.retrievers._path_helpers import deserialize_entity_model
from synalinks.src.modules.retrievers._path_helpers import resolve_endpoint
from synalinks.src.modules.retrievers._path_helpers import serialize_entity_model
from synalinks.src.saving import serialization_lib


class PathRegexSearchInput(DataModel):
    """Input shape for :class:`PathRegexSearch`."""

    subj_regex_search: str = Field(
        description="Regex pattern (RE2) for the subject endpoint",
    )
    obj_regex_search: str = Field(
        description="Regex pattern (RE2) for the object endpoint",
    )


@synalinks_export(
    [
        "synalinks.modules.PathRegexSearch",
        "synalinks.PathRegexSearch",
    ]
)
class PathRegexSearch(Module):
    """Regex variable-length path search where BOTH endpoints match.

    LM-driven wrapper around
    :meth:`KnowledgeBase.path_regex_search`. Returns paths of
    ``min_hops..max_hops`` edges whose subject endpoint matches
    ``subj_regex_search`` AND whose object endpoint matches
    ``obj_regex_search``. Regex uses RE2, so patterns are linear-time
    and not vulnerable to catastrophic backtracking.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        subj_schema (dict): JSON schema of the subject entity.
        subj_entity_model (Entity | SymbolicDataModel): Subject entity
            model. One of ``subj_schema``, ``subj_entity_model``, or
            ``subj_label`` must be provided.
        subj_label (str): Subject entity label.
        obj_schema (dict): JSON schema of the object entity.
        obj_entity_model (Entity | SymbolicDataModel): Object entity
            model. One of ``obj_schema``, ``obj_entity_model``, or
            ``obj_label`` must be provided.
        obj_label (str): Object entity label.
        rel_label (str): Optional rel-label constraint applied to
            every hop.
        min_hops (int): Minimum hop count, inclusive. Defaults to 1.
        max_hops (int): Maximum hop count, inclusive. Defaults to 3.
        k (int): Maximum number of results. Defaults to 10.
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

    def __init__(
        self,
        *,
        knowledge_base=None,
        language_model=None,
        subj_schema=None,
        subj_entity_model=None,
        subj_label: Optional[str] = None,
        obj_schema=None,
        obj_entity_model=None,
        obj_label: Optional[str] = None,
        rel_label: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        k: int = 10,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
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

        self.subj_schema, self.subj_label = resolve_endpoint(
            subj_schema, subj_entity_model, subj_label, "subj"
        )
        self.subj_entity_model = subj_entity_model
        self.obj_schema, self.obj_label = resolve_endpoint(
            obj_schema, obj_entity_model, obj_label, "obj"
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

        if output_format not in ("json", "csv"):
            raise ValueError(
                f"`output_format` must be 'json' or 'csv', got {output_format!r}"
            )
        self.output_format = output_format

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}")
        self.k = k
        self.fields = fields
        self.case_sensitive = case_sensitive

        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query

        self.query_generator = Generator(
            data_model=PathRegexSearchInput,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="path_regex_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
        payload = query.get_json()
        subj_pattern = payload.get("subj_regex_search")
        obj_pattern = payload.get("obj_regex_search")
        if not subj_pattern or not obj_pattern:
            return None

        rows = await self.knowledge_base.path_regex_search(
            subj_pattern=subj_pattern,
            obj_pattern=obj_pattern,
            subj_label=self.subj_label,
            obj_label=self.obj_label,
            label=self.rel_label,
            min_hops=self.min_hops,
            max_hops=self.max_hops,
            k=self.k,
            fields=self.fields,
            case_sensitive=self.case_sensitive,
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
            "subj_schema": self.subj_schema,
            "subj_label": self.subj_label,
            "obj_schema": self.obj_schema,
            "obj_label": self.obj_label,
            "rel_label": self.rel_label,
            "min_hops": self.min_hops,
            "max_hops": self.max_hops,
            "k": self.k,
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
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
        endpoint_models_config = {
            "subj_entity_model": serialize_entity_model(
                self.subj_entity_model, "subj_entity_model_" + self.name
            ),
            "obj_entity_model": serialize_entity_model(
                self.obj_entity_model, "obj_entity_model_" + self.name
            ),
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **endpoint_models_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        subj_entity_model = deserialize_entity_model(
            config.pop("subj_entity_model", None)
        )
        obj_entity_model = deserialize_entity_model(config.pop("obj_entity_model", None))
        return cls(
            knowledge_base=knowledge_base,
            language_model=language_model,
            subj_entity_model=subj_entity_model,
            obj_entity_model=obj_entity_model,
            **config,
        )
