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
from synalinks.src.saving import serialization_lib


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
class RelationRegexSearch(Module):
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
            is not given. One of ``schema``, ``relation_model``, or
            ``label`` must be provided.
        label (str): Target relation label. Defaults to the schema's
            ``title``. One of ``schema``, ``relation_model``, or
            ``label`` must be provided.
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

    def __init__(
        self,
        *,
        knowledge_base=None,
        language_model=None,
        schema=None,
        relation_model=None,
        label: Optional[str] = None,
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

        if schema is None and relation_model is not None:
            schema = relation_model.get_schema()
        if schema is None and label is None:
            raise ValueError("One of `schema`, `relation_model`, or `label` is required")
        self.schema = schema
        self.relation_model = relation_model

        if label is None:
            label = schema.get("title")
            if not label:
                raise ValueError(
                    "Could not infer `label` from `schema` (no `title`); "
                    "pass `label` explicitly."
                )
        self.label = label

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
            data_model=RelationRegexSearchInput,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="relation_regex_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
        pattern = query.get_json().get("regex_search")
        if not pattern:
            return None

        rows = await self.knowledge_base.relation_regex_search(
            pattern,
            label=self.label,
            fields=self.fields,
            case_sensitive=self.case_sensitive,
            k=self.k,
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
