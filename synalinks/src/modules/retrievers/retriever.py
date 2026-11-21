# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.backend import GenericResult
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.modules.retrievers.infer_helpers import concat_infer_fields
from synalinks.src.saving import serialization_lib


def resolve_target(schema, data_model, target):
    """Return `(schema, target)`, each inferred when not given.

    The schema comes from `data_model`, and the target (a table name or a
    label) from the schema's `title`; a target still `None` is left for the
    language model to infer per call.
    """
    if schema is None and data_model is not None:
        schema = data_model.get_schema()
    if target is None and schema is not None:
        target = schema.get("title") or None
    return schema, target


class Retriever(Module):
    """Base of the retrievers that turn their inputs into a knowledge base search.

    An embedded `Generator` turns the module's inputs into a `query_model`
    query, `search` runs it against the knowledge base, and the module returns
    the result (a `result_model`), preceded by the query when `return_query`
    and by the inputs when `return_inputs`. Each target the module was not
    given (a table name, a label) is an enum field the generator fills, among
    the knowledge base's own (see `build_query_generator`).

    A subclass sets `query_model`, `data_model_args` (its data model arguments,
    serialized in its config) and, when it is not a `GenericResult`,
    `result_model`; its `__init__` calls `build_query_generator` once its
    targets are known, and its `get_config` puts its own arguments before
    `super().get_config()`.
    """

    query_model = None
    result_model = GenericResult
    data_model_args = ()

    def __init__(
        self,
        *,
        knowledge_base=None,
        language_model=None,
        k=10,
        output_format="json",
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs=True,
        return_query=True,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.knowledge_base = self._own(_get_kb(knowledge_base), given=knowledge_base)
        self.language_model = _get_lm(language_model)
        if output_format not in ("json", "csv"):
            raise ValueError(
                f"`output_format` must be 'json' or 'csv', got {output_format!r}"
            )
        self.output_format = output_format
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}")
        self.k = k
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

    def build_query_generator(self, infer_specs):
        """Create the query generator, also inferring each `(field, description,
        values)` of `infer_specs` (an enum of `values`)."""
        if infer_specs:
            target = {
                "schema": concat_infer_fields(self.query_model.get_schema(), infer_specs)
            }
        else:
            target = {"data_model": self.query_model}
        stem = type(self).__module__.rsplit(".", 1)[-1]
        self.query_generator = Generator(
            **target,
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
            name=f"{stem}_query_generator_{self.name}",
        )

    async def search(self, query):
        """Run the generated `query` against the knowledge base.

        Returns the JSON of the `result_model`, or `None` when the query
        lacks what the search needs.
        """
        raise NotImplementedError

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
        result = await self.search(query)
        if result is None:
            return None
        results = JsonDataModel(
            json=result,
            schema=self.result_model.get_schema(),
            name=self.name,
        )
        return await self._with_query_and_inputs(inputs, query, results)

    async def compute_output_spec(self, inputs, training=False):
        query = await self.query_generator(inputs, training=training)
        results = SymbolicDataModel(
            schema=self.result_model.get_schema(),
            name=self.name,
        )
        return await self._with_query_and_inputs(inputs, query, results)

    async def _with_query_and_inputs(self, inputs, query, results):
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
            "knowledge_base": serialization_lib.serialize_synalinks_object(
                self.knowledge_base,
            ),
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            ),
        }
        for arg in self.data_model_args:
            data_model = getattr(self, arg)
            if data_model is not None and not is_symbolic_data_model(data_model):
                data_model = data_model.to_symbolic_data_model(name=f"{arg}_{self.name}")
            config[arg] = (
                serialization_lib.serialize_synalinks_object(data_model)
                if data_model is not None
                else None
            )
        return config

    @classmethod
    def from_config(cls, config):
        config["knowledge_base"] = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        config["language_model"] = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        for arg in cls.data_model_args:
            serialized = config.pop(arg, None)
            config[arg] = (
                serialization_lib.deserialize_synalinks_object(serialized)
                if serialized is not None
                else None
            )
        return cls(**config)
