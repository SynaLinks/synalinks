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


class RegexSearchInput(DataModel):
    """Input shape for `RegexSearch`."""

    regex_search: str = Field(
        description=(
            "Regex pattern (RE2 syntax) to match against the table's string fields"
        ),
    )


@synalinks_export(
    [
        "synalinks.modules.RegexSearch",
        "synalinks.RegexSearch",
    ]
)
class RegexSearch(Module):
    """Regex matching against a single KB table.

    LM-driven wrapper around `KnowledgeBase.regex_search`. An
    embedded `Generator` turns the module's inputs into a
    `RegexSearchInput` query (the ``regex_search`` field),
    which is then matched against the table. Use this when you want
    exact-shape textual matching without the vector branch (e.g.
    matching log lines, SKU codes, or any other structured text).

    Regex uses RE2 (DuckDB's engine), so patterns are linear-time and
    not vulnerable to catastrophic backtracking — safe even when the
    pattern comes from an untrusted source.

    Single-table only: to retrieve from multiple tables, compose
    several `RegexSearch` modules in the program DAG and merge
    their outputs explicitly.

    Example:

    ```python
    import synalinks
    import asyncio

    class LogLine(synalinks.DataModel):
        id: str = synalinks.Field(description="Log id")
        text: str = synalinks.Field(description="Log line")

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="The user question")

    async def main():
        kb = synalinks.KnowledgeBase(
            uri="duckdb://logs.db",
            data_models=[LogLine],
        )
        lm = synalinks.LanguageModel(model="ollama/mistral")
        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.RegexSearch(
            knowledge_base=kb,
            language_model=lm,
            data_model=LogLine,
            k=5,
        )(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await program(Query(query="find error codes in the logs"))
        print(result.get("result"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        language_model (LanguageModel): The language model used to
            generate the ``regex_search`` pattern from the inputs.
        schema (dict): JSON schema of the table's row. Used to infer
            ``table_name`` from its ``title`` when not given
            explicitly. Mutually inferrable with ``data_model``.
        data_model (DataModel | SymbolicDataModel): Data model
            providing ``schema`` via ``.get_schema()`` when ``schema``
            is not given. One of ``schema``, ``data_model``, or
            ``table_name`` must be provided.
        table_name (str): Target table. Defaults to the schema's
            ``title``. One of ``schema``, ``data_model``, or
            ``table_name`` must be provided.
        k (int): Maximum number of results to return. Defaults to 10.
        fields (list): Field names to match against. Defaults to
            every string field on the schema. Names are
            snake_case-normalized to match stored column names.
        case_sensitive (bool): When ``False``, regex matches are
            case-insensitive. Defaults to ``True``.
        output_format (str): How the underlying adapter renders rows.
            ``"json"`` (default, list of dicts) or ``"csv"`` (CSV
            string).
        prompt_template (str): Custom prompt template for the query
            generator.
        examples (list): Example inputs/outputs for few-shot learning.
        instructions (str): Custom instructions for the query generator.
        seed_instructions (str): Seed instructions for variability.
        temperature (float): Temperature for the language model.
            Defaults to 0.0.
        use_inputs_schema (bool): Whether to include the input schema
            in the prompt. Defaults to False.
        use_outputs_schema (bool): Whether to include the output schema
            in the prompt. Defaults to False.
        return_inputs (bool): Whether to include the original inputs in
            the output. Defaults to True.
        return_query (bool): Whether to include the generated query in
            the output. Defaults to True.
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
        data_model=None,
        table_name: Optional[str] = None,
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

        if schema is None and data_model is not None:
            schema = data_model.get_schema()
        if schema is None and table_name is None:
            raise ValueError("One of `schema`, `data_model`, or `table_name` is required")
        self.schema = schema
        self.data_model = data_model

        if table_name is None:
            table_name = schema.get("title")
            if not table_name:
                raise ValueError(
                    "Could not infer `table_name` from `schema` "
                    "(no `title`); pass `table_name` explicitly."
                )
        self.table_name = table_name

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
            data_model=RegexSearchInput,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="regex_search_query_generator_" + self.name,
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

        rows = await self.knowledge_base.regex_search(
            pattern,
            table_name=self.table_name,
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
            "table_name": self.table_name,
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
        dm = self.data_model
        if dm is not None and not is_symbolic_data_model(dm):
            dm = dm.to_symbolic_data_model(name="data_model_" + self.name)
        data_model_config = {
            "data_model": (
                serialization_lib.serialize_synalinks_object(dm)
                if dm is not None
                else None
            ),
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **data_model_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        data_model_serialized = config.pop("data_model", None)
        data_model = (
            serialization_lib.deserialize_synalinks_object(data_model_serialized)
            if data_model_serialized is not None
            else None
        )
        return cls(
            knowledge_base=knowledge_base,
            language_model=language_model,
            data_model=data_model,
            **config,
        )
