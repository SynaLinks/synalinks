# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_table_names
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


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
class RegexSearch(Retriever):
    """Regex matching against a single KB table.

    LM-driven wrapper around `KnowledgeBase.regex_search`. An
    embedded `Generator` turns the module's inputs into a
    `RegexSearchInput` query (the ``regex_search`` field),
    which is then matched against the table. Use this when you want
    exact-shape textual matching without the vector branch (e.g.
    matching log lines, SKU codes, or any other structured text).

    Regex uses RE2 (DuckDB's engine), so patterns are linear-time and
    not vulnerable to catastrophic backtracking, safe even when the
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
            is not given.
        table_name (str): Target table. Defaults to the schema's
            ``title``. **Optional**: when neither ``table_name`` nor a
            schema to derive it from is given, the language model infers
            the target table per call (constrained to the knowledge
            base's actual tables).
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
        temperature (float): Optional temperature for the language model.
            Defaults to None (the model's default).
        max_tokens (int): Optional cap on the generation length. Defaults
            to None (the model's default).
        top_p (float): Optional nucleus-sampling probability. Defaults to
            None (the model's default).
        top_k (int): Optional top-k sampling cutoff. Defaults to None
            (the model's default).
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

    query_model = RegexSearchInput
    data_model_args = ("data_model",)

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        table_name: Optional[str] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.table_name = resolve_target(schema, data_model, table_name)
        self.data_model = data_model
        self.fields = fields
        self.case_sensitive = case_sensitive

        self.build_query_generator(
            [
                (
                    "table_name",
                    "The knowledge-base table to search, chosen to best "
                    "answer the inputs.",
                    kb_table_names(self.knowledge_base),
                )
            ]
            if self.table_name is None
            else []
        )

    async def search(self, query):
        query_json = query.get_json()
        pattern = query_json.get("regex_search")
        # Fixed table, or the one the LM inferred this call.
        table_name = self.table_name or query_json.get("table_name")
        if not pattern or not table_name:
            return None

        rows = await self.knowledge_base.regex_search(
            pattern,
            table_name=table_name,
            fields=self.fields,
            case_sensitive=self.case_sensitive,
            k=self.k,
            output_format=self.output_format,
        )
        return {"result": rows}

    def get_config(self):
        return {
            "schema": self.schema,
            "table_name": self.table_name,
            "k": self.k,
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
            "output_format": self.output_format,
            **super().get_config(),
        }
