# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import List
from typing import Optional

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


def default_text2sql_instructions() -> str:
    """Default instructions for the Text2SQL generator.

    The schema itself is concatenated to the inputs at call time (in the
    ``database_schema`` field), so the LM sees the full table/column
    listing in the user message and the instructions don't need to
    enumerate tables. This keeps the instructions stable when the
    underlying knowledge base gains or loses tables between calls.
    """
    return """
Your task is to translate the user request into a single read-only SQL
query against the tables described in the `database_schema` field.

Constraints:
- Emit exactly one `SELECT` statement. `INSERT`, `UPDATE`, `DELETE`,
  `DROP`, `ALTER`, `COPY ... TO`, and multi-statement queries are
  rejected by the engine.
- Table and column names are case-sensitive: tables are PascalCase
  (e.g. ``Customer``), columns are snake_case (e.g. ``customer_id``).
- Use the `database_schema` field already provided in the input to
  pick the right tables and columns — do not invent identifiers.
- When the user asks for "the top N" or "a few", add an explicit
  ``LIMIT`` clause; result sets are capped server-side anyway.
""".strip()


def _format_schema(knowledge_base) -> str:
    """Render the knowledge-base schema as a single text block.

    Mirrors the ``get_database_schema`` SQL agent tool — each table is
    listed with its PascalCase title and snake_case columns annotated
    with type and description — but returns a plain string so it can
    be embedded as a single field on a ``DatabaseSchema`` data model.
    """
    symbolic_models = knowledge_base.get_symbolic_data_models()
    sections = []
    for model in symbolic_models:
        schema = model.get_schema()
        table_name = schema.get("title", "Unknown")
        properties = schema.get("properties", {})
        columns = []
        for col_name, col_info in properties.items():
            col_type = col_info.get("type", "unknown")
            col_desc = col_info.get("description", "")
            columns.append(f"  - {col_name} ({col_type}): {col_desc}")
        sections.append(f"Table: {table_name}\n" + "\n".join(columns))
    return "\n\n".join(sections)


class DatabaseSchema(DataModel):
    """Schema description concatenated onto the inputs at call time."""

    database_schema: str = Field(
        description="The available tables and their columns",
    )


class SQLQuery(DataModel):
    """The structured output of the Text2SQL generator."""

    sql_query: str = Field(
        description="A single read-only SELECT SQL query",
    )


class SQLQueryResult(DataModel):
    """The result of executing the generated SQL query."""

    sql_query: str = Field(
        description="The SQL query that was executed",
    )
    result: List[Any] = Field(
        description="The rows returned by the query",
    )


@synalinks_export(
    [
        "synalinks.modules.Text2SQL",
        "synalinks.Text2SQL",
    ]
)
class Text2SQL(Module):
    """Translate a natural-language request into SQL and execute it.

    The module concatenates the knowledge base's schema (table names,
    columns, types, descriptions) onto the input data model so the LM
    has the full schema in context, generates a single ``SELECT``
    query, and runs it through :meth:`KnowledgeBase.sql` with
    ``read_only=True``.

    Safety is enforced by the knowledge base, not by string filtering.
    The DuckDB adapter parses the query with the engine's own parser
    and rejects anything that isn't a ``SELECT`` (including
    ``COPY ... TO 'file'`` exfiltration, ``ATTACH``, multi-statement
    injection), and the connection has ``enable_external_access=false``
    so ``read_csv`` / ``read_parquet`` / httpfs can't reach the host
    filesystem or network.

    Example:

    ```python
    import synalinks
    import asyncio

    class Customer(synalinks.DataModel):
        id: str = synalinks.Field(description="Customer ID")
        name: str = synalinks.Field(description="Customer name")
        country: str = synalinks.Field(description="Customer country")

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="Natural language question")

    async def main():
        kb = synalinks.KnowledgeBase(
            uri="duckdb://my_db.db",
            data_models=[Customer],
        )
        await kb.update(Customer(id="C1", name="Alice", country="USA"))

        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.Text2SQL(
            knowledge_base=kb,
            language_model=lm,
        )(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await program(Query(query="How many customers are in the USA?"))
        print(result.get("sql_query"))
        print(result.get("result"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to query.
            Required.
        language_model (LanguageModel): The language model that
            translates the request into SQL.
        k (int | None): Maximum number of rows returned by the
            executed query. The LM's SQL is wrapped in
            ``SELECT * FROM ({sql}) LIMIT k`` so unbounded
            ``SELECT *`` queries can't drain a large table into the
            response. A ``LIMIT`` inside the LM's own SQL still
            applies first. Pass ``None`` to disable the outer wrap
            and run the LM's SQL as-is. Defaults to 50.
        output_format (str): How the executed query renders its rows.
            ``"json"`` (default) returns a list of dicts; ``"csv"``
            returns a CSV string.
        prompt_template (str): Forwarded to the underlying
            :class:`Generator`.
        examples (list): Few-shot examples for the generator.
        instructions (str): Override the default system instructions.
            When omitted, the default is built from the knowledge
            base's table titles.
        seed_instructions (list): Optional seed instructions for
            prompt optimization.
        temperature (float): LM sampling temperature. Defaults to 0.0
            for deterministic SQL generation.
        reasoning_effort (str): Forwarded to the generator (for
            reasoning-capable LMs).
        use_inputs_schema (bool): Include the (input + schema) schema
            in the prompt.
        use_outputs_schema (bool): Include the output schema in the
            prompt.
        return_inputs (bool): Concatenate the original inputs onto
            the output. Defaults to ``False``.
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
        k: Optional[int] = 50,
        output_format: str = "json",
        prompt_template=None,
        examples=None,
        instructions: Optional[str] = None,
        seed_instructions=None,
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        use_inputs_schema: bool = False,
        use_outputs_schema: bool = False,
        return_inputs: bool = False,
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

        if output_format not in ("json", "csv"):
            raise ValueError(
                f"`output_format` must be 'json' or 'csv', got {output_format!r}"
            )
        self.output_format = output_format

        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError(f"`k` must be a positive integer or None, got {k!r}")
        self.k = k

        # Schema text is fetched at call time, not here — the KB can
        # gain tables between construction and use, and a Program
        # built once but called many times must not freeze a stale
        # schema. Default instructions are schema-agnostic for the
        # same reason; the table list ends up in the inputs.
        if instructions is None:
            instructions = default_text2sql_instructions()
        self.instructions = instructions
        self.seed_instructions = seed_instructions

        self.prompt_template = prompt_template
        self.examples = examples
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs

        self.generator = Generator(
            data_model=SQLQuery,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="generator_" + self.name,
        )

    def _schema_data_model(self) -> JsonDataModel:
        """Snapshot the KB schema as a data model for the current call.

        Re-fetched on every call so DDL changes (new tables, dropped
        tables, renamed columns) are picked up immediately without
        having to rebuild the module.
        """
        return JsonDataModel(
            json={"database_schema": _format_schema(self.knowledge_base)},
            schema=DatabaseSchema.get_schema(),
            name="database_schema_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        # Concatenate the schema onto the inputs so the LM sees both
        # the user request and the full table/column listing in a
        # single structured payload.
        inputs_with_schema = await ops.concat(
            inputs,
            self._schema_data_model(),
            name="inputs_with_schema_" + self.name,
        )

        sql_query_model = await self.generator(inputs_with_schema, training=training)
        if not sql_query_model:
            return None

        sql_query = sql_query_model.get_json().get("sql_query", "").strip()
        stripped = sql_query.rstrip(";").strip()
        if not stripped:
            result_rows: List[Any] = []
        else:
            # Wrap the LM's SQL in an outer LIMIT so even unbounded
            # SELECT * queries can't drain a large table. Any LIMIT
            # inside the LM's own query still applies first. When
            # ``k`` is None the wrap is skipped entirely.
            if self.k is None:
                executed_sql = stripped
            else:
                executed_sql = f"SELECT * FROM ({stripped}) LIMIT {self.k}"
            try:
                result_rows = await self.knowledge_base.sql(
                    executed_sql,
                    read_only=True,
                    output_format=self.output_format,
                )
            except Exception as e:
                result_rows = [{"error": str(e)}]

        output = JsonDataModel(
            json={"sql_query": sql_query, "result": result_rows},
            schema=SQLQueryResult.get_schema(),
            name=self.name,
        )

        if self.return_inputs:
            output = await ops.concat(
                inputs,
                output,
                name="output_with_inputs_" + self.name,
            )
        return output

    async def compute_output_spec(self, inputs, training=False):
        inputs_with_schema = await ops.concat(
            inputs,
            SymbolicDataModel(schema=DatabaseSchema.get_schema()),
            name="inputs_with_schema_" + self.name,
        )
        _ = await self.generator(inputs_with_schema, training=training)
        output = SymbolicDataModel(
            schema=SQLQueryResult.get_schema(),
            name=self.name,
        )
        if self.return_inputs:
            output = await ops.concat(
                inputs,
                output,
                name="output_with_inputs_" + self.name,
            )
        return output

    def get_config(self):
        config = {
            "k": self.k,
            "output_format": self.output_format,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs": self.return_inputs,
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
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            knowledge_base=knowledge_base,
            language_model=language_model,
            **config,
        )
