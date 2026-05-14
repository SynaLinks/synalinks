# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


def get_default_instructions(tables: List[str]) -> str:
    """Default instructions for the SQL agent.

    Args:
        tables: The PascalCase names of tables available in the
            knowledge base. Embedded in the prompt so the LM doesn't
            have to call ``get_database_schema`` first for trivial
            lookups.

    Returns:
        A prompt string giving the LM the tool-use plan and the
        SELECT-only safety constraint.
    """
    return f"""
You are an SQL analyst with read-only access to a knowledge base.

Available tables: {tables}

Plan:
1. If you don't already know the schema, call `get_database_schema` first.
2. When you need to inspect representative values, call `get_table_sample`.
3. Build a single `SELECT` query and execute it with `run_sql_query`. Iterate
   on the query (read the error, fix the SQL, retry) until you have the data.
4. Once you have an answer, stop calling tools and produce the final response.

Constraints:
- Only `SELECT` statements are accepted. `INSERT`, `UPDATE`, `DELETE`,
  `DROP`, `ALTER`, `COPY ... TO`, and multi-statement queries are
  rejected by the engine — don't waste turns trying them.
- Table and column names are case-sensitive: tables are PascalCase
  (e.g. ``Customer``), columns are snake_case (e.g. ``customer_id``).
- Result sets are automatically capped server-side. If the result
  shows ``may_have_more=true``, refine the query (add filters or
  ``ORDER BY ... LIMIT n``) rather than asking for more rows.
""".strip()


def _row_count(results, output_format: str) -> int:
    """Count rows in a result set regardless of ``output_format``.

    For ``"csv"`` the result is a string with a header row; for
    ``"json"`` it's a list of dicts. Returns 0 for empty inputs.
    """
    if output_format == "csv":
        if not results:
            return 0
        lines = results.splitlines()
        return max(len(lines) - 1, 0)
    return len(results)


def _build_tools(knowledge_base, output_format: str = "csv", k: int = 50):
    """Build the three SQL tools bound to a knowledge base.

    Returns a list of plain async functions (not wrapped as ``Tool``
    objects yet) so the caller can decide how to register them.

    ``output_format`` is shared by ``get_table_sample`` and
    ``run_sql_query`` and fixed per-agent at construction time:
    ``"csv"`` (default) compacts result sets so the LM spends fewer
    input tokens reading them; ``"json"`` returns a list of dicts.
    The ``get_database_schema`` tool ignores the format — its output
    is always a small textual summary.

    ``k`` caps the page size the LM can pull through
    ``get_table_sample``: whatever ``limit`` the LM passes is clamped
    to ``min(limit, k)`` so a runaway request can't drain a large
    table into the conversation.
    """

    async def get_database_schema():
        """Get the complete database schema including all tables and their columns.

        Returns a list of all tables with their column names, types, and
        descriptions. Use this first to discover what data is available
        before writing queries.
        """
        symbolic_models = knowledge_base.get_symbolic_data_models()
        schema_info = []
        for model in symbolic_models:
            schema = model.get_schema()
            table_name = schema.get("title", "Unknown")
            properties = schema.get("properties", {})
            columns = []
            for col_name, col_info in properties.items():
                col_type = col_info.get("type", "unknown")
                col_desc = col_info.get("description", "")
                columns.append(f"  - {col_name} ({col_type}): {col_desc}")
            schema_info.append(f"Table: {table_name}\n" + "\n".join(columns))
        return {
            "schema": "\n\n".join(schema_info),
            "table_count": len(symbolic_models),
        }

    async def get_table_sample(table_name: str, limit: int, offset: int):
        """Get a sample of rows from a table to understand the data format.

        Args:
            table_name (str): The name of the table to sample (PascalCase).
            limit (int): Number of rows to return (recommended: 3-5).
                Capped server-side at the agent's ``k`` setting.
            offset (int): Number of rows to skip (use 0 to start at the top).
        """
        symbolic_models = knowledge_base.get_symbolic_data_models()
        available_tables = [
            m.get_schema().get("title", "Unknown") for m in symbolic_models
        ]
        if table_name not in available_tables:
            return {
                "error": (
                    f"Table {table_name!r} not found. "
                    f"Available: {available_tables}"
                )
            }
        effective_limit = max(1, min(limit, k))
        # Table name validated against the known-tables list, so direct
        # interpolation is safe; LIMIT / OFFSET are bound parameters.
        sql = f"SELECT * FROM {table_name} LIMIT ? OFFSET ?"
        try:
            results = await knowledge_base.query(
                sql,
                params=[effective_limit, offset],
                read_only=True,
                output_format=output_format,
            )
        except Exception as e:
            return {"error": str(e)}
        return {
            "table": table_name,
            "sample_data": results,
            "row_count": _row_count(results, output_format),
            "output_format": output_format,
            "limit": effective_limit,
            "offset": offset,
            "limit_capped": effective_limit < limit,
        }

    async def run_sql_query(sql_query: str):
        """Execute a read-only SELECT query and return the rows.

        Only SELECT statements are accepted. The knowledge base parses
        the query with DuckDB's own parser and rejects non-SELECT
        statements (writes, COPY-to-file, ATTACH, EXPORT, multi-
        statement injection). External file / network access is also
        blocked at the connection level.

        Results are capped to the agent's ``k`` setting via an outer
        ``LIMIT``. Any ``LIMIT`` clause inside the user query still
        applies first, so writing ``LIMIT 5`` returns 5 rows even
        when the cap is higher.

        Args:
            sql_query (str): A SELECT SQL query to execute.
        """
        stripped = sql_query.strip().rstrip(";").strip()
        if not stripped:
            return {
                "success": False,
                "error": "Empty SQL query.",
                "query": sql_query,
            }
        wrapped_sql = f"SELECT * FROM ({stripped}) LIMIT {k}"
        try:
            results = await knowledge_base.query(
                wrapped_sql, read_only=True, output_format=output_format
            )
        except Exception as e:
            return {"success": False, "error": str(e), "query": sql_query}
        rc = _row_count(results, output_format)
        return {
            "success": True,
            "query": sql_query,
            "row_count": rc,
            "row_cap": k,
            "results": results,
            "output_format": output_format,
            "may_have_more": rc >= k,
        }

    return [get_database_schema, get_table_sample, run_sql_query]


@synalinks_export(
    [
        "synalinks.modules.SQLAgent",
        "synalinks.SQLAgent",
    ]
)
class SQLAgent(Module):
    """A ready-to-use SQL agent backed by a knowledge base.

    SQLAgent is a thin specialization of :class:`FunctionCallingAgent`
    that pre-wires three SQL tools bound to a :class:`KnowledgeBase`:

    - ``get_database_schema``: discovers all tables and their columns.
    - ``get_table_sample``: fetches a few rows so the LM can see the
      data shape before writing queries.
    - ``run_sql_query``: executes a ``SELECT`` query via
      :meth:`KnowledgeBase.query` with ``read_only=True``.

    The constructor mirrors :class:`FunctionCallingAgent` — every
    parameter on that class is accepted here with identical
    semantics. The only additions are ``knowledge_base`` (required)
    and ``output_format`` (controls the SQL tools' result rendering).
    User-supplied ``tools`` are appended to the three built-in tools.

    Safety is enforced by the knowledge base, not by string filtering.
    The DuckDB adapter parses the query with the engine's parser and
    rejects anything that isn't a ``SELECT`` (including
    ``COPY ... TO 'file'`` exfiltration, ``ATTACH``, multi-statement
    injection), and the connection has
    ``enable_external_access=false`` so ``read_csv`` / ``read_parquet``
    / httpfs can't reach the host filesystem or network.

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

    class SQLAnswer(synalinks.DataModel):
        answer: str = synalinks.Field(description="Answer in natural language")
        sql_query: str = synalinks.Field(description="SQL that produced it")

    async def main():
        kb = synalinks.KnowledgeBase(
            uri="duckdb://my_db.db",
            data_models=[Customer],
        )
        await kb.update(Customer(id="C1", name="Alice", country="USA"))

        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
        )(inputs)
        agent = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await agent(Query(query="How many customers are in the USA?"))
        print(result.get("answer"))
        print(result.get("sql_query"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to query.
            Required.
        k (int): Maximum page size (rows per call) the LM can pull
            through ``get_table_sample`` and ``run_sql_query``.
            ``get_table_sample`` clamps the LM's ``limit`` argument
            to ``min(limit, k)``; ``run_sql_query`` wraps the LM's
            SQL in ``SELECT * FROM ({sql}) LIMIT k`` so even unbounded
            ``SELECT *`` queries can't drain a large table into the
            conversation. A ``LIMIT`` inside the LM's own SQL still
            applies first. Defaults to 50.
        output_format (str): How the SQL tools render result sets to
            the LM. ``"csv"`` (default) is compact and minimizes input
            tokens; ``"json"`` returns a list of dicts. Applies to both
            ``get_table_sample`` and ``run_sql_query``.
        tools (list): Additional :class:`Tool` instances (or plain
            async functions) to expose alongside the three built-in
            SQL tools — for example a calculator, a datetime helper,
            a web-search tool. Tool names must not collide with the
            built-ins (``get_database_schema``, ``get_table_sample``,
            ``run_sql_query``) or a ``ValueError`` is raised.
        schema (dict): JSON schema for the final answer.
        data_model (DataModel): DataModel for the final answer.
            Mutually exclusive with ``schema``.
        language_model (LanguageModel): The language model that drives
            the agent loop.
        prompt_template (str): Forwarded to the tool-call generator.
        examples (list): Few-shot examples for the tool-call generator.
        instructions (str): Override the default system instructions.
            When omitted, the default is built from the knowledge
            base's tables so the LM knows what's available without an
            extra schema call.
        final_instructions (str): Instructions for the final-answer
            generator. Defaults to ``instructions``.
        temperature (float): LM sampling temperature. Defaults to 0.0
            for deterministic SQL generation.
        use_inputs_schema (bool): Include the input schema in the
            prompt.
        use_outputs_schema (bool): Include the output schema in the
            prompt.
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When ``True``, the tool-call
            generator emits a ``thinking`` field per round.
        autonomous (bool): When ``True`` (default), the agent runs the
            tool loop end-to-end. When ``False``, returns one step at
            a time for human-in-the-loop workflows.
        return_inputs_with_trajectory (bool): When ``True`` (default),
            the full message trajectory is included alongside the
            final answer.
        max_iterations (int): Maximum number of tool-call rounds.
            Defaults to 5.
        streaming (bool): Stream the final answer when no ``schema``
            is set. Defaults to ``False``.
        name (str): Module name.
        description (str): Module description.
    """

    def __init__(
        self,
        *,
        knowledge_base=None,
        k: int = 50,
        output_format: str = "csv",
        tools: Optional[List] = None,
        schema=None,
        data_model=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions: Optional[str] = None,
        final_instructions: Optional[str] = None,
        temperature: float = 0.0,
        use_inputs_schema: bool = False,
        use_outputs_schema: bool = False,
        reasoning_effort: Optional[str] = None,
        use_chain_of_thought: bool = False,
        autonomous: bool = True,
        return_inputs_with_trajectory: bool = True,
        max_iterations: int = 5,
        streaming: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(name=name, description=description)

        if knowledge_base is None:
            raise ValueError("`knowledge_base` is required")
        self.knowledge_base = knowledge_base
        self.language_model = _get_lm(language_model)

        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        if output_format not in ("csv", "json"):
            raise ValueError(
                f"`output_format` must be 'csv' or 'json', got {output_format!r}"
            )
        self.output_format = output_format

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}")
        self.k = k

        if instructions is None:
            tables = [
                m.get_schema().get("title", "Unknown")
                for m in self.knowledge_base.get_symbolic_data_models()
            ]
            instructions = get_default_instructions(tables)
        self.instructions = instructions
        self.final_instructions = final_instructions

        self.prompt_template = prompt_template
        self.examples = examples
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        self.autonomous = autonomous
        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.max_iterations = max_iterations
        self.streaming = streaming

        builtin_tools = [
            Tool(fn)
            for fn in _build_tools(
                self.knowledge_base,
                output_format=self.output_format,
                k=self.k,
            )
        ]
        builtin_names = {t.name for t in builtin_tools}

        # Stash the user-supplied tools as-is for get_config round-trips;
        # the merged list goes to FunctionCallingAgent below.
        self.extra_tools = list(tools) if tools else []
        merged_tools = list(builtin_tools)
        for extra in self.extra_tools:
            extra_tool = extra if isinstance(extra, Tool) else Tool(extra)
            if extra_tool.name in builtin_names:
                raise ValueError(
                    f"Tool name {extra_tool.name!r} collides with a built-in "
                    f"SQL tool. Rename the additional tool."
                )
            merged_tools.append(extra_tool)
        # Leading-underscore check is centralized in FunctionCallingAgent.

        self.agent = FunctionCallingAgent(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            final_instructions=self.final_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            reasoning_effort=self.reasoning_effort,
            use_chain_of_thought=self.use_chain_of_thought,
            tools=merged_tools,
            autonomous=self.autonomous,
            return_inputs_with_trajectory=self.return_inputs_with_trajectory,
            max_iterations=self.max_iterations,
            streaming=self.streaming,
            name="agent_" + self.name,
        )

    async def call(self, inputs, training=False):
        return await self.agent(inputs, training=training)

    async def compute_output_spec(self, inputs, training=False):
        return await self.agent.compute_output_spec(inputs, training=training)

    def get_config(self):
        config = {
            "schema": self.schema,
            "k": self.k,
            "output_format": self.output_format,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "final_instructions": self.final_instructions,
            "temperature": self.temperature,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "reasoning_effort": self.reasoning_effort,
            "use_chain_of_thought": self.use_chain_of_thought,
            "autonomous": self.autonomous,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
            "max_iterations": self.max_iterations,
            "streaming": self.streaming,
            "name": self.name,
            "description": self.description,
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
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(
                    t if isinstance(t, Tool) else Tool(t)
                )
                for t in self.extra_tools
            ]
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **tools_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        tools = [
            serialization_lib.deserialize_synalinks_object(t)
            for t in config.pop("tools", [])
        ]
        return cls(
            knowledge_base=knowledge_base,
            language_model=language_model,
            tools=tools,
            **config,
        )
