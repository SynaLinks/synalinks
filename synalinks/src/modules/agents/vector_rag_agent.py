# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.tool import Tool
from synalinks.src.saving import serialization_lib

SEARCH_TYPES = ("similarity", "fulltext", "hybrid_fts")


def get_default_instructions(tables: List[str], search_type: str) -> str:
    """Default system instructions for the RAG agent.

    Args:
        tables: PascalCase names of tables available for retrieval.
            Embedded in the prompt so the LM can pick a target table
            without a separate schema call.
        search_type: Which retrieval mode the agent is configured for.
            Shapes the guidance on how to phrase ``query`` arguments.

    Returns:
        A prompt string giving the LM the retrieval loop and the
        query-writing guidance for the configured search mode.
    """
    if search_type == "similarity":
        retrieval_hint = (
            "Use natural-language descriptions of what you need — the "
            "search is vector-similarity over embeddings, so paraphrase "
            "the user's intent rather than guessing keywords."
        )
    elif search_type == "fulltext":
        retrieval_hint = (
            "Use keyword-rich queries — the search is BM25 full-text, "
            "so the words you pick must appear in the documents."
        )
    else:  # hybrid_fts
        retrieval_hint = (
            "Use natural-language queries that contain the keywords you "
            "expect to appear in matching documents — the search fuses "
            "vector similarity and BM25 with Reciprocal Rank Fusion, so "
            "both signals contribute."
        )
    return f"""
You are a retrieval-augmented assistant with access to a knowledge base.

Available tables: {tables}
Search mode: {search_type}

Plan:
1. If you don't already know what's available, call `get_knowledge_base_schema`.
2. Call `search_knowledge_base` with the table you want and a query.
   {retrieval_hint}
3. If a search result references an id you want to inspect in full,
   call `get_record_by_id`.
4. Once you have enough context, stop calling tools and answer.

Constraints:
- Only retrieve when the user's question actually needs grounded
  information. Trivial questions don't need a search.
- Reformulate the user's question into focused queries; don't just pass
  the raw user text.
- If a search returns nothing useful, retry with a different phrasing
  before giving up.
""".strip()


def _row_count(results, output_format: str) -> int:
    """Count rows in a result set regardless of ``output_format``."""
    if output_format == "csv":
        if not results:
            return 0
        lines = results.splitlines()
        return max(len(lines) - 1, 0)
    return len(results)


def _build_tools(
    knowledge_base,
    *,
    search_type: str = "hybrid_fts",
    k: int = 5,
    similarity_threshold: Optional[float] = None,
    fulltext_threshold: Optional[float] = None,
    output_format: str = "csv",
):
    """Build the three RAG tools bound to a knowledge base.

    Returns a list of plain async functions (not wrapped as ``Tool``
    objects) so the caller can decide how to register them.

    Args:
        knowledge_base: The knowledge base to retrieve from.
        search_type: Which retrieval method the ``search_knowledge_base``
            tool dispatches to. One of ``"similarity"``, ``"fulltext"``,
            ``"hybrid_fts"``.
        k: Top-k for searches. Fixed per-agent — not exposed to the LM.
        similarity_threshold: Maximum vector distance for the similarity
            and hybrid modes.
        fulltext_threshold: Minimum BM25 score for the fulltext and
            hybrid modes.
        output_format: ``"csv"`` (default, compact) or ``"json"``
            (list of dicts). Applies to search results.
    """

    async def get_knowledge_base_schema():
        """Get the schema of every table in the knowledge base.

        Returns a textual summary of each table with its columns and
        types. Use this first to discover what tables are available
        before calling ``search_knowledge_base``.
        """
        symbolic_models = knowledge_base.get_symbolic_data_models()
        schema_info = []
        for model in symbolic_models:
            schema = model.get_schema()
            table_name = schema.get("title", "Unknown")
            table_desc = schema.get("description", "")
            properties = schema.get("properties", {})
            columns = []
            for col_name, col_info in properties.items():
                col_type = col_info.get("type", "unknown")
                col_desc = col_info.get("description", "")
                columns.append(f"  - {col_name} ({col_type}): {col_desc}")
            header = f"Table: {table_name}"
            if table_desc:
                header += f"\n  {table_desc}"
            schema_info.append(header + "\n" + "\n".join(columns))
        return {
            "schema": "\n\n".join(schema_info),
            "table_count": len(symbolic_models),
        }

    async def search_knowledge_base(table_name: str, query: str):
        """Search a table for records relevant to a query.

        Returns up to ``k`` results (configured per-agent at
        construction time, default 5).

        Args:
            table_name (str): The table to search (PascalCase, e.g.
                ``Document``). Call ``get_knowledge_base_schema`` if
                you're unsure what's available.
            query (str): The query string. For similarity / hybrid
                modes, use natural language. For fulltext mode, use
                keyword-rich phrasing.
        """
        symbolic_models = knowledge_base.get_symbolic_data_models()
        available_tables = [
            m.get_schema().get("title", "Unknown") for m in symbolic_models
        ]
        if table_name not in available_tables:
            return {
                "error": (
                    f"Table {table_name!r} not found. Available: {available_tables}"
                )
            }
        try:
            if search_type == "similarity":
                results = await knowledge_base.similarity_search(
                    query,
                    table_name=table_name,
                    k=k,
                    threshold=similarity_threshold,
                    output_format=output_format,
                )
            elif search_type == "fulltext":
                results = await knowledge_base.fulltext_search(
                    query,
                    table_name=table_name,
                    k=k,
                    threshold=fulltext_threshold,
                    output_format=output_format,
                )
            else:  # hybrid_fts
                results = await knowledge_base.hybrid_fts_search(
                    query,
                    table_name=table_name,
                    k=k,
                    similarity_threshold=similarity_threshold,
                    fulltext_threshold=fulltext_threshold,
                    output_format=output_format,
                )
        except Exception as e:
            return {"error": str(e), "query": query, "table": table_name}
        return {
            "table": table_name,
            "query": query,
            "search_type": search_type,
            "results": results,
            "row_count": _row_count(results, output_format),
            "output_format": output_format,
        }

    async def get_record_by_id(table_name: str, record_id: str):
        """Fetch a single record by its primary key.

        Use this after a search returns ids and you need to read the
        full record (e.g. long content fields that were truncated in
        search results).

        Args:
            table_name (str): The table the record lives in
                (PascalCase).
            record_id (str): The primary-key value of the record.
        """
        symbolic_models = knowledge_base.get_symbolic_data_models()
        available_tables = [
            m.get_schema().get("title", "Unknown") for m in symbolic_models
        ]
        if table_name not in available_tables:
            return {
                "error": (
                    f"Table {table_name!r} not found. Available: {available_tables}"
                )
            }
        try:
            record = await knowledge_base.get(record_id, table_name=table_name)
        except Exception as e:
            return {"error": str(e), "table": table_name, "record_id": record_id}
        if record is None:
            return {
                "table": table_name,
                "record_id": record_id,
                "found": False,
            }
        return {
            "table": table_name,
            "record_id": record_id,
            "found": True,
            "record": record.get_json(),
        }

    return [get_knowledge_base_schema, search_knowledge_base, get_record_by_id]


@synalinks_export(
    [
        "synalinks.modules.VectorRAGAgent",
        "synalinks.VectorRAGAgent",
    ]
)
class VectorRAGAgent(FunctionCallingAgent):
    """A ready-to-use retrieval-augmented agent backed by a knowledge base.

    VectorRAGAgent is a thin specialization of
    `FunctionCallingAgent` that pre-wires three retrieval tools
    bound to a `KnowledgeBase`:

    - ``get_knowledge_base_schema``: lists available tables and columns.
    - ``search_knowledge_base``: dispatches to similarity / fulltext /
      hybrid_fts depending on the configured ``search_type``.
    - ``get_record_by_id``: full-record lookup after a search returns
      an id.

    The constructor mirrors `FunctionCallingAgent` — every
    parameter on that class is accepted here with identical semantics.
    The only additions are ``knowledge_base`` (required), the
    retrieval knobs (``search_type``, ``k``, ``similarity_threshold``,
    ``fulltext_threshold``), and ``output_format``. User-supplied
    ``tools`` are appended to the three built-in retrieval tools.

    Compared to a hardcoded RAG pipeline (always retrieve, then
    answer), the agent decides *if* retrieval is needed, *which* table
    to search, and *how* to phrase the query. Multiple searches per
    turn are allowed.

    Example:

    ```python
    import synalinks
    import asyncio

    class Document(synalinks.DataModel):
        id: str = synalinks.Field(description="Document id")
        title: str = synalinks.Field(description="Title")
        content: str = synalinks.Field(description="Body text")

    async def main():
        embedding_model = synalinks.EmbeddingModel(
            model="gemini/text-embedding-004",
        )
        kb = synalinks.KnowledgeBase(
            uri="duckdb://docs.db",
            data_models=[Document],
            embedding_model=embedding_model,
        )
        # ... populate kb ...

        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await synalinks.VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
        )(inputs)
        agent = synalinks.Program(inputs=inputs, outputs=outputs)

        messages = synalinks.ChatMessages(messages=[
            synalinks.ChatMessage(role="user", content="What is the PTO policy?")
        ])
        result = await agent(messages)
        print(result.get("messages")[-1].get("content"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to retrieve
            from. Required.
        search_type (str): Retrieval mode for the
            ``search_knowledge_base`` tool. One of:

            - ``"similarity"``: vector-similarity over embeddings.
            - ``"fulltext"``: BM25 keyword search.
            - ``"hybrid_fts"`` (default): vector + BM25 fused with RRF.

            Requires the knowledge base to have an embedding model
            configured for ``"similarity"`` and ``"hybrid_fts"``.
        k (int): Top-k for searches. Fixed per-agent at construction
            time — the LM doesn't pass it. Defaults to 5.
        similarity_threshold (float): Maximum vector distance for the
            similarity and hybrid modes. Optional.
        fulltext_threshold (float): Minimum BM25 score for the
            fulltext and hybrid modes. Optional.
        output_format (str): How search results are rendered to the
            LM. ``"csv"`` (default) is compact; ``"json"`` returns a
            list of dicts.
        tools (list): Additional `Tool` instances (or plain
            async functions) to expose alongside the three built-in
            retrieval tools. Tool names must not collide with the
            built-ins (``get_knowledge_base_schema``,
            ``search_knowledge_base``, ``get_record_by_id``) or a
            ``ValueError`` is raised.
        schema (dict): JSON schema for the final answer.
        data_model (DataModel): DataModel for the final answer.
            Mutually exclusive with ``schema``.
        language_model (LanguageModel): The language model that drives
            the agent loop.
        prompt_template (str): Forwarded to the tool-call generator.
        examples (list): Few-shot examples for the tool-call generator.
        instructions (str): Override the default system instructions.
            When omitted, defaults are built from the knowledge base's
            tables and the configured ``search_type``.
        final_instructions (str): Instructions for the final-answer
            generator. Defaults to ``instructions``.
        temperature (float): LM sampling temperature. Defaults to 0.0.
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
        workdir (str): Optional. Path to a working directory. When it contains an
            ``AGENTS.md`` file, its contents are injected as an additional input
            so the agent follows the declared project conventions. Defaults to
            ``None``.
        skills (list): Optional. Folder paths (Agent Skill roots) whose skills
            are listed for the agent as an ``<available_skills>`` context message
            (see `FunctionCallingAgent`). Defaults to ``None``.
        name (str): Module name.
        description (str): Module description.
    """

    def __init__(
        self,
        *,
        knowledge_base=None,
        search_type: str = "hybrid_fts",
        k: int = 5,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
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
        workdir: Optional[str] = None,
        skills=None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        if knowledge_base is None:
            raise ValueError(
                "`knowledge_base` is required for VectorRAGAgent: pass a "
                "KnowledgeBase (or a URI) the agent can query."
            )
        # Domain attributes the `_get_builtin_tools` hook depends on must be set
        # before `super().__init__()` (which calls the hook).
        self.knowledge_base = _get_kb(knowledge_base)

        if search_type not in SEARCH_TYPES:
            raise ValueError(
                f"`search_type` must be one of {SEARCH_TYPES}, got {search_type!r}"
            )
        self.search_type = search_type

        if output_format not in ("csv", "json"):
            raise ValueError(
                f"`output_format` must be 'csv' or 'json', got {output_format!r}"
            )
        self.output_format = output_format

        self.k = k
        self.similarity_threshold = similarity_threshold
        self.fulltext_threshold = fulltext_threshold

        if instructions is None:
            tables = [
                m.get_schema().get("title", "Unknown")
                for m in self.knowledge_base.get_symbolic_data_models()
            ]
            instructions = get_default_instructions(tables, self.search_type)

        super().__init__(
            schema=schema,
            data_model=data_model,
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            temperature=temperature,
            use_inputs_schema=use_inputs_schema,
            use_outputs_schema=use_outputs_schema,
            reasoning_effort=reasoning_effort,
            use_chain_of_thought=use_chain_of_thought,
            tools=tools,
            autonomous=autonomous,
            return_inputs_with_trajectory=return_inputs_with_trajectory,
            max_iterations=max_iterations,
            streaming=streaming,
            workdir=workdir,
            skills=skills,
            name=name,
            description=description,
        )

    def _get_builtin_tools(self):
        return [
            Tool(fn)
            for fn in _build_tools(
                self.knowledge_base,
                search_type=self.search_type,
                k=self.k,
                similarity_threshold=self.similarity_threshold,
                fulltext_threshold=self.fulltext_threshold,
                output_format=self.output_format,
            )
        ]

    def _builtin_tool_kind(self):
        return "retrieval"

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "search_type": self.search_type,
                "k": self.k,
                "similarity_threshold": self.similarity_threshold,
                "fulltext_threshold": self.fulltext_threshold,
                "output_format": self.output_format,
                "knowledge_base": serialization_lib.serialize_synalinks_object(
                    self.knowledge_base,
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["knowledge_base"] = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        return super().from_config(config)
