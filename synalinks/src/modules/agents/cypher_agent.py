# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


def get_default_instructions(node_labels: List[str], rel_labels: List[str]) -> str:
    """Default instructions for the Cypher agent.

    Args:
        node_labels: PascalCase node labels available in the graph.
        rel_labels: PascalCase relation labels available in the graph.
            Both are embedded in the prompt so the LM doesn't have to
            call ``get_graph_schema`` first for trivial lookups.

    Returns:
        A prompt string giving the LM the tool-use plan and the
        read-only Cypher safety constraint.
    """
    return f"""
You are a graph analyst with read-only access to a knowledge graph.

Available node labels: {node_labels}
Available relation labels: {rel_labels}

Plan:
1. If you don't already know the schema, call `get_graph_schema` first.
2. When you need to inspect representative nodes, call `get_node_sample`.
3. Build a single `MATCH ... RETURN` Cypher query and execute it with
   `run_cypher_query`. Iterate on the query (read the error, fix the
   Cypher, retry) until you have the data.
4. Once you have an answer, stop calling tools and produce the final response.

Constraints:
- Only read-only Cypher is accepted. `CREATE`, `MERGE`, `SET`,
  `DELETE`, `DETACH DELETE`, `REMOVE`, `DROP`, `ALTER`, `COPY`,
  `INSTALL`, and `LOAD` are rejected by the engine — don't waste turns
  trying them.
- Node and relation labels are case-sensitive PascalCase (e.g.
  ``Person``, ``LivesIn``); property names are snake_case.
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


def _cap_rows(results, output_format: str, k: int):
    """Truncate a result set to at most ``k`` rows.

    Cypher has no general-purpose subquery-wrapping syntax equivalent
    to SQL's ``SELECT * FROM ({sql}) LIMIT k``, so the cap is enforced
    in Python after the engine returns. For ``"json"`` we slice the
    list; for ``"csv"`` we keep the header line plus the first ``k``
    data lines.
    """
    if output_format == "csv":
        if not results:
            return results
        lines = results.splitlines()
        if len(lines) <= 1:
            return results
        header, data = lines[0], lines[1:]
        capped = data[:k]
        return "\n".join([header, *capped])
    return results[:k]


def _build_tools(knowledge_base, output_format: str = "csv", k: int = 50):
    """Build the three Cypher tools bound to a knowledge base.

    Returns a list of plain async functions (not wrapped as ``Tool``
    objects yet) so the caller can decide how to register them.

    ``output_format`` is shared by ``get_node_sample`` and
    ``run_cypher_query`` and fixed per-agent at construction time:
    ``"csv"`` (default) compacts result sets so the LM spends fewer
    input tokens reading them; ``"json"`` returns a list of dicts.
    The ``get_graph_schema`` tool ignores the format — its output is
    always a small textual summary.

    ``k`` caps the page size the LM can pull through
    ``get_node_sample``: whatever ``limit`` the LM passes is clamped
    to ``min(limit, k)`` so a runaway request can't drain a large
    label into the conversation. ``run_cypher_query`` post-caps the
    engine's result set to ``k`` rows for the same reason.
    """

    async def get_graph_schema():
        """Get the complete graph schema: all node and relation labels.

        Returns one entry per node label (with its property names,
        types, and descriptions) and one entry per relation label
        (with its endpoint labels and edge properties). Use this first
        to discover what's available before writing queries.
        """
        node_models = knowledge_base.get_symbolic_entities()
        rel_models = knowledge_base.get_symbolic_relations()

        nodes_info = []
        for model in node_models:
            schema = model.get_schema()
            label = schema.get("title", "Unknown")
            properties = schema.get("properties", {})
            columns = []
            for col_name, col_info in properties.items():
                if col_name == "label":
                    continue
                col_type = col_info.get("type", "unknown")
                col_desc = col_info.get("description", "")
                columns.append(f"  - {col_name} ({col_type}): {col_desc}")
            nodes_info.append(f"Node: {label}\n" + "\n".join(columns))

        rels_info = []
        for model in rel_models:
            schema = model.get_schema()
            label = schema.get("title", "Unknown")
            properties = schema.get("properties", {})
            defs = schema.get("$defs", {})

            def _ref_label(spec):
                ref = spec.get("$ref", "") if isinstance(spec, dict) else ""
                key = ref.rsplit("/", 1)[-1] if ref else ""
                return defs.get(key, {}).get("title", key) or "?"

            subj_label = _ref_label(properties.get("subj", {}))
            obj_label = _ref_label(properties.get("obj", {}))

            attrs = []
            for col_name, col_info in properties.items():
                if col_name in ("label", "subj", "obj"):
                    continue
                col_type = col_info.get("type", "unknown")
                col_desc = col_info.get("description", "")
                attrs.append(f"  - {col_name} ({col_type}): {col_desc}")
            header = f"Relation: ({subj_label})-[:{label}]->({obj_label})"
            rels_info.append(header + ("\n" + "\n".join(attrs) if attrs else ""))

        return {
            "schema": "\n\n".join(nodes_info + rels_info),
            "node_count": len(node_models),
            "relation_count": len(rel_models),
        }

    async def get_node_sample(label: str, limit: int, offset: int):
        """Get a sample of nodes from a label to understand the data format.

        Args:
            label (str): The node label to sample (PascalCase).
            limit (int): Number of nodes to return (recommended: 3-5).
                Capped server-side at the agent's ``k`` setting.
            offset (int): Number of nodes to skip (use 0 to start at the top).
        """
        node_models = knowledge_base.get_symbolic_entities()
        available_labels = [m.get_schema().get("title", "Unknown") for m in node_models]
        if label not in available_labels:
            return {
                "error": (
                    f"Node label {label!r} not found. Available: {available_labels}"
                )
            }
        effective_limit = max(1, min(limit, k))
        # Label validated against the known-labels list, so direct
        # interpolation is safe; SKIP / LIMIT are bound parameters.
        query = f"MATCH (n:{label}) RETURN n SKIP $offset LIMIT $limit"
        try:
            results = await knowledge_base.cypher(
                query,
                params={"offset": offset, "limit": effective_limit},
                read_only=True,
                output_format=output_format,
            )
        except Exception as e:
            return {"error": str(e)}
        return {
            "label": label,
            "sample_data": results,
            "row_count": _row_count(results, output_format),
            "output_format": output_format,
            "limit": effective_limit,
            "offset": offset,
            "limit_capped": effective_limit < limit,
        }

    async def run_cypher_query(cypher_query: str):
        """Execute a read-only Cypher query and return the rows.

        Only read-only Cypher is accepted. The graph adapter rejects
        any query containing a write/admin keyword (``CREATE``,
        ``MERGE``, ``SET``, ``DELETE``, ``DETACH``, ``REMOVE``,
        ``DROP``, ``ALTER``, ``COPY``, ``INSTALL``, ``LOAD``) — this
        also blocks ``COPY ... FROM`` file ingestion through an
        otherwise-legitimate read query.

        Results are capped to the agent's ``k`` setting. Any ``LIMIT``
        clause inside the user query still applies first, so writing
        ``LIMIT 5`` returns 5 rows even when the cap is higher.

        Args:
            cypher_query (str): A read-only Cypher query to execute.
        """
        stripped = cypher_query.strip().rstrip(";").strip()
        if not stripped:
            return {
                "success": False,
                "error": "Empty Cypher query.",
                "query": cypher_query,
            }
        try:
            results = await knowledge_base.cypher(
                stripped, read_only=True, output_format=output_format
            )
        except Exception as e:
            return {"success": False, "error": str(e), "query": cypher_query}
        capped = _cap_rows(results, output_format, k)
        rc = _row_count(capped, output_format)
        full_rc = _row_count(results, output_format)
        return {
            "success": True,
            "query": cypher_query,
            "row_count": rc,
            "row_cap": k,
            "results": capped,
            "output_format": output_format,
            "may_have_more": full_rc > rc or rc >= k,
        }

    return [get_graph_schema, get_node_sample, run_cypher_query]


@synalinks_export(
    [
        "synalinks.modules.CypherAgent",
        "synalinks.CypherAgent",
    ]
)
class CypherAgent(Module):
    """A ready-to-use Cypher agent backed by a knowledge base.

    CypherAgent is a thin specialization of :class:`FunctionCallingAgent`
    that pre-wires three Cypher tools bound to a :class:`KnowledgeBase`
    (graph adapter):

    - ``get_graph_schema``: discovers all node and relation labels
      with their properties.
    - ``get_node_sample``: fetches a few nodes from a given label so
      the LM can see the data shape before writing queries.
    - ``run_cypher_query``: executes a read-only Cypher query via
      :meth:`KnowledgeBase.cypher` with ``read_only=True``.

    The constructor mirrors :class:`FunctionCallingAgent` — every
    parameter on that class is accepted here with identical semantics.
    The only additions are ``knowledge_base`` (required, must expose a
    graph adapter) and ``output_format`` (controls the Cypher tools'
    result rendering). User-supplied ``tools`` are appended to the
    three built-in tools.

    Safety is enforced by the knowledge base, not by string filtering.
    The Ladybug adapter scans the query (after stripping comments and
    string literals) for write/admin keywords and rejects anything
    that could mutate state or access files (``CREATE``, ``MERGE``,
    ``SET``, ``DELETE``, ``DETACH``, ``REMOVE``, ``DROP``, ``ALTER``,
    ``COPY``, ``INSTALL``, ``LOAD``).

    Example:

    ```python
    import synalinks
    import asyncio

    class Person(synalinks.Entity):
        name: str = synalinks.Field(description="Person name")

    class City(synalinks.Entity):
        name: str = synalinks.Field(description="City name")

    class LivesIn(synalinks.Relation):
        subj: Person
        obj: City

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="Natural language question")

    class CypherAnswer(synalinks.DataModel):
        answer: str = synalinks.Field(description="Answer in natural language")
        cypher_query: str = synalinks.Field(description="Cypher that produced it")

    async def main():
        kb = synalinks.KnowledgeBase(
            graph_uri="ladybug://my_graph.lb",
            entity_models=[Person, City],
            relation_models=[LivesIn],
            embedding_model=synalinks.EmbeddingModel(model="ollama/mxbai-embed-large"),
        )
        await kb.update_relations(
            LivesIn(subj=Person(name="Alice"), obj=City(name="Paris"))
        )

        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.CypherAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=CypherAnswer,
        )(inputs)
        agent = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await agent(Query(query="Who lives in Paris?"))
        print(result.get("answer"))
        print(result.get("cypher_query"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to query.
            Must have a graph adapter attached (i.e. constructed with
            ``graph_uri=...``). Required.
        k (int): Maximum page size (rows per call) the LM can pull
            through ``get_node_sample`` and ``run_cypher_query``.
            ``get_node_sample`` clamps the LM's ``limit`` argument
            to ``min(limit, k)``. ``run_cypher_query`` post-caps the
            engine result to ``k`` rows so even unbounded
            ``MATCH (n) RETURN n`` queries can't drain a large label
            into the conversation. A ``LIMIT`` inside the LM's own
            query still applies first. Defaults to 50.
        output_format (str): How the Cypher tools render result sets
            to the LM. ``"csv"`` (default) is compact and minimizes
            input tokens; ``"json"`` returns a list of dicts. Applies
            to both ``get_node_sample`` and ``run_cypher_query``.
        tools (list): Additional :class:`Tool` instances (or plain
            async functions) to expose alongside the three built-in
            Cypher tools — for example a calculator, a datetime
            helper, a web-search tool. Tool names must not collide
            with the built-ins (``get_graph_schema``, ``get_node_sample``,
            ``run_cypher_query``) or a ``ValueError`` is raised.
        schema (dict): JSON schema for the final answer.
        data_model (DataModel): DataModel for the final answer.
            Mutually exclusive with ``schema``.
        language_model (LanguageModel): The language model that drives
            the agent loop.
        prompt_template (str): Forwarded to the tool-call generator.
        examples (list): Few-shot examples for the tool-call generator.
        instructions (str): Override the default system instructions.
            When omitted, the default is built from the knowledge
            base's node and relation labels so the LM knows what's
            available without an extra schema call.
        final_instructions (str): Instructions for the final-answer
            generator. Defaults to ``instructions``.
        temperature (float): LM sampling temperature. Defaults to 0.0
            for deterministic Cypher generation.
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
            raise ValueError(
                "`knowledge_base` is required for CypherAgent: pass a "
                "KnowledgeBase with a graph adapter (graph_uri=...)."
            )
        self.knowledge_base = _get_kb(knowledge_base)
        # Fail fast if the KB has no graph adapter — the tools all
        # call graph methods, so a SQL-only KB would only error at
        # tool-invocation time inside the agent loop.
        self.knowledge_base._require_graph_adapter()
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
            node_labels = [
                m.get_schema().get("title", "Unknown")
                for m in self.knowledge_base.get_symbolic_entities()
            ]
            rel_labels = [
                m.get_schema().get("title", "Unknown")
                for m in self.knowledge_base.get_symbolic_relations()
            ]
            instructions = get_default_instructions(node_labels, rel_labels)
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

        self.extra_tools = list(tools) if tools else []
        merged_tools = list(builtin_tools)
        for extra in self.extra_tools:
            extra_tool = extra if isinstance(extra, Tool) else Tool(extra)
            if extra_tool.name in builtin_names:
                raise ValueError(
                    f"Tool name {extra_tool.name!r} collides with a built-in "
                    f"Cypher tool. Rename the additional tool."
                )
            merged_tools.append(extra_tool)

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
