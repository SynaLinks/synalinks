# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Dict
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


def default_text2cypher_instructions(k: Optional[int]) -> str:
    """Default instructions for the Text2Cypher generator.

    The graph schema itself is concatenated to the inputs at call time
    (in the ``graph_schema`` field), so the LM sees the entity / relation
    listing in the user message and the instructions don't need to
    enumerate labels. This keeps the instructions stable when the
    underlying graph gains or loses labels between calls.

    Args:
        k: Per-call row cap to mention in the LIMIT guidance, or
            ``None`` to drop the explicit suggestion.
    """
    limit_hint = (
        f" Add an explicit `LIMIT {k}` (or smaller) to bound the result set."
        if k is not None
        else ""
    )
    return f"""
Your task is to translate the user request into a single read-only Cypher
query against the labels described in the `graph_schema` field.

Constraints:
- Emit exactly one read query. `CREATE`, `MERGE`, `SET`, `DELETE`,
  `REMOVE`, `DROP`, `LOAD`, `CALL` to write procedures, `COPY`, and any
  schema mutation are rejected by the engine — don't waste turns trying
  them.
- Node and relation labels are case-sensitive and match exactly the
  spellings shown in the `graph_schema` field — do not invent labels
  or properties.
- Always end with a `RETURN` clause naming the columns you want.{limit_hint}
""".strip()


def _format_node_schema(schema: Dict[str, Any]) -> str:
    """Render a single entity (node) schema as a text block."""
    label = schema.get("title", "Unknown")
    properties = schema.get("properties", {})
    lines = [f"({label})"]
    for col_name, col_info in properties.items():
        if col_name == "label":
            # The `label` const is already encoded in the heading.
            continue
        col_type = col_info.get("type", col_info.get("const", "unknown"))
        col_desc = col_info.get("description", "")
        lines.append(f"  - {col_name} ({col_type}): {col_desc}")
    return "\n".join(lines)


def _ref_target(prop: Dict[str, Any]) -> Optional[str]:
    """Return the trailing component of a ``$ref`` if present."""
    ref = prop.get("$ref")
    if not ref:
        return None
    return ref.rsplit("/", 1)[-1]


def _format_relation_schema(schema: Dict[str, Any]) -> str:
    """Render a single relation (edge) schema as a text block.

    Uses Cypher ASCII-art ``(:Subj)-[:LABEL]->(:Obj)`` for the heading
    so the LM sees the relation in exactly the shape it must produce in
    the query.
    """
    label = schema.get("title", "Unknown")
    properties = schema.get("properties", {})
    subj_label = _ref_target(properties.get("subj", {})) or "?"
    obj_label = _ref_target(properties.get("obj", {})) or "?"

    lines = [f"(:{subj_label})-[:{label}]->(:{obj_label})"]
    for col_name, col_info in properties.items():
        if col_name in ("label", "subj", "obj"):
            continue
        col_type = col_info.get("type", col_info.get("const", "unknown"))
        col_desc = col_info.get("description", "")
        lines.append(f"  - {col_name} ({col_type}): {col_desc}")
    return "\n".join(lines)


def _format_graph_schema(knowledge_base) -> str:
    """Render the graph schema (entities + relations) as one text block.

    Mirrors the SQL-side ``_format_schema`` shape: a section per
    construct, separated by blank lines. Returns a plain string so it
    can be embedded as a single field on a ``GraphSchema`` data model.
    """
    sections: List[str] = []

    entities = knowledge_base.get_symbolic_entities()
    if entities:
        sections.append("# Entities")
        for entity in entities:
            sections.append(_format_node_schema(entity.get_schema()))

    relations = knowledge_base.get_symbolic_relations()
    if relations:
        sections.append("# Relations")
        for relation in relations:
            sections.append(_format_relation_schema(relation.get_schema()))

    return "\n\n".join(sections)


class GraphSchema(DataModel):
    """Schema description concatenated onto the inputs at call time."""

    graph_schema: str = Field(
        description="The available entity and relation labels with their properties",
    )


class CypherQuery(DataModel):
    """The structured output of the Text2Cypher generator."""

    cypher_query: str = Field(
        description="A single read-only Cypher query",
    )


class CypherQueryResult(DataModel):
    """The result of executing the generated Cypher query."""

    cypher_query: str = Field(
        description="The Cypher query that was executed",
    )
    result: List[Any] = Field(
        description="The rows returned by the query",
    )


@synalinks_export(
    [
        "synalinks.modules.Text2Cypher",
        "synalinks.Text2Cypher",
    ]
)
class Text2Cypher(Module):
    """Translate a natural-language request into Cypher and execute it.

    The module concatenates the knowledge base's graph schema (node
    labels, relation labels, endpoint pairings, properties) onto the
    input data model so the LM has the full schema in context,
    generates a single read-only Cypher query, and runs it through
    :meth:`KnowledgeBase.cypher` with ``read_only=True``.

    The schema is fetched on every call from
    :meth:`KnowledgeBase.get_symbolic_entities` and
    :meth:`KnowledgeBase.get_symbolic_relations`, so DDL changes
    (new labels, dropped labels, renamed properties) are picked up
    without rebuilding the module.

    Safety is enforced by the knowledge base, not by string filtering.
    The graph adapter's ``read_only=True`` path rejects any Cypher
    containing a write / admin keyword (``CREATE``, ``MERGE``, ``SET``,
    ``DELETE``, ``REMOVE``, ``DROP``, ``LOAD``, write-mode ``CALL``,
    ``COPY``).

    Example:

    ```python
    import synalinks
    import asyncio

    class Person(synalinks.Entity):
        name: str = synalinks.Field(description="Person name")

    class City(synalinks.Entity):
        name: str = synalinks.Field(description="City name")
        country: str = synalinks.Field(description="Country")

    class LivesIn(synalinks.Relation):
        subj: Person
        obj: City

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="Natural language question")

    async def main():
        kb = synalinks.KnowledgeBase(
            uri="ladybug://my_graph.db",
            entity_models=[Person, City],
            relation_models=[LivesIn],
        )

        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.Text2Cypher(
            knowledge_base=kb,
            language_model=lm,
        )(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await program(Query(query="Who lives in Paris?"))
        print(result.get("cypher_query"))
        print(result.get("result"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to query.
            Must be backed by a graph adapter. Required.
        language_model (LanguageModel): The language model that
            translates the request into Cypher.
        k (int | None): Per-call row cap mentioned in the default
            instructions to steer the LM toward adding ``LIMIT k`` in
            its query. The cap is not enforced post-hoc on the query
            string (Cypher subquery wrapping is fragile across
            engines); pass ``None`` to drop the hint entirely.
            Defaults to 50.
        output_format (str): How the executed query renders its rows.
            ``"json"`` (default) returns a list of dicts; ``"csv"``
            returns a CSV string.
        prompt_template (str): Forwarded to the underlying
            :class:`Generator`.
        examples (list): Few-shot examples for the generator.
        instructions (str): Override the default system instructions.
        seed_instructions (list): Optional seed instructions for
            prompt optimization.
        temperature (float): LM sampling temperature. Defaults to 0.0
            for deterministic Cypher generation.
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

        # Schema text is fetched at call time, not here — the graph
        # can gain labels between construction and use, and a Program
        # built once but called many times must not freeze a stale
        # schema. Default instructions are schema-agnostic for the
        # same reason; the label inventory ends up in the inputs.
        if instructions is None:
            instructions = default_text2cypher_instructions(self.k)
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
            data_model=CypherQuery,
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
        """Snapshot the graph schema as a data model for the current call.

        Re-fetched on every call so DDL changes (new labels, dropped
        labels, renamed properties) are picked up immediately without
        having to rebuild the module.
        """
        return JsonDataModel(
            json={"graph_schema": _format_graph_schema(self.knowledge_base)},
            schema=GraphSchema.get_schema(),
            name="graph_schema_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        # Concatenate the schema onto the inputs so the LM sees both
        # the user request and the entity / relation listing in a
        # single structured payload.
        inputs_with_schema = await ops.concat(
            inputs,
            self._schema_data_model(),
            name="inputs_with_schema_" + self.name,
        )

        cypher_query_model = await self.generator(inputs_with_schema, training=training)
        if not cypher_query_model:
            return None

        cypher_query = cypher_query_model.get_json().get("cypher_query", "").strip()
        if not cypher_query:
            result_rows: List[Any] = []
        else:
            try:
                result_rows = await self.knowledge_base.cypher(
                    cypher_query,
                    read_only=True,
                    output_format=self.output_format,
                )
            except Exception as e:
                result_rows = [{"error": str(e)}]

        output = JsonDataModel(
            json={"cypher_query": cypher_query, "result": result_rows},
            schema=CypherQueryResult.get_schema(),
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
            SymbolicDataModel(schema=GraphSchema.get_schema()),
            name="inputs_with_schema_" + self.name,
        )
        _ = await self.generator(inputs_with_schema, training=training)
        output = SymbolicDataModel(
            schema=CypherQueryResult.get_schema(),
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
