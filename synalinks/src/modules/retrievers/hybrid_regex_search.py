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


class HybridRegexSearchInput(DataModel):
    """Input shape for :class:`HybridRegexSearch`.

    The ``regex_patterns`` list is optional — when omitted, the
    adapter falls back to plain vector similarity over
    ``similarity_search``.
    """

    similarity_search: List[str] = Field(
        description="Natural-language queries for the vector branch",
    )
    regex_patterns: Optional[List[str]] = Field(
        description="Regex patterns (RE2 syntax) for the regex branch",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.modules.HybridRegexSearch",
        "synalinks.HybridRegexSearch",
    ]
)
class HybridRegexSearch(Module):
    """Reciprocal-Rank-Fusion of vector similarity + regex matching.

    LM-driven wrapper around
    :meth:`KnowledgeBase.hybrid_regex_search`. The vector side's text
    comes from the input's ``similarity_search`` field; the regex
    side's patterns come from ``regex_patterns``. When
    ``regex_patterns`` is empty the adapter falls back to plain
    vector similarity.

    Vectors capture semantic similarity; regex captures exact textual
    shape. The two signals are orthogonal — give the same intent in
    both forms and the fused ranking surfaces rows that match on
    either axis. Regex uses RE2 (DuckDB's engine), so patterns are
    linear-time and not vulnerable to catastrophic backtracking.

    Single-table only: to retrieve from multiple tables, compose
    several :class:`HybridRegexSearch` modules in the program DAG and
    merge their outputs explicitly.

    Example:

    ```python
    import synalinks
    import asyncio

    class LogLine(synalinks.DataModel):
        id: str = synalinks.Field(description="Log id")
        text: str = synalinks.Field(description="Log line")

    class Query(synalinks.DataModel):
        similarity_search: list[str] = synalinks.Field(
            description="Natural-language queries",
        )
        regex_patterns: list[str] | None = synalinks.Field(
            description="Regex patterns",
            default=None,
        )

    async def main():
        kb = synalinks.KnowledgeBase(
            uri="duckdb://logs.db",
            data_models=[LogLine],
        )
        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.HybridRegexSearch(
            knowledge_base=kb,
            data_model=LogLine,
            k=5,
        )(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await program(Query(
            similarity_search=["server crash"],
            regex_patterns=[r"error \\d+", r"panic:"],
        ))
        print(result.get("result"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
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
        k (int): Maximum number of results. Defaults to 10.
        k_rank (int): RRF smoothing constant. Lower values emphasize
            top ranks more strongly. Defaults to 60.
        similarity_threshold (float): Optional vector-distance
            threshold for the vector branch.
        ef_search (int): HNSW search-time candidate-list depth
            (forwarded to the vector branch).
        fields (list): Field names to match against in the regex
            branch. Defaults to every string field on the schema.
        case_sensitive (bool): When ``False``, regex matches are
            case-insensitive. Defaults to ``True``.
        output_format (str): How the underlying adapter renders rows.
            ``"json"`` (default) or ``"csv"``.
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
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
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
        self.k_rank = k_rank
        self.similarity_threshold = similarity_threshold
        self.ef_search = ef_search
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
            data_model=HybridRegexSearchInput,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="hybrid_regex_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
        payload = query.get_json()
        queries = payload.get("similarity_search", [])
        patterns = payload.get("regex_patterns")
        # Need at least one signal — vector or regex — to look up.
        if not queries and not patterns:
            return None

        rows = await self.knowledge_base.hybrid_regex_search(
            text_or_texts=queries,
            pattern_or_patterns=patterns or None,
            table_name=self.table_name,
            k=self.k,
            k_rank=self.k_rank,
            similarity_threshold=self.similarity_threshold,
            ef_search=self.ef_search,
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
            "schema": self.schema,
            "table_name": self.table_name,
            "k": self.k,
            "k_rank": self.k_rank,
            "similarity_threshold": self.similarity_threshold,
            "ef_search": self.ef_search,
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
