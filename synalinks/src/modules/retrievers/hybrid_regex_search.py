# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_table_names
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class HybridRegexSearchInput(DataModel):
    """Input shape for `HybridRegexSearch`.

    The ``regex_patterns`` list is optional: when omitted, the
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
class HybridRegexSearch(Retriever):
    """Reciprocal-Rank-Fusion of vector similarity + regex matching.

    LM-driven wrapper around
    `KnowledgeBase.hybrid_regex_search`. The vector side's text
    comes from the input's ``similarity_search`` field; the regex
    side's patterns come from ``regex_patterns``. When
    ``regex_patterns`` is empty the adapter falls back to plain
    vector similarity.

    Vectors capture semantic similarity; regex captures exact textual
    shape. The two signals are orthogonal: give the same intent in
    both forms and the fused ranking surfaces rows that match on
    either axis. Regex uses RE2 (DuckDB's engine), so patterns are
    linear-time and not vulnerable to catastrophic backtracking.

    Single-table only: to retrieve from multiple tables, compose
    several `HybridRegexSearch` modules in the program DAG and
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
        embedding_model = synalinks.EmbeddingModel(model="ollama/mxbai-embed-large")
        kb = synalinks.KnowledgeBase(
            uri="duckdb://logs.db",
            data_models=[LogLine],
            embedding_model=embedding_model,
        )
        record = await synalinks.EmbedKnowledge(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(LogLine(id="l1", text="panic: error 42").to_json_data_model())
        await kb.update(record)
        lm = synalinks.LanguageModel(model="ollama/mistral")
        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.HybridRegexSearch(
            knowledge_base=kb,
            language_model=lm,
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
            is not given.
        table_name (str): Target table. Defaults to the schema's
            ``title``. **Optional**: when neither ``table_name`` nor a
            schema to derive it from is given, the language model infers
            the target table per call (constrained to the knowledge
            base's actual tables).
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

    query_model = HybridRegexSearchInput
    data_model_args = ("data_model",)

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        table_name: Optional[str] = None,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.table_name = resolve_target(schema, data_model, table_name)
        self.data_model = data_model
        self.k_rank = k_rank
        self.similarity_threshold = similarity_threshold
        self.ef_search = ef_search
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
        payload = query.get_json()
        queries = payload.get("similarity_search", [])
        patterns = payload.get("regex_patterns")
        # Fixed table, or the one the LM inferred this call.
        table_name = self.table_name or payload.get("table_name")
        # Need at least one signal (vector or regex) and a table to look up.
        if (not queries and not patterns) or not table_name:
            return None

        rows = await self.knowledge_base.hybrid_regex_search(
            text_or_texts=queries,
            pattern_or_patterns=patterns or None,
            table_name=table_name,
            k=self.k,
            k_rank=self.k_rank,
            similarity_threshold=self.similarity_threshold,
            ef_search=self.ef_search,
            fields=self.fields,
            case_sensitive=self.case_sensitive,
            output_format=self.output_format,
        )
        return {"result": rows}

    def get_config(self):
        return {
            "schema": self.schema,
            "table_name": self.table_name,
            "k": self.k,
            "k_rank": self.k_rank,
            "similarity_threshold": self.similarity_threshold,
            "ef_search": self.ef_search,
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
            "output_format": self.output_format,
            **super().get_config(),
        }
