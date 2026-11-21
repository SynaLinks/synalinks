# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_table_names
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class HybridFTSSearchInput(DataModel):
    """Input shape for `HybridFTSSearch`.

    The ``keywords`` list is optional: when omitted, the adapter
    re-uses the vector side's text for BM25 scoring as well.
    """

    similarity_search: List[str] = Field(
        description="Natural-language queries for the vector branch",
    )
    keywords: Optional[List[str]] = Field(
        description="Optional keyword queries for the BM25 branch",
        default=None,
    )


@synalinks_export(
    [
        "synalinks.modules.HybridFTSSearch",
        "synalinks.HybridFTSSearch",
    ]
)
class HybridFTSSearch(Retriever):
    """Reciprocal-Rank-Fusion of vector similarity + BM25 fulltext.

    LM-driven wrapper around
    `KnowledgeBase.hybrid_fts_search`. The vector side's text
    comes from the input's ``similarity_search`` field; the BM25
    side's text comes from ``keywords`` if present, otherwise the
    adapter falls back to ``similarity_search``.

    Single-table only: to retrieve from multiple tables, compose
    several `HybridFTSSearch` modules in the program DAG and
    merge their outputs explicitly.

    Example:

    ```python
    import synalinks
    import asyncio

    class Document(synalinks.DataModel):
        id: str = synalinks.Field(description="Document id")
        text: str = synalinks.Field(description="Document text")

    class Query(synalinks.DataModel):
        similarity_search: list[str] = synalinks.Field(
            description="Natural-language queries",
        )
        keywords: list[str] | None = synalinks.Field(
            description="Optional keywords for BM25",
            default=None,
        )

    async def main():
        embedding_model = synalinks.EmbeddingModel(model="ollama/mxbai-embed-large")
        kb = synalinks.KnowledgeBase(
            uri="duckdb://docs.db",
            data_models=[Document],
            embedding_model=embedding_model,
        )
        record = await synalinks.EmbedKnowledge(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(Document(id="d1", text="On SIGTERM, drain connections").to_json_data_model())
        await kb.update(record)
        lm = synalinks.LanguageModel(model="ollama/mistral")
        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.HybridFTSSearch(
            knowledge_base=kb,
            language_model=lm,
            data_model=Document,
            k=5,
        )(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await program(Query(
            similarity_search=["graceful shutdown"],
            keywords=["SIGTERM", "drain"],
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
        fulltext_threshold (float): Optional BM25 score threshold for
            the fulltext branch.
        ef_search (int): HNSW search-time candidate-list depth
            (forwarded to the vector branch).
        conjunctive (bool): When ``True``, BM25 requires every term to
            match (AND-mode). Defaults to ``False`` (OR-mode).
        bm25_b (float): Optional override for BM25's ``b`` parameter
            (document-length normalization).
        bm25_k (float): Optional override for BM25's ``k1`` parameter
            (term-frequency saturation).
        output_format (str): How the underlying adapter renders rows.
            ``"json"`` (default) or ``"csv"``.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    query_model = HybridFTSSearchInput
    data_model_args = ("data_model",)

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        table_name: Optional[str] = None,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        bm25_k: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.table_name = resolve_target(schema, data_model, table_name)
        self.data_model = data_model
        self.k_rank = k_rank
        self.similarity_threshold = similarity_threshold
        self.fulltext_threshold = fulltext_threshold
        self.ef_search = ef_search
        self.conjunctive = conjunctive
        self.bm25_b = bm25_b
        self.bm25_k = bm25_k

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
        keywords = payload.get("keywords")
        # Fixed table, or the one the LM inferred this call.
        table_name = self.table_name or payload.get("table_name")
        if not queries or not table_name:
            return None

        rows = await self.knowledge_base.hybrid_fts_search(
            text_or_texts=queries,
            keywords=keywords,
            table_name=table_name,
            k=self.k,
            k_rank=self.k_rank,
            similarity_threshold=self.similarity_threshold,
            fulltext_threshold=self.fulltext_threshold,
            ef_search=self.ef_search,
            conjunctive=self.conjunctive,
            bm25_b=self.bm25_b,
            bm25_k=self.bm25_k,
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
            "fulltext_threshold": self.fulltext_threshold,
            "ef_search": self.ef_search,
            "conjunctive": self.conjunctive,
            "bm25_b": self.bm25_b,
            "bm25_k": self.bm25_k,
            "output_format": self.output_format,
            **super().get_config(),
        }
