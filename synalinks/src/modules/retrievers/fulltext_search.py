# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.retrievers.infer_helpers import kb_table_names
from synalinks.src.modules.retrievers.retriever import Retriever
from synalinks.src.modules.retrievers.retriever import resolve_target


class FullTextSearchInput(DataModel):
    """Input shape for `FullTextSearch`."""

    fulltext_search: List[str] = Field(
        description="Keyword queries for BM25 full-text search",
    )


@synalinks_export(
    [
        "synalinks.modules.FullTextSearch",
        "synalinks.FullTextSearch",
    ]
)
class FullTextSearch(Retriever):
    """BM25 full-text search against a single KB table.

    LM-driven wrapper around `KnowledgeBase.fulltext_search`. An
    embedded `Generator` turns the module's inputs into a
    `FullTextSearchInput` query (the ``fulltext_search`` field),
    which is then run against the table. Use this when you want a
    keyword-matching retrieval step without the vector branch (e.g.
    the corpus has no embeddings, or BM25 alone is the right
    relevance signal).

    Single-table only: to retrieve from multiple tables, compose
    several `FullTextSearch` modules in the program DAG and
    merge their outputs explicitly.

    Example:

    ```python
    import synalinks
    import asyncio

    class Document(synalinks.DataModel):
        id: str = synalinks.Field(description="Document id")
        text: str = synalinks.Field(description="Document text")

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="The user question")

    async def main():
        kb = synalinks.KnowledgeBase(
            uri="duckdb://docs.db",
            data_models=[Document],
        )
        await kb.update(
            Document(id="d1", text="On SIGTERM, drain connections").to_json_data_model()
        )
        lm = synalinks.LanguageModel(model="ollama/mistral")
        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.FullTextSearch(
            knowledge_base=kb,
            language_model=lm,
            data_model=Document,
            k=5,
        )(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)

        result = await program(Query(query="graceful shutdown on SIGTERM"))
        print(result.get("result"))

    asyncio.run(main())
    ```

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        language_model (LanguageModel): The language model used to
            generate the ``fulltext_search`` query from the inputs.
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
        threshold (float): Optional minimum BM25 score threshold.
            Higher score = better match; rows below ``threshold``
            are dropped by the adapter.
        conjunctive (bool): When ``True``, BM25 requires every term to
            match (AND-mode). Defaults to ``False`` (OR-mode).
        bm25_b (float): Optional override for BM25's ``b`` parameter
            (document-length normalization).
        bm25_k (float): Optional override for BM25's ``k1`` parameter
            (term-frequency saturation).
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
        max_tokens (int): Optional cap on generation length. Defaults
            to None (model default).
        top_p (float): Optional nucleus-sampling threshold. Defaults to
            None (model default).
        top_k (int): Optional top-k sampling cutoff. Defaults to None
            (model default).
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

    query_model = FullTextSearchInput
    data_model_args = ("data_model",)

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        table_name: Optional[str] = None,
        threshold: Optional[float] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        bm25_k: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.schema, self.table_name = resolve_target(schema, data_model, table_name)
        self.data_model = data_model
        self.threshold = threshold
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
        query_json = query.get_json()
        queries = query_json.get("fulltext_search", [])
        # Fixed table, or the one the LM inferred this call.
        table_name = self.table_name or query_json.get("table_name")
        if not queries or not table_name:
            return None

        rows = await self.knowledge_base.fulltext_search(
            queries,
            table_name=table_name,
            k=self.k,
            threshold=self.threshold,
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
            "threshold": self.threshold,
            "conjunctive": self.conjunctive,
            "bm25_b": self.bm25_b,
            "bm25_k": self.bm25_k,
            "output_format": self.output_format,
            **super().get_config(),
        }
