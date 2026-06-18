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
from synalinks.src.modules.retrievers.infer_helpers import concat_infer_fields
from synalinks.src.modules.retrievers.infer_helpers import kb_table_names
from synalinks.src.saving import serialization_lib


class HybridFTSSearchInput(DataModel):
    """Input shape for `HybridFTSSearch`.

    The ``keywords`` list is optional — when omitted, the adapter
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
class HybridFTSSearch(Module):
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
        kb = synalinks.KnowledgeBase(
            uri="duckdb://docs.db",
            data_models=[Document],
        )
        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.HybridFTSSearch(
            knowledge_base=kb,
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
            ``title``. **Optional** — when neither ``table_name`` nor a
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
        fulltext_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        conjunctive: bool = False,
        bm25_b: Optional[float] = None,
        bm25_k: Optional[float] = None,
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
        self.schema = schema
        self.data_model = data_model

        # `table_name` is optional: when it (and a schema to infer it from) is
        # absent, the LM picks the target table per call (see query_generator).
        if table_name is None and schema is not None:
            table_name = schema.get("title") or None
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
        self.fulltext_threshold = fulltext_threshold
        self.ef_search = ef_search
        self.conjunctive = conjunctive
        self.bm25_b = bm25_b
        self.bm25_k = bm25_k

        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query

        if self.table_name is None:
            gen_target = {
                "schema": concat_infer_fields(
                    HybridFTSSearchInput.get_schema(),
                    [
                        (
                            "table_name",
                            "The knowledge-base table to search, chosen to best "
                            "answer the inputs.",
                            kb_table_names(self.knowledge_base),
                        )
                    ],
                )
            }
        else:
            gen_target = {"data_model": HybridFTSSearchInput}

        self.query_generator = Generator(
            **gen_target,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="hybrid_fts_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
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
            "fulltext_threshold": self.fulltext_threshold,
            "ef_search": self.ef_search,
            "conjunctive": self.conjunctive,
            "bm25_b": self.bm25_b,
            "bm25_k": self.bm25_k,
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
