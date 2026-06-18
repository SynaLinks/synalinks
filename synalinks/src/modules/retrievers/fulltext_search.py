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
class FullTextSearch(Module):
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
            ``title``. **Optional** — when neither ``table_name`` nor a
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
        temperature (float): Temperature for the language model.
            Defaults to 0.0.
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

    def __init__(
        self,
        *,
        knowledge_base=None,
        language_model=None,
        schema=None,
        data_model=None,
        table_name: Optional[str] = None,
        k: int = 10,
        threshold: Optional[float] = None,
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
        self.threshold = threshold
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
                    FullTextSearchInput.get_schema(),
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
            gen_target = {"data_model": FullTextSearchInput}

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
            name="fulltext_search_query_generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        query = await self.query_generator(inputs, training=training)
        if not query:
            return None
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
            "threshold": self.threshold,
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
