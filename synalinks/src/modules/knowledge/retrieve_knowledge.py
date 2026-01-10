from typing import Any
from typing import Dict
from typing import List
from typing import Literal

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import GenericResult
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

SEARCH_TYPES = ["similarity", "fulltext", "hybrid"]


def default_retriever_instructions(tables):
    """The default instructions for the entity retriever"""
    return f"""
Your task is to retrieve information among the following tables: {tables}.
First, decide step-by-step which tables you need, then use the `search` to
perform a lookup.
The `search` field should be a list of natural language search queries for the
information to look for.
""".strip()


class KnowledgeBaseSchema(DataModel):
    knowledge_base_schema: List[Dict[str, Any]] = Field(
        description="The knowledge base schema",
    )


class SearchQuery(DataModel):
    tables: List[str] = Field(description="The tables to lookup")
    search: List[str] = Field(description="The list of similarity search request")


@synalinks_export(
    [
        "synalinks.modules.RetrieveKnowledge",
        "synalinks.RetrieveKnowledge",
    ]
)
class RetrieveKnowledge(Module):
    """Module for retrieving knowledge from a knowledge base.

    This module uses a language model to generate search queries and retrieves
    relevant information from a knowledge base using configurable search methods.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
        language_model (LanguageModel): The language model used to generate
            search queries.
        data_models (list): List of data models to search. Defaults to all
            models in the knowledge base.
        search_type (str): The type of search to perform. One of:
            - "similarity": Vector-based semantic search using embeddings.
            - "fulltext": BM25-based full-text search.
            - "hybrid": Combines both using Reciprocal Rank Fusion (default).
        k (int): Maximum number of results to return. Defaults to 10.
        similarity_threshold (float): Maximum distance threshold for similarity
            search (lower = better match). Only used when search_type is
            "similarity" or "hybrid".
        fulltext_threshold (float): Minimum BM25 score threshold for fulltext
            search (higher = better match). Only used when search_type is
            "fulltext" or "hybrid".
        k_rank (int): RRF smoothing constant for hybrid search. Lower values
            emphasize top ranks more strongly. Defaults to 60.
        prompt_template (str): Custom prompt template for the search query
            generator.
        examples (list): Example inputs/outputs for few-shot learning.
        instructions (str): Custom instructions for the search query generator.
        seed_instructions (str): Seed instructions for variability.
        temperature (float): Temperature for the language model. Defaults to 0.0.
        use_inputs_schema (bool): Whether to include input schema in the prompt.
        use_outputs_schema (bool): Whether to include output schema in the prompt.
        return_inputs (bool): Whether to include original inputs in the output.
        return_query (bool): Whether to include the generated search query in
            the output.
        name (str): Name of the module.
        description (str): Description of the module.
        trainable (bool): Whether the module is trainable.
    """

    def __init__(
        self,
        knowledge_base=None,
        language_model=None,
        data_models=None,
        search_type: Literal["similarity", "fulltext", "hybrid"] = "hybrid",
        k=10,
        similarity_threshold=None,
        fulltext_threshold=None,
        k_rank=60,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=0.0,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs=True,
        return_query=True,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.knowledge_base = knowledge_base
        self.language_model = language_model

        if search_type not in SEARCH_TYPES:
            raise ValueError(
                f"`search_type` must be one of {SEARCH_TYPES}, got '{search_type}'"
            )
        self.search_type = search_type

        self.k = k
        self.similarity_threshold = similarity_threshold
        self.fulltext_threshold = fulltext_threshold
        self.k_rank = k_rank

        self.prompt_template = prompt_template
        self.examples = examples

        if not data_models:
            data_models = knowledge_base.get_symbolic_data_models()

        self.data_models = data_models

        tables = [data_model.get_schema().get("title") for data_model in self.data_models]

        if not instructions:
            instructions = default_retriever_instructions(tables)

        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query

        self.search_query_generator = Generator(
            data_model=SearchQuery,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
            name="search_query_generator_" + self.name,
        )

    async def _perform_search(self, search_terms, target_data_models):
        """Perform the search based on the configured search type.

        Args:
            search_terms: List of search query strings.
            target_data_models: List of data models to search.

        Returns:
            List of search results.
        """
        if self.search_type == "similarity":
            return await self.knowledge_base.similarity_search(
                search_terms,
                data_models=target_data_models,
                k=self.k,
                threshold=self.similarity_threshold,
            )
        elif self.search_type == "fulltext":
            return await self.knowledge_base.fulltext_search(
                search_terms,
                data_models=target_data_models,
                k=self.k,
                threshold=self.fulltext_threshold,
            )
        else:  # hybrid
            return await self.knowledge_base.hybrid_search(
                search_terms,
                data_models=target_data_models,
                k=self.k,
                k_rank=self.k_rank,
                similarity_threshold=self.similarity_threshold,
                fulltext_threshold=self.fulltext_threshold,
            )

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        # Generate search query using the language model
        search_query = await self.search_query_generator(inputs, training=training)

        if not search_query:
            return None

        # Get the tables and search terms from the generated query
        tables = search_query.get_json().get("tables", [])
        search_terms = search_query.get_json().get("search", [])

        if not search_terms:
            return None

        # Filter data models to only those requested
        target_data_models = []
        for dm in self.data_models:
            schema = dm.get_schema()
            if schema.get("title") in tables:
                target_data_models.append(dm)

        if not target_data_models:
            target_data_models = self.data_models

        # Perform search based on configured search type
        search_results = await self._perform_search(search_terms, target_data_models)

        results = JsonDataModel(
            json={"result": search_results},
            schema=GenericResult.get_schema(),
            name="retrieval_results_" + self.name,
        )
        if self.return_query:
            results = await ops.logical_and(
                search_query,
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
        search_query = await self.search_query_generator(inputs, training=training)
        results = SymbolicDataModel(
            schema=GenericResult.get_schema(),
            name="retrieval_results_" + self.name,
        )
        if self.return_query:
            results = await ops.logical_and(
                search_query,
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
            "search_type": self.search_type,
            "k": self.k,
            "similarity_threshold": self.similarity_threshold,
            "fulltext_threshold": self.fulltext_threshold,
            "k_rank": self.k_rank,
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
        data_models_config = {
            "data_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        data_model.to_symbolic_data_model(
                            name="data_models" + (f"_{i}_" if i > 0 else "_") + self.name
                        )
                    )
                    if not is_symbolic_data_model(data_model)
                    else serialization_lib.serialize_synalinks_object(data_model)
                )
                for i, data_model in enumerate(self.data_models)
            ]
        }
        return {
            **config,
            **knowledge_base_config,
            **language_model_config,
            **data_models_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base"),
        )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        data_models_config = config.pop("data_models")
        data_models = [
            serialization_lib.deserialize_synalinks_object(data_model)
            for data_model in data_models_config
        ]
        return cls(
            knowledge_base=knowledge_base,
            data_models=data_models,
            language_model=language_model,
            **config,
        )
