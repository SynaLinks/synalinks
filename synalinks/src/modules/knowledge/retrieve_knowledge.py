import json
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import get_args

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import GenericResult
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

# The canonical set of search types. The `Literal` is the single source
# of truth (used in the `__init__` annotation); `SEARCH_TYPES` is derived
# from it so the two can never drift.
SearchType = Literal[
    "similarity",
    "fulltext",
    "hybrid_fts",
    "regex",
    "hybrid_regex",
]
SEARCH_TYPES: List[str] = list(get_args(SearchType))

# Backwards-compatible aliases. Keys are accepted at the public surface
# and silently translated to the canonical value. `"hybrid"` used to mean
# "vector + BM25 fulltext"; it's now spelled `"hybrid_fts"` to pair
# symmetrically with `"hybrid_regex"`.
_SEARCH_TYPE_ALIASES: Dict[str, str] = {
    "hybrid": "hybrid_fts",
}


def default_retriever_instructions(tables, search_type="hybrid"):
    """The default instructions for the entity retriever.

    The body of the instructions tells the LM what kind of strings to
    put in each output field. The output *schema* also depends on the
    search type (see ``_search_query_schema_for``): the hybrid_regex
    variant adds a ``patterns`` field, so the prompt names it
    explicitly.
    """
    if search_type == "regex":
        guidance = (
            "The `search` field should be a list of regular-expression "
            "patterns (RE2 syntax) to match against the text fields of "
            "the chosen tables. Prefer anchors, character classes, and "
            "alternation over natural-language phrasing — the patterns "
            "are matched literally, not interpreted."
        )
    elif search_type == "hybrid_regex":
        guidance = (
            "Emit **both** a natural-language `search` list (for "
            "vector similarity over the chosen tables) AND a `patterns` "
            "list of regular-expression patterns (RE2 syntax) that "
            "capture the exact textual shape of what you are looking "
            "for (anchors, character classes, alternation). The two "
            "signals are merged by Reciprocal Rank Fusion, so it is OK "
            "for each list to err on the side of recall."
        )
    else:
        guidance = (
            "The `search` field should be a list of natural language "
            "search queries for the information to look for."
        )
    return f"""
Your task is to retrieve information among the following tables: {tables}.
First, decide step-by-step which tables you need, then use the `search` to
perform a lookup.
{guidance}
""".strip()


class KnowledgeBaseSchema(DataModel):
    knowledge_base_schema: List[Dict[str, Any]] = Field(
        description="The knowledge base schema",
    )


class SearchQuery(DataModel):
    """Output schema used by every search type except ``hybrid_regex``."""

    tables: List[str] = Field(description="The tables to lookup")
    search: List[str] = Field(description="The list of similarity search request")


class HybridRegexSearchQuery(DataModel):
    """Output schema for ``search_type="hybrid_regex"``.

    Adds a ``patterns`` field so the LM can supply the regex side of
    the hybrid lookup explicitly. Embedding a regex pattern for vector
    search makes no sense, and treating a natural-language sentence as
    a regex literally never matches — the two signals need separate
    inputs.
    """

    tables: List[str] = Field(description="The tables to lookup")
    search: List[str] = Field(
        description="Natural-language queries for vector similarity",
    )
    patterns: List[str] = Field(
        description="Regex patterns (RE2 syntax) for exact-shape matching",
    )


def _search_query_schema_for(search_type: str):
    """Pick the LM output schema that matches the configured search type."""
    if search_type == "hybrid_regex":
        return HybridRegexSearchQuery
    return SearchQuery


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
            - "hybrid_fts": Vector + BM25 fulltext, fused with RRF
              (default). Accepts the legacy alias ``"hybrid"``.
            - "regex": RE2 pattern matching against the string fields of
              each table. The LM is instructed to produce regex patterns
              instead of natural-language queries.
            - "hybrid_regex": Vector + regex, fused with RRF. The LM
              emits both natural-language queries (vector side) and
              regex patterns (regex side); the two signals are merged
              with Reciprocal Rank Fusion. The output schema becomes
              ``HybridRegexSearchQuery`` (adds a ``patterns`` field).
        k (int): Maximum number of results to return. Defaults to 10.
        similarity_threshold (float): Maximum distance threshold for similarity
            search (lower = better match). Only used when search_type is
            "similarity" or "hybrid".
        fulltext_threshold (float): Minimum BM25 score threshold for fulltext
            search (higher = better match). Only used when search_type is
            "fulltext" or "hybrid".
        k_rank (int): RRF smoothing constant for hybrid search. Lower values
            emphasize top ranks more strongly. Defaults to 60.
        fields (list): Field names to match against in regex search.
            Defaults to every string field on the schema. Only used when
            ``search_type="regex"``.
        case_sensitive (bool): When ``False``, regex matches are
            case-insensitive. Only used when ``search_type="regex"``.
            Defaults to ``True``.
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
        *,
        knowledge_base=None,
        language_model=None,
        data_models=None,
        search_type: SearchType = "hybrid_fts",
        k=10,
        similarity_threshold=None,
        fulltext_threshold=None,
        k_rank=60,
        fields: Optional[List[str]] = None,
        case_sensitive: bool = True,
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
        self.language_model = _get_lm(language_model)

        # Translate legacy aliases (e.g. "hybrid" -> "hybrid_fts") before
        # validating against the canonical set, so users on the old name
        # keep working without a code change.
        search_type = _SEARCH_TYPE_ALIASES.get(search_type, search_type)
        if search_type not in SEARCH_TYPES:
            raise ValueError(
                f"`search_type` must be one of {SEARCH_TYPES}, got '{search_type}'"
            )
        self.search_type = search_type

        self.k = k
        self.similarity_threshold = similarity_threshold
        self.fulltext_threshold = fulltext_threshold
        self.k_rank = k_rank
        self.fields = fields
        self.case_sensitive = case_sensitive

        self.prompt_template = prompt_template
        self.examples = examples

        if not data_models:
            data_models = knowledge_base.get_symbolic_data_models()

        self.data_models = data_models

        tables = [data_model.get_schema().get("title") for data_model in self.data_models]

        if not instructions:
            instructions = default_retriever_instructions(
                tables, search_type=self.search_type
            )

        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_query = return_query

        # The LM output schema is chosen by search_type — the hybrid_regex
        # mode needs an extra `patterns` field, every other mode reuses
        # the legacy `SearchQuery` shape.
        self.search_query_generator = Generator(
            data_model=_search_query_schema_for(self.search_type),
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

    async def _perform_search(
        self, search_terms, patterns, target_data_models
    ):
        """Perform the search across one or more tables.

        The adapter's search methods are single-table; this layer
        iterates over the LM-selected tables and merges the per-table
        result sets. Dedup uses a sorted-items signature because row
        dicts may contain unhashable values (lists, dicts), and we
        don't know which field is the primary key here.

        Per-table k is set to ``self.k`` and the merged top-k is
        taken at the end — when ``score`` is present (similarity /
        fulltext / hybrid), the merge sorts by descending score
        (lower is better for plain similarity, so we negate); for
        score-less results (regex), insertion order is preserved.

        Args:
            search_terms: List of search query strings.
            patterns: List of regex patterns from the LM. Only used
                by the ``hybrid_regex`` mode.
            target_data_models: List of data models to search.

        Returns:
            Merged top-k list of result dicts.
        """
        aggregated: List[Dict[str, Any]] = []
        seen: set = set()

        async def _search_one(table_name):
            if self.search_type == "similarity":
                return await self.knowledge_base.similarity_search(
                    search_terms,
                    table_name=table_name,
                    k=self.k,
                    threshold=self.similarity_threshold,
                )
            if self.search_type == "fulltext":
                return await self.knowledge_base.fulltext_search(
                    search_terms,
                    table_name=table_name,
                    k=self.k,
                    threshold=self.fulltext_threshold,
                )
            if self.search_type == "regex":
                # regex_search is single-pattern; iterate the LM's
                # patterns and merge their rows per table.
                rows_per_table: List[Dict[str, Any]] = []
                seen_local: set = set()
                for pattern in search_terms:
                    rows = await self.knowledge_base.regex_search(
                        pattern,
                        table_name=table_name,
                        fields=self.fields,
                        case_sensitive=self.case_sensitive,
                        k=self.k,
                    )
                    for row in rows:
                        sig = json.dumps(row, sort_keys=True, default=str)
                        if sig not in seen_local:
                            seen_local.add(sig)
                            rows_per_table.append(row)
                return rows_per_table
            if self.search_type == "hybrid_regex":
                return await self.knowledge_base.hybrid_regex_search(
                    search_terms,
                    pattern_or_patterns=patterns or None,
                    table_name=table_name,
                    k=self.k,
                    k_rank=self.k_rank,
                    similarity_threshold=self.similarity_threshold,
                    fields=self.fields,
                    case_sensitive=self.case_sensitive,
                )
            # hybrid_fts (default)
            return await self.knowledge_base.hybrid_fts_search(
                search_terms,
                table_name=table_name,
                k=self.k,
                k_rank=self.k_rank,
                similarity_threshold=self.similarity_threshold,
                fulltext_threshold=self.fulltext_threshold,
            )

        for dm in target_data_models:
            table_name = dm.get_schema().get("title")
            try:
                rows = await _search_one(table_name)
            except Exception:
                rows = []
            for row in rows:
                sig = json.dumps(row, sort_keys=True, default=str)
                if sig in seen:
                    continue
                seen.add(sig)
                aggregated.append(row)

        # Sort: similarity → ascending score; fulltext / hybrid →
        # descending; regex → insertion order (no score field).
        if self.search_type == "similarity":
            aggregated.sort(key=lambda r: r.get("score", float("inf")))
        elif self.search_type in ("fulltext", "hybrid_fts", "hybrid_regex"):
            aggregated.sort(
                key=lambda r: r.get("score", float("-inf")), reverse=True
            )
        return aggregated[: self.k]

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        # Generate search query using the language model
        search_query = await self.search_query_generator(inputs, training=training)

        if not search_query:
            return None

        # Get the tables, search terms, and (for hybrid_regex) patterns
        # from the generated query.
        query_json = search_query.get_json()
        tables = query_json.get("tables", [])
        search_terms = query_json.get("search", [])
        # `patterns` is only present when the LM ran against the
        # HybridRegexSearchQuery schema; harmlessly empty otherwise.
        patterns = query_json.get("patterns", [])

        # Hybrid_regex needs at least one of (search_terms, patterns);
        # every other mode needs search_terms. Otherwise we have nothing
        # to look up.
        if self.search_type == "hybrid_regex":
            if not search_terms and not patterns:
                return None
        elif not search_terms:
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
        search_results = await self._perform_search(
            search_terms, patterns, target_data_models
        )

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
            "fields": list(self.fields) if self.fields is not None else None,
            "case_sensitive": self.case_sensitive,
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
