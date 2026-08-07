# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
from typing import List
from typing import Optional

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import KnowledgeGraph
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.modules.retrievers.global_graph_search import GlobalGraphSearch
from synalinks.src.saving import serialization_lib


class CommunityAnswer(DataModel):
    """The *map* step's output: one scored partial answer per community."""

    partial_answer: str = Field(
        description=(
            "The partial answer to the inputs, using ONLY the provided "
            "community subgraph (its entities and relations)"
        ),
    )
    relevance_score: float = Field(
        description=(
            "How relevant this community is to the inputs, between 0 and "
            "100 (0 = the community contains nothing useful)"
        ),
    )


class CommunityAnswers(DataModel):
    """The *reduce* step's input: every kept partial answer, best first."""

    community_answers: List[CommunityAnswer] = Field(
        description="The community partial answers, most relevant first",
    )


@synalinks_export(
    [
        "synalinks.modules.GlobalGraphMapReduce",
        "synalinks.GlobalGraphMapReduce",
    ]
)
class GlobalGraphMapReduce(Module):
    """GraphRAG-style *global* answering: map over communities, then reduce.

    The LM orchestration on top of `GlobalGraphSearch`. The embedded
    retriever returns the graph's ``k`` most important communities as
    subgraphs; the **map** generator then answers the inputs against each
    community independently (in parallel), producing a partial answer and
    a 0-100 ``relevance_score``; finally the **reduce** generator combines
    the kept partial answers (best first) into the final answer. This is
    the query-time half of GraphRAG global search: the whole-graph
    question answering (*"what are the main themes across the corpus"*)
    that no single-neighbourhood retrieval can support.

    **Requires** `KnowledgeBase.build_communities` to have run at index
    time; with no communities the module returns ``None``. The
    entity-centric counterpart is `LocalGraphSearch` (seed + expand); the
    retrieval-only counterpart is `GlobalGraphSearch` (subgraphs, no LM).

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        language_model (LanguageModel): The language model used by the map
            and reduce generators. Required.
        schema (dict): The target JSON schema of the final (reduce) answer.
            If not provided use the `data_model` to infer it. When neither
            is given, the reduce step returns a `ChatMessage` (like
            `Generator`).
        data_model (DataModel | SymbolicDataModel): The final answer's
            data model, when ``schema`` is not given.
        node_labels (list): Optional whitelist of node labels to include
            (``None`` = every stamped table). See `GlobalGraphSearch`.
        rel_labels (list): Optional whitelist of relation labels to
            include (``None`` = all). See `GlobalGraphSearch`.
        k (int): Maximum number of communities to map over. Defaults
            to 10.
        members_per_community (int): Optional cap on member entities per
            community, best first by rank. See `GlobalGraphSearch`.
        score_threshold (float): Partial answers whose
            ``relevance_score`` is strictly below this are dropped before
            the reduce step. Defaults to 0.0 (keep everything); set e.g.
            1.0 to discard zero-scored communities like the original
            GraphRAG.
        map_instructions (str): Optional instructions for the map
            generator (answer against one community + score it).
        reduce_instructions (str): Optional instructions for the reduce
            generator (combine the partial answers).
        temperature (float): Optional. The temperature for the LM calls.
        max_tokens (int): Optional. Maximum number of tokens to generate.
        top_p (float): Optional. The nucleus sampling probability.
        top_k (int): Optional. The top-k sampling cutoff.
        use_inputs_schema (bool): Whether to use the inputs schema in the
            prompts (Default to False) (see `Generator`).
        use_outputs_schema (bool): Whether to use the outputs schema in
            the prompts (Default to False) (see `Generator`).
        return_inputs (bool): Whether to concatenate the inputs onto the
            output. Defaults to False.
        return_community_answers (bool): Whether to concatenate the kept
            partial answers onto the output. Defaults to False.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    DEFAULT_MAP_INSTRUCTIONS = (
        "Answer the inputs using ONLY the provided community subgraph "
        "(its entities and relations). Then rate the community's relevance "
        "to the inputs with `relevance_score` between 0 and 100 "
        "(0 = nothing in this community helps)."
    )
    DEFAULT_REDUCE_INSTRUCTIONS = (
        "Combine the community partial answers into one comprehensive "
        "answer to the inputs. Weigh each partial answer by its relevance "
        "score and ignore the irrelevant ones."
    )

    def __init__(
        self,
        *,
        knowledge_base=None,
        language_model=None,
        schema=None,
        data_model=None,
        node_labels: Optional[List[str]] = None,
        rel_labels: Optional[List[str]] = None,
        k: int = 10,
        members_per_community: Optional[int] = None,
        score_threshold: float = 0.0,
        map_instructions: Optional[str] = None,
        reduce_instructions: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        use_inputs_schema: bool = False,
        use_outputs_schema: bool = False,
        return_inputs: bool = False,
        return_community_answers: bool = False,
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

        if not schema and data_model:
            schema = data_model.get_schema()
        # `schema` may stay None: the reduce generator then returns a
        # ChatMessage, exactly like a bare `Generator`.
        self.schema = schema

        self.node_labels = node_labels
        self.rel_labels = rel_labels
        self.k = k
        self.members_per_community = members_per_community

        if score_threshold < 0:
            raise ValueError(f"`score_threshold` must be >= 0, got {score_threshold!r}")
        self.score_threshold = score_threshold

        self.map_instructions = map_instructions
        self.reduce_instructions = reduce_instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.return_community_answers = return_community_answers

        # Retrieval-only submodule: the communities to map over. The inputs
        # are re-concatenated per community here, so it must not carry them.
        self.global_graph_search = GlobalGraphSearch(
            knowledge_base=self.knowledge_base,
            node_labels=self.node_labels,
            rel_labels=self.rel_labels,
            k=self.k,
            members_per_community=self.members_per_community,
            return_inputs=False,
            name="global_graph_search_" + self.name,
        )
        common_lm_kwargs = dict(
            language_model=self.language_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=False,
        )
        self.map_generator = Generator(
            data_model=CommunityAnswer,
            instructions=self.map_instructions or self.DEFAULT_MAP_INSTRUCTIONS,
            name="map_generator_" + self.name,
            **common_lm_kwargs,
        )
        self.reduce_generator = Generator(
            schema=self.schema,
            instructions=self.reduce_instructions or self.DEFAULT_REDUCE_INSTRUCTIONS,
            name="reduce_generator_" + self.name,
            **common_lm_kwargs,
        )

    async def _map_one(self, inputs, community_json, training=False):
        """Answer the inputs against one community subgraph."""
        community = JsonDataModel(
            json=community_json,
            schema=KnowledgeGraph.get_schema(),
            name="community_" + self.name,
        )
        map_inputs = await ops.logical_and(
            inputs,
            community,
            name="map_inputs_" + self.name,
        )
        answer = await self.map_generator(map_inputs, training=training)
        return answer.get_json() if answer else None

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        communities = await self.global_graph_search(inputs, training=training)
        if not communities:
            return None
        community_jsons = communities.get_json().get("knowledge_graphs", [])
        if not community_jsons:
            return None

        # Map: one scored partial answer per community, in parallel.
        answers = await asyncio.gather(
            *[
                self._map_one(inputs, community_json, training=training)
                for community_json in community_jsons
            ]
        )
        kept = [
            a
            for a in answers
            if a is not None and (a.get("relevance_score") or 0.0) >= self.score_threshold
        ]
        if not kept:
            return None
        kept.sort(key=lambda a: a.get("relevance_score") or 0.0, reverse=True)

        # Reduce: combine the kept partial answers, best first.
        community_answers = JsonDataModel(
            json={"community_answers": kept},
            schema=CommunityAnswers.get_schema(),
            name="community_answers_" + self.name,
        )
        reduce_inputs = await ops.logical_and(
            inputs,
            community_answers,
            name="reduce_inputs_" + self.name,
        )
        results = await self.reduce_generator(reduce_inputs, training=training)
        if not results:
            return None
        if self.return_community_answers:
            results = await ops.logical_and(
                community_answers,
                results,
                name="results_with_community_answers_" + self.name,
            )
        if self.return_inputs:
            results = await ops.logical_and(
                inputs,
                results,
                name="results_with_inputs_" + self.name,
            )
        return results

    async def compute_output_spec(self, inputs, training=False):
        # Trace every submodule so their variables are built and tracked.
        await self.global_graph_search(inputs, training=training)
        map_inputs = await ops.logical_and(
            inputs,
            SymbolicDataModel(
                schema=KnowledgeGraph.get_schema(),
                name="community_" + self.name,
            ),
            name="map_inputs_" + self.name,
        )
        await self.map_generator(map_inputs, training=training)

        community_answers = SymbolicDataModel(
            schema=CommunityAnswers.get_schema(),
            name="community_answers_" + self.name,
        )
        reduce_inputs = await ops.logical_and(
            inputs,
            community_answers,
            name="reduce_inputs_" + self.name,
        )
        results = await self.reduce_generator(reduce_inputs, training=training)
        if self.return_community_answers:
            results = await ops.logical_and(
                community_answers,
                results,
                name="results_with_community_answers_" + self.name,
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
            "node_labels": self.node_labels,
            "rel_labels": self.rel_labels,
            "k": self.k,
            "members_per_community": self.members_per_community,
            "score_threshold": self.score_threshold,
            "map_instructions": self.map_instructions,
            "reduce_instructions": self.reduce_instructions,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs": self.return_inputs,
            "return_community_answers": self.return_community_answers,
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
