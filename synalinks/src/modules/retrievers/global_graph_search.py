# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import KnowledgeGraphs
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


@synalinks_export(
    [
        "synalinks.modules.GlobalGraphSearch",
        "synalinks.GlobalGraphSearch",
    ]
)
class GlobalGraphSearch(Module):
    """GraphRAG-style *global* search: return the graph's top communities.

    Theme-centric graph retrieval. Reads the communities
    `KnowledgeBase.build_communities` materialized at index time and
    returns the ``k`` most important ones, each rebuilt as a
    `KnowledgeGraph` (its member entities plus their internal relations),
    ordered by aggregate PageRank. These are the subgraphs that answer
    *"what are the overall patterns across the **whole** graph"*, and the
    units an LM map-reduce step summarises then combines.

    Unlike the other retrievers this module runs no language model and
    ignores the *content* of its inputs: global search is whole-graph, not
    query-seeded; relevance to a specific question is decided downstream
    (in the map-reduce), not here. Inputs still flow through so the module
    slots into a program DAG; ``return_inputs`` carries them to the output.
    The entity-centric counterpart is `LocalGraphSearch`.

    **Requires** `build_communities` to have run on the knowledge base
    first; otherwise an empty `KnowledgeGraphs` is returned.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to search.
            Required.
        node_labels (list): Optional whitelist of node labels to include
            (``None`` = every stamped table).
        rel_labels (list): Optional whitelist of relation labels to
            include (``None`` = all).
        k (int): Maximum number of communities (subgraphs) to return.
            Defaults to 10.
        members_per_community (int): Optional cap on member entities per
            community, best first by rank. ``None`` keeps the whole
            community.
        return_inputs (bool): Whether to concatenate the inputs onto the
            output. Defaults to True.
        name (str): Module name.
        description (str): Module description.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    def __init__(
        self,
        *,
        knowledge_base=None,
        node_labels: Optional[List[str]] = None,
        rel_labels: Optional[List[str]] = None,
        k: int = 10,
        members_per_community: Optional[int] = None,
        return_inputs: bool = True,
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
        self.node_labels = node_labels
        self.rel_labels = rel_labels

        if not isinstance(k, int) or k < 1:
            raise ValueError(f"`k` must be a positive integer, got {k!r}")
        self.k = k

        if members_per_community is not None and (
            not isinstance(members_per_community, int) or members_per_community < 1
        ):
            raise ValueError(
                "`members_per_community` must be a positive integer or None, "
                f"got {members_per_community!r}"
            )
        self.members_per_community = members_per_community
        self.return_inputs = return_inputs

    async def call(self, inputs, training=False):
        if not inputs:
            return None

        graphs = await self.knowledge_base.community_graph_search(
            node_labels=self.node_labels,
            rel_labels=self.rel_labels,
            k=self.k,
            members_per_community=self.members_per_community,
        )
        results = JsonDataModel(
            json=graphs.get_json(),
            schema=KnowledgeGraphs.get_schema(),
            name=self.name,
        )
        if self.return_inputs:
            results = await ops.logical_and(
                inputs,
                results,
                name="results_with_inputs_" + self.name,
            )
        return results

    async def compute_output_spec(self, inputs, training=False):
        results = SymbolicDataModel(
            schema=KnowledgeGraphs.get_schema(),
            name=self.name,
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
            "node_labels": self.node_labels,
            "rel_labels": self.rel_labels,
            "k": self.k,
            "members_per_community": self.members_per_community,
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
        return {
            **config,
            **knowledge_base_config,
        }

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        return cls(
            knowledge_base=knowledge_base,
            **config,
        )
