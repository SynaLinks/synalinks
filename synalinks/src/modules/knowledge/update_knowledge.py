# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_entities
from synalinks.src.backend import is_entity
from synalinks.src.backend import is_knowledge_graph
from synalinks.src.backend import is_knowledge_graphs
from synalinks.src.backend import is_relation
from synalinks.src.backend import is_relations
from synalinks.src.knowledge_bases import get as _get_kb
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


@synalinks_export(
    [
        "synalinks.modules.UpdateKnowledge",
        "synalinks.UpdateKnowledge",
    ]
)
class UpdateKnowledge(Module):
    """Update (insert or upsert) data models in the given knowledge base.

    Dispatches on the input's shape:

    * ``KnowledgeGraph`` (``is_knowledge_graph``) →
      :meth:`KnowledgeBase.update_knowledge_graph` (bulk entities + relations).
    * ``KnowledgeGraphs`` wrapper (``is_knowledge_graphs``) →
      :meth:`KnowledgeBase.update_knowledge_graph` once per wrapped KG
      (the KB has no bulk multi-graph endpoint).
    * ``Entities`` wrapper (``is_entities``) →
      :meth:`KnowledgeBase.update_entities` with the wrapped list (bulk nodes).
    * ``Relations`` wrapper (``is_relations``) →
      :meth:`KnowledgeBase.update_relations` with the wrapped list (bulk edges).
    * ``Relation`` (``is_relation``) →
      :meth:`KnowledgeBase.update_relations` (single edge; endpoints upserted
      as needed).
    * ``Entity`` (``is_entity``) →
      :meth:`KnowledgeBase.update_entities` (single node).
    * Anything else → :meth:`KnowledgeBase.update` (SQL row/table store);
      the first declared field is used as the primary key for upsert.

    To pass a Python list of entities or relations rather than a wrapper
    data model, ``tree.map_structure`` flattens the inputs so each item
    is dispatched individually — wrap as ``Entities`` / ``Relations``
    for a single bulk round-trip instead of N per-item ones.

    The output is a name-mangled clone of the input (``"updated_" +
    name``), passed through unchanged. The KB methods' return values
    (assigned ids) are ignored — use the KB directly if you need them.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to update.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        *,
        knowledge_base=None,
        name=None,
        description=None,
        trainable=False,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.knowledge_base = _get_kb(knowledge_base)

    async def _update(self, data_model):
        # Order matters:
        #   * KnowledgeGraph passes is_entities AND is_relations
        #     (it has both fields) — check it FIRST.
        #   * is_relation also passes is_entity (a Relation has a
        #     `label` field) — relation check must come before entity.
        if is_knowledge_graph(data_model):
            await self.knowledge_base.update_knowledge_graph(data_model)
        elif is_knowledge_graphs(data_model):
            # No bulk multi-graph endpoint — iterate per KG.
            for kg in data_model.get_nested_entity_list("knowledge_graphs") or []:
                await self.knowledge_base.update_knowledge_graph(kg)
        elif is_entities(data_model):
            # Bulk wrapper — pass the underlying list to update_entities
            # so the adapter sees one batched call instead of N per-entity
            # round-trips.
            await self.knowledge_base.update_entities(data_model.get("entities") or [])
        elif is_relations(data_model):
            await self.knowledge_base.update_relations(data_model.get("relations") or [])
        elif is_relation(data_model):
            await self.knowledge_base.update_relations(data_model)
        elif is_entity(data_model):
            await self.knowledge_base.update_entities(data_model)
        else:
            await self.knowledge_base.update(data_model)
        return data_model.clone(name="updated_" + data_model.name)

    async def call(self, inputs):
        if not inputs:
            return None
        # Await each update on the current event loop, sequentially (avoids the
        # transient thread-loops `run_maybe_nested` created — which orphaned
        # litellm's global client and serialized DB writes anyway). flatten/pack
        # mirrors `map_structure` since data models are tree leaves.
        leaves = tree.flatten(inputs)
        outputs = [await self._update(leaf) for leaf in leaves]
        return tree.pack_sequence_as(inputs, outputs)

    async def compute_output_spec(self, inputs):
        return tree.map_structure(
            lambda x: x.clone(name="updated_" + x.name),
            inputs,
        )

    def get_config(self):
        config = {
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        knowledge_base_config = {
            "knowledge_base": serialization_lib.serialize_synalinks_object(
                self.knowledge_base
            )
        }
        return {**knowledge_base_config, **config}

    @classmethod
    def from_config(cls, config):
        knowledge_base = serialization_lib.deserialize_synalinks_object(
            config.pop("knowledge_base")
        )
        return cls(knowledge_base=knowledge_base, **config)
