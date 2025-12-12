## `UpdateKnowledge`

Bases: `Module`

Update the given knowledge base.

This module requires an `Entity`, `Relation`, `Entities`, `Relations` or `KnowledgeGraph` data model as input.

This module perform alignment automatically, also called deduplication, by using the similarity search of the knowledge base. This way of performing alignment is more performant than using a linear alignement algorithm as it use the hierarchical small world neighbors (HSWN) algorithm of the knowledge base.

It however needs to have the entities embeded using the `Embedding` module before updating the knwoledge base.

Parameters:

| Name             | Type            | Description                                                                                                                                              | Default |
| ---------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `knowledge_base` | `KnowledgeBase` | The knowledge base to update.                                                                                                                            | `None`  |
| `threshold`      | `float`         | Similarity threshold for entity alignment. Entities with similarity above this threshold may be merged. Should be between 0.0 and 1.0 (Defaults to 0.8). | `0.8`   |
| `name`           | `str`           | Optional. The name of the module.                                                                                                                        | `None`  |
| `description`    | `str`           | Optional. The description of the module.                                                                                                                 | `None`  |
| `trainable`      | `bool`          | Whether the module's variables should be trainable.                                                                                                      | `False` |

Source code in `synalinks/src/modules/knowledge/update_knowledge.py`

```
@synalinks_export(
    [
        "synalinks.modules.UpdateKnowledge",
        "synalinks.UpdateKnowledge",
    ]
)
class UpdateKnowledge(Module):
    """Update the given knowledge base.

    This module requires an `Entity`, `Relation`, `Entities`,
    `Relations` or `KnowledgeGraph` data model as input.

    This module perform alignment automatically, also called deduplication,
    by using the similarity search of the knowledge base. This way of performing
    alignment is more performant than using a linear alignement algorithm as it use
    the hierarchical small world neighbors (HSWN) algorithm of the knowledge base.

    It however needs to have the entities embeded using the `Embedding` module before
    updating the knwoledge base.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to update.
        threshold (float): Similarity threshold for entity alignment.
            Entities with similarity above this threshold may be merged.
            Should be between 0.0 and 1.0 (Defaults to 0.8).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        knowledge_base=None,
        threshold=0.8,
        name=None,
        description=None,
        trainable=False,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.knowledge_base = knowledge_base
        self.threshold = threshold

    async def call(self, inputs):
        if not inputs:
            return None
        if is_knowledge_graph(inputs):
            for entity in inputs.get_nested_entity_list("entities"):
                _ = await ops.update_knowledge(
                    entity,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=entity.name + "_updated",
                )
            for relation in inputs.get_nested_entity_list("relations"):
                subj = relation.get_nested_entity("subj")
                if not subj:
                    continue
                _ = await ops.update_knowledge(
                    subj,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=subj.name + "_updated",
                )
                obj = relation.get_nested_entity("obj")
                if not obj:
                    continue
                _ = await ops.update_knowledge(
                    obj,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=obj.name + "_updated",
                )
                _ = await ops.update_knowledge(
                    relation,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=relation.name + "_updated",
                )
            return inputs.clone(name=inputs.name + "_updated")
        elif is_entities(inputs):
            for entity in inputs.get_nested_entity_list("entities"):
                _ = await ops.update_knowledge(
                    entity,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=entity.name + "_updated",
                )
            return inputs.clone(name=inputs.name + "_updated")
        elif is_relations(inputs):
            for relation in inputs.get_nested_entity_list("relations"):
                subj = relation.get_nested_entity("subj")
                if not subj:
                    continue
                _ = await ops.update_knowledge(
                    subj,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=subj.name + "_updated",
                )
                obj = relation.get_nested_entity("obj")
                if not obj:
                    continue
                _ = await ops.update_knowledge(
                    obj,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=obj.name + "_updated",
                )
                _ = await ops.update_knowledge(
                    relation,
                    knowledge_base=self.knowledge_base,
                    threshold=self.threshold,
                    name=relation.name + "_updated",
                )
            return inputs.clone(name=inputs.name + "_updated")
        elif is_relation(inputs):
            subj = inputs.get_nested_entity("subj")
            _ = await ops.update_knowledge(
                subj,
                knowledge_base=self.knowledge_base,
                threshold=self.threshold,
                name=subj.name + "_updated",
            )
            obj = inputs.get_nested_entity("obj")
            _ = await ops.update_knowledge(
                obj,
                knowledge_base=self.knowledge_base,
                threshold=self.threshold,
                name=obj.name + "_updated",
            )
            return inputs.clone(name=inputs.name + "_updated")
        elif is_entity(inputs):
            _ = await ops.update_knowledge(
                inputs,
                knowledge_base=self.knowledge_base,
                threshold=self.threshold,
                name=inputs.name + "_updated",
            )
            return inputs.clone(name=inputs.name + "_updated")
        else:
            return None

    async def compute_output_spec(self, inputs):
        return inputs.clone()

    def get_config(self):
        config = {
            "threshold": self.threshold,
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
```
