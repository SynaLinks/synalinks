## `UpdateKnowledge`

Bases: `Module`

Update (insert or upsert) data models in the given knowledge base.

This module stores data models in the knowledge base, using the first field of the data model as the primary key for upsert operations.

Parameters:

| Name             | Type            | Description                                         | Default |
| ---------------- | --------------- | --------------------------------------------------- | ------- |
| `knowledge_base` | `KnowledgeBase` | The knowledge base to update.                       | `None`  |
| `name`           | `str`           | Optional. The name of the module.                   | `None`  |
| `description`    | `str`           | Optional. The description of the module.            | `None`  |
| `trainable`      | `bool`          | Whether the module's variables should be trainable. | `False` |

Source code in `synalinks/src/modules/knowledge/update_knowledge.py`

```
@synalinks_export(
    [
        "synalinks.modules.UpdateKnowledge",
        "synalinks.UpdateKnowledge",
    ]
)
class UpdateKnowledge(Module):
    """Update (insert or upsert) data models in the given knowledge base.

    This module stores data models in the knowledge base, using the first
    field of the data model as the primary key for upsert operations.

    Args:
        knowledge_base (KnowledgeBase): The knowledge base to update.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
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
        self.knowledge_base = knowledge_base

    async def _update(self, data_model):
        await self.knowledge_base.update(data_model)
        return data_model.clone(name="updated_" + data_model.name)

    async def call(self, inputs):
        if not inputs:
            return None
        outputs = tree.map_structure(
            lambda x: run_maybe_nested(self._update(x)),
            inputs,
        )
        return outputs

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
```
