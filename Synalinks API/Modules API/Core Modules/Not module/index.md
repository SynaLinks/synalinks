## `Not`

Bases: `Module`

Not module.

This module should be used as a placeholder when no operation is to be performed and the output should be None.

This module is useful to implement stop conditions when combined with a conditional branch or as placeholder (like the Identity) before implementing guards that leverage the xor operation.

Parameters:

| Name       | Type                | Description                    | Default |
| ---------- | ------------------- | ------------------------------ | ------- |
| `**kwargs` | `keyword arguments` | The default module's arguments | `{}`    |

Source code in `synalinks/src/modules/core/not_module.py`

```
@synalinks_export(["synalinks.modules.Not", "synalinks.Not"])
class Not(Module):
    """Not module.

    This module should be used as a placeholder when no operation is to be
    performed and the output should be None.

    This module is useful to implement stop conditions when combined with a conditional
    branch or as placeholder (like the Identity) before implementing guards that leverage
    the xor operation.

    Args:
        **kwargs (keyword arguments): The default module's arguments
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.built = True

    async def call(self, inputs):
        if isinstance(inputs, (JsonDataModel, SymbolicDataModel)):
            return None
        return tree.map_structure(
            lambda x: None,
            inputs,
        )

    async def compute_output_spec(self, inputs):
        if isinstance(inputs, (JsonDataModel, SymbolicDataModel)):
            return inputs.clone()
        return tree.map_structure(
            lambda x: x.clone(),
            inputs,
        )
```
