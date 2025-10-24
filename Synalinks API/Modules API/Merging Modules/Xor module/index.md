## `Xor`

Bases: `Module`

Perform a logical Xor operation.

It takes as input a list of data models, If more than two data models are not None, then it output None. otherwise it output the one that is not None.

Table:

| `x1`   | `x2`   | Logical Xor (`^`) |
| ------ | ------ | ----------------- |
| `x1`   | `x2`   | `None`            |
| `x1`   | `None` | `x1`              |
| `None` | `x2`   | `x2`              |
| `None` | `None` | `None`            |

Parameters:

| Name       | Type                | Description                                | Default |
| ---------- | ------------------- | ------------------------------------------ | ------- |
| `**kwargs` | `keyword arguments` | Standard keyword arguments for the module. | `{}`    |

Source code in `synalinks/src/modules/merging/logical_xor.py`

```
@synalinks_export(
    [
        "synalinks.Xor",
        "synalinks.modules.Xor",
    ]
)
class Xor(Module):
    """Perform a logical Xor operation.

    It takes as input a list of data models,
    If more than two data models are not None, then it output None.
    otherwise it output the one that is not None.

    Table:

    | `x1`   | `x2`   | Logical Xor (`^`)|
    | ------ | ------ | ---------------- |
    | `x1`   | `x2`   | `None`           |
    | `x1`   | `None` | `x1`             |
    | `None` | `x2`   | `x2`             |
    | `None` | `None` | `None`           |

    Args:
        **kwargs (keyword arguments): Standard keyword arguments for the module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def compute_output_spec(self, inputs, training=False):
        return inputs[0].clone()

    async def call(self, inputs, training=False):
        output = inputs[0]
        for i in range(1, len(inputs)):
            if inputs[i]:
                if not output:
                    output = inputs[i]
                else:
                    return None
        return output.clone(name=self.name)
```
