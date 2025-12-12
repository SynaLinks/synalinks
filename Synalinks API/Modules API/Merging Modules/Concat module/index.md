## `Concat`

Bases: `Module`

Perform a concatenation operation.

It takes as input a list of data models, and returns a concatenation of them.

If any input is None, an exception is raised.

Table:

| `x1`   | `x2`   | Concat (`+`) |
| ------ | ------ | ------------ |
| `x1`   | `x2`   | `x1 + x2`    |
| `x1`   | `None` | `Exception`  |
| `None` | `x2`   | `Exception`  |
| `None` | `None` | `Exception`  |

Parameters:

| Name       | Type                | Description                                | Default |
| ---------- | ------------------- | ------------------------------------------ | ------- |
| `**kwargs` | `keyword arguments` | Standard keyword arguments for the module. | `{}`    |

Source code in `synalinks/src/modules/merging/concat.py`

```
@synalinks_export(
    [
        "synalinks.Concat",
        "synalinks.Concatenate",
        "synalinks.modules.Concat",
        "synalinks.modules.Concatenate",
    ]
)
class Concat(Module):
    """Perform a concatenation operation.

    It takes as input a list of data models,
    and returns a concatenation of them.

    If any input is None, an exception is raised.

    Table:

    | `x1`   | `x2`   | Concat (`+`)      |
    | ------ | ------ | ----------------- |
    | `x1`   | `x2`   | `x1 + x2`         |
    | `x1`   | `None` | `Exception`       |
    | `None` | `x2`   | `Exception`       |
    | `None` | `None` | `Exception`       |

    Args:
        **kwargs (keyword arguments): Standard keyword arguments for the module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def call(self, inputs, training=False):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = await ops.concat(
                output,
                inputs[i],
                name=f"module_concat_{i}_" + self.name,
            )
        return output
```
