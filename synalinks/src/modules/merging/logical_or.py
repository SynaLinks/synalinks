# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.module import Module


@synalinks_export(
    [
        "synalinks.Or",
        "synalinks.modules.Or",
    ]
)
class Or(Module):
    """Perform a logical Or operation.

    It takes as input a list of data models,
    and returns a concatenation of them (if all are provided)
    otherwise it output the one that is not None.

    If any input is None, it is ignored.

    Table:

    | `x1`   | `x2`   | Logical Or (`|`) |
    | ------ | ------ | ---------------- |
    | `x1`   | `x2`   | `x1 + x2`        |
    | `x1`   | `None` | `x1`             |
    | `None` | `x2`   | `x2`             |
    | `None` | `None` | `None`           |

    Args:
        **kwargs (keyword arguments): Standard keyword arguments for the module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def call(self, inputs, training=False):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = await ops.logical_or(
                output,
                inputs[i],
                name=f"module_or_{i}_" + self.name,
            )
        return output
