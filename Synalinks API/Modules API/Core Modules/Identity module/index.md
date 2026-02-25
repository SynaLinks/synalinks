## `Identity`

Bases: `Module`

Identity module.

This module should be used as a placeholder when no operation is to be performed. The module just returns its `inputs` argument as output.

This module can be really useful during development process in order to implement the whole program architecture before the individual modules.

It avoid any data models naming issue that you could have by just forwarding the inputs, that way you can implement the general program architecture, validate it and implement the individual modules later.

Example:

```
import synalinks

class MyAwesomeModule(synalinks.Program):

    def __init__(
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

    async def build(self, inputs):
        outputs = await synalinks.Identity()(inputs)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )
```

Parameters:

| Name       | Type                | Description                    | Default |
| ---------- | ------------------- | ------------------------------ | ------- |
| `**kwargs` | `keyword arguments` | The default module's arguments | `{}`    |

Source code in `synalinks/src/modules/core/identity.py`

````
@synalinks_export(["synalinks.modules.Identity", "synalinks.Identity"])
class Identity(Module):
    """Identity module.

    This module should be used as a placeholder when no operation is to be
    performed. The module just returns its `inputs` argument as output.

    This module can be really useful during development process in order to
    implement the whole program architecture before the individual modules.

    It avoid any data models naming issue that you could have by just
    forwarding the inputs, that way you can implement the general
    program architecture, validate it and implement the individual
    modules later.

    Example:

    ```python
    import synalinks

    class MyAwesomeModule(synalinks.Program):

        def __init__(
            name=None,
            description=None,
            trainable=True,
        ):
            super().__init__(
                name=name,
                description=description,
                trainable=trainable,
            )

        async def build(self, inputs):
            outputs = await synalinks.Identity()(inputs)

            super().__init__(
                inputs=inputs,
                outputs=outputs,
                name=self.name,
                description=self.description,
                trainable=self.trainable,
            )
    ```

    Args:
        **kwargs (keyword arguments): The default module's arguments
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.built = True

    async def call(self, inputs):
        if isinstance(inputs, (JsonDataModel, SymbolicDataModel)):
            return inputs.clone()
        return tree.map_structure(
            lambda x: x.clone(),
            inputs,
        )
````
