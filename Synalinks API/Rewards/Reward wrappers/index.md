## `ProgramAsJudge`

Bases: `Reward`

Wrap a `Program` into a `Reward`.

You can use this to create advanced reward functions that use a Synalinks `Program`. The program should have two inputs and one output.

**Note:** The output data model/schema should have a field named `reward`.

Example:

```
# ... your program declaration

program = synalinks.Program(
    inputs=x0,
    outputs=xn,
)

program.compile(
    reward=synalinks.rewards.ProgramAsJudge(program=program)
    optimizer=synalinks.optimizers.RandomFewShot(),
)
```

Parameters:

| Name       | Type      | Description                                             | Default    |
| ---------- | --------- | ------------------------------------------------------- | ---------- |
| `program`  | `Program` | The reward program to wrap.                             | *required* |
| `name`     | `str`     | Optional. string name of the reward instance.           | `None`     |
| `in_mask`  | `list`    | Optional. list of keys to keep to compute the reward.   | `None`     |
| `out_mask` | `list`    | Optional. list of keys to remove to compute the reward. | `None`     |

Source code in `synalinks/src/rewards/reward_wrappers.py`

````
@synalinks_export(
    [
        "synalinks.ProgramAsJudge",
        "synalinks.rewards.ProgramAsJudge",
    ]
)
class ProgramAsJudge(Reward):
    """Wrap a `Program` into a `Reward`.

    You can use this to create advanced reward functions that use a Synalinks `Program`.
    The program should have two inputs and one output.

    **Note:** The output data model/schema should have a field named `reward`.

    Example:

    ```python
    # ... your program declaration

    program = synalinks.Program(
        inputs=x0,
        outputs=xn,
    )

    program.compile(
        reward=synalinks.rewards.ProgramAsJudge(program=program)
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    Args:
        program (Program): The reward program to wrap.
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
    """

    def __init__(
        self,
        program,
        reduction="mean",
        name=None,
        in_mask=None,
        out_mask=None,
    ):
        super().__init__(
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
        )
        self.program = program

    async def call(self, y_true, y_pred):
        result = await self.program([y_true, y_pred])
        return float(result.get("reward", 0.0))

    def get_config(self):
        config = super().get_config()
        config.update({"program": self.program})
        return config

    def __repr__(self):
        return f"<ProgramAsJudge({self.program})>"
````

## `RewardFunctionWrapper`

Bases: `Reward`

Wrap a stateless function into a `Reward`.

You can use this to quickly build a reward from a function. The function needs to have the signature `fn(y_true, y_pred)`.

Example:

```
def my_reward(y_true, y_pred):
    # ...
    return reward

program.compile(
    reward=synalinks.rewards.RewardFunctionWrapper,
    optimizer=synalinks.optimizers.RandomFewShot()
)
```

Parameters:

| Name       | Type                | Description                                                                 | Default    |
| ---------- | ------------------- | --------------------------------------------------------------------------- | ---------- |
| `fn`       | `callable`          | The reward function to wrap, with signature fn(y_true, y_pred, \*\*kwargs). | *required* |
| `name`     | `str`               | Optional. string name of the reward instance.                               | `None`     |
| `in_mask`  | `list`              | Optional. list of keys to keep to compute the reward.                       | `None`     |
| `out_mask` | `list`              | Optional. list of keys to remove to compute the reward.                     | `None`     |
| `**kwargs` | `keyword arguments` | Keyword arguments to pass on to fn.                                         | `{}`       |

Source code in `synalinks/src/rewards/reward_wrappers.py`

````
@synalinks_export("synalinks.rewards.RewardFunctionWrapper")
class RewardFunctionWrapper(Reward):
    """Wrap a stateless function into a `Reward`.

    You can use this to quickly build a reward from a function. The function needs
    to have the signature `fn(y_true, y_pred)`.

    Example:

    ```python

    def my_reward(y_true, y_pred):
        # ...
        return reward

    program.compile(
        reward=synalinks.rewards.RewardFunctionWrapper,
        optimizer=synalinks.optimizers.RandomFewShot()
    )
    ```

    Args:
        fn (callable): The reward function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
        **kwargs (keyword arguments): Keyword arguments to pass on to `fn`.
    """

    def __init__(
        self,
        fn,
        reduction="mean",
        name=None,
        in_mask=None,
        out_mask=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
        )
        self.fn = fn
        self._fn_kwargs = kwargs

    async def call(self, y_true, y_pred):
        return await self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"fn": serialization_lib.serialize_synalinks_object(self.fn)})
        config.update(serialization_lib.serialize_synalinks_object(self._fn_kwargs))
        return config

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = serialization_lib.deserialize_synalinks_object(config)
        return cls(**config)

    def __repr__(self):
        return f"<RewardFunctionWrapper({self.fn}, kwargs={self._fn_kwargs})>"
````
