# Modified from: keras/src/losses/losses.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.backend.common.json_utils import without_echoed_inputs
from synalinks.src.rewards.reward import Reward
from synalinks.src.rewards.reward import check_async_reward_fn
from synalinks.src.saving import serialization_lib


@synalinks_export("synalinks.rewards.RewardFunctionWrapper")
class RewardFunctionWrapper(Reward):
    """Wrap a stateless function into a `Reward`.

    You can use this to quickly build a reward from a function. The function needs
    to have the signature `fn(y_true, y_pred)` and to be declared with
    `async def`, since reward functions are awaited: a synchronous one raises a
    `TypeError` here rather than failing later inside the training loop.

    Example:

    ```python
    async def my_reward(y_true, y_pred):
        # ...
        return reward

    program.compile(
        reward=synalinks.rewards.RewardFunctionWrapper(fn=my_reward),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    Wrapping is optional: `compile(reward=my_reward)` accepts the bare function
    and wraps it for you, naming the reward after the function. Reach for this
    class explicitly when you need masks, a custom `reduction`, or extra keyword
    arguments forwarded to `fn`:

    ```python
    async def length_under(y_true, y_pred, limit=100):
        return 1.0 if len(y_pred.get("answer")) < limit else 0.0

    program.compile(
        reward=synalinks.rewards.RewardFunctionWrapper(
            fn=length_under,
            limit=200,
            in_mask=["answer"],
        ),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    Args:
        fn (callable): Async reward function to wrap, with signature
            ``fn(y_true, y_pred, **kwargs)``.
        reduction (str): Optional. How the per-sample rewards of a batch are
            reduced, one of `"mean"`, `"sum"`, `"min"`, `"max"`, `"none"` or
            `None` (see `Reward`). Defaults to `"mean"`.
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
        in_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
        **kwargs (keyword arguments): Keyword arguments to pass on to `fn`.
    """

    def __init__(
        self,
        fn,
        reduction="mean",
        name=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        check_async_reward_fn(fn, self.__class__.__name__)
        self.fn = fn
        self._fn_kwargs = kwargs

    async def call(self, y_true, y_pred):
        return await self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = super().get_config()
        config["fn"] = serialization_lib.serialize_synalinks_object(self.fn)
        # Keep fn kwargs under their own key so they cannot collide with
        # base-class fields like ``name`` or ``reduction``.
        config["fn_kwargs"] = serialization_lib.serialize_synalinks_object(
            self._fn_kwargs
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = serialization_lib.deserialize_synalinks_object(config)
        fn_kwargs = config.pop("fn_kwargs", None) or {}
        return cls(**config, **fn_kwargs)

    def __repr__(self):
        return f"<RewardFunctionWrapper({self.fn}, kwargs={self._fn_kwargs})>"


async def judge_inputs(y_true, y_pred, x=None):
    """Build what a judge grades from the program's inputs, output and target.

    The program's inputs (when given) come first with their fields prefixed
    with `input_`, then its output with `output_`, then the gold reference
    (when given) with `gold_`. The output fields that merely repeat an input
    (same key and value, e.g. from `return_inputs=True`) are dropped, so each
    appears once, as an input; an output made only of such fields is kept
    whole. Prefixing also renames chat `messages`, so every judge reads a
    conversation as data instead of continuing it.

    Args:
        y_true (JsonDataModel): The gold reference, or `None`.
        y_pred (JsonDataModel): The program's output to grade.
        x (JsonDataModel | list): Optional. The program's inputs (a list for a
            multi-input program).

    Returns:
        (JsonDataModel | SymbolicDataModel): The judge's inputs.
    """
    inputs = None
    for x_i in x if isinstance(x, (list, tuple)) else [x]:
        if x_i is not None:
            inputs = x_i if inputs is None else await ops.concat(inputs, x_i)
    parts = []
    if inputs is not None:
        concrete = not (is_symbolic_data_model(inputs) or is_symbolic_data_model(y_pred))
        output = y_pred.get_json() if concrete else {}
        kept = without_echoed_inputs(output, inputs.get_json()) if concrete else {}
        # An output that only echoes its input is still the output: shown whole.
        if kept and len(kept) < len(output):
            schema = y_pred.get_schema()
            y_pred = JsonDataModel(
                json=kept,
                schema={
                    **schema,
                    "properties": {
                        key: value
                        for key, value in schema["properties"].items()
                        if key in kept
                    },
                    "required": [
                        key for key in schema.get("required", []) if key in kept
                    ],
                },
                name=y_pred.name,
            )
        parts.append(await ops.prefix(inputs, prefix="input", name="input_x"))
    parts.append(await ops.prefix(y_pred, prefix="output", name="output_y_pred"))
    if y_true:
        parts.append(await ops.prefix(y_true, prefix="gold", name="gold_y_true"))
    result = parts[0]
    for part in parts[1:]:
        result = await ops.concat(result, part)
    return result


@synalinks_export(
    [
        "synalinks.ProgramAsJudge",
        "synalinks.rewards.ProgramAsJudge",
    ]
)
class ProgramAsJudge(Reward):
    """Wrap a `Program` into a `Reward`.

    You can use this to create advanced reward functions that use a Synalinks `Program`.
    The program should have two inputs (`y_true`, `y_pred`) and one output.
    A program declaring a third input also receives the inputs of the program
    being judged (`x`), e.g. to grade an answer against its question. `x` is
    `None` when the reward is called without it, so declare that input with
    `optional=True` to judge with or without it (the same goes for `y_true`
    when there is no ground truth).

    **Note:** The output data model/schema should have a field named `reward`.

    Example:

    ```python
    # ... your judge program declaration (inputs: y_true, y_pred)

    judge_program = synalinks.Program(
        inputs=[x0, x1],
        outputs=xn,
    )

    program.compile(
        reward=synalinks.rewards.ProgramAsJudge(program=judge_program),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    With the program's inputs:

    ```python
    y_true = synalinks.Input(data_model=Answer, optional=True)
    y_pred = synalinks.Input(data_model=Answer)
    x = synalinks.Input(data_model=Query, optional=True)
    # ... your judge program declaration (inputs: y_true, y_pred, x)

    judge_program = synalinks.Program(
        inputs=[y_true, y_pred, x],
        outputs=xn,
    )
    ```

    Args:
        program (Program): The reward program to wrap.
        reduction (str): Optional. How the per-sample rewards of a batch are
            reduced, one of `"mean"`, `"sum"`, `"min"`, `"max"`, `"none"` or
            `None` (see `Reward`). Defaults to `"mean"`.
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
        in_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
    """

    # Set by a judge that builds its own program from its arguments.
    program_class = None

    def __init__(
        self,
        program,
        reduction="mean",
        name=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        self.program = program

    async def close(self):
        """Close the judge's program: an agent judge's sandbox, say. Idempotent."""
        await self.program.close()

    async def call(self, y_true, y_pred, x=None):
        # The built-in judges, and a program declaring a third input, take `x`.
        if self.program_class or len(getattr(self.program, "inputs", None) or ()) == 3:
            result = await self.program([y_true, y_pred, x])
        else:
            result = await self.program([y_true, y_pred])
        if result is None:
            warnings.warn(
                f"{self.__class__.__name__}: judge program returned None "
                "(likely an LLM / provider failure). Scoring this sample as "
                "0.0 and continuing. Check the underlying language model "
                "and structured-output configuration.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
        return float(result.get("reward", 0.0))

    def get_config(self):
        config = super().get_config()
        config["program"] = serialization_lib.serialize_synalinks_object(self.program)
        return config

    @classmethod
    def from_config(cls, config):
        if cls.program_class is None:
            if "program" in config:
                config = serialization_lib.deserialize_synalinks_object(config)
            return cls(**config)
        # A judge that builds its own program takes the program's arguments, not
        # the program: building it would also build what it opens (a sandbox).
        config = dict(config)
        program = config.pop("program")
        arguments = cls.program_class.resolve_config(program["config"])
        for key in ("name", "description", "trainable"):
            arguments.pop(key, None)
        return cls(**arguments, **config)

    def __repr__(self):
        return f"<ProgramAsJudge({self.program})>"
