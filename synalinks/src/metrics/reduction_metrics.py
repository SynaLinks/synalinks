# Modified from: keras/src/metrics/reduction_metrics.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src import rewards
from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend.common import numpy
from synalinks.src.metrics.metric import Metric
from synalinks.src.saving import serialization_lib


def reduce_to_samplewise_values(values, reduce_fn):
    values = numpy.convert_to_tensor(values)
    values_ndim = len(values.shape)
    if values_ndim > 1:
        values = reduce_fn(values, axis=list(range(1, values_ndim)))
    return values


class Total(DataModel):
    total: float = 0.0


@synalinks_export("synalinks.metrics.Sum")
class Sum(Metric):
    """Compute the (weighted) sum of the given values.

    For example, if `values` is `[1, 3, 5, 7]` then their sum is 16.

    This metric creates one variable, `total`.
    This is ultimately returned as the sum value.

    Args:
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).

    Example:

    ```python
    >>> m = metrics.Sum()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result()
    16.0
    ```
    """

    def __init__(
        self,
        name="sum",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        self.total = self.add_variable(
            data_model=Total,
            name="total",
        )

    async def update_state(self, values):
        values = reduce_to_samplewise_values(values, reduce_fn=numpy.sum)
        total = self.total.get("total")
        self.total.update({"total": float(numpy.sum(total, values))})

    def reset_state(self):
        self.total.assign(Total())

    def result(self):
        return self.total.get("total")


class TotalWithCount(DataModel):
    total: float = 0.0
    count: int = 0


@synalinks_export("synalinks.metrics.Mean")
class Mean(Metric):
    """Compute the mean of the given values.

    For example, if values is `[1, 3, 5, 7]` then the mean is 4.

    This metric creates two variables, `total` and `count`.
    The mean value returned is simply `total` divided by `count`.

    Args:
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).

    Example:

    ```python
    >>> m = Mean()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result()
    4.0
    ```
    """

    def __init__(
        self,
        name="mean",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        self.total_with_count = self.add_variable(
            data_model=TotalWithCount, name="total_with_count"
        )

    async def update_state(self, values):
        values = reduce_to_samplewise_values(values, reduce_fn=numpy.mean)
        total = self.total_with_count.get("total")
        self.total_with_count.update({"total": float(total + numpy.sum(values))})
        if len(values.shape) >= 1:
            num_samples = numpy.shape(values)[0]
        else:
            num_samples = 1
        count = self.total_with_count.get("count")
        self.total_with_count.update({"count": int(count + num_samples)})

    def reset_state(self):
        self.total_with_count.assign(TotalWithCount())

    def result(self):
        return float(
            numpy.divide_no_nan(
                self.total_with_count.get("total"),
                self.total_with_count.get("count"),
            )
        )


@synalinks_export("synalinks.metrics.MeanMetricWrapper")
class MeanMetricWrapper(Mean):
    """Wrap a stateless metric function with the `Mean` metric.

    You could use this class to quickly build a mean metric from a function. The
    function needs to have the signature `fn(y_true, y_pred)` and return a
    per-sample reward array. `MeanMetricWrapper.result()` will return
    the average metric value across all samples seen so far.

    Args:
        fn (callable): The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
        **kwargs (keyword arguments): Keyword arguments to pass on to `fn`.
    """

    def __init__(
        self,
        fn,
        name=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        self._fn = fn
        self._fn_kwargs = kwargs

        # If we are wrapping a Synalinks reward, register the metric's
        # direction as "up" (needs to be maximized during training).
        if (
            self._fn in rewards.ALL_OBJECTS
            or hasattr(self._fn, "__class__")
            and self._fn.__class__ in rewards.ALL_OBJECTS
        ):
            self.direction = "up"

    async def update_state(self, y_true, y_pred):
        y_pred = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_pred)
        y_true = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_true)
        if self.in_mask or self.in_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: (
                    x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern)
                    if x is not None
                    else x
                ),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: (
                    x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern)
                    if x is not None
                    else x
                ),
                y_true,
            )
        if self.out_mask or self.out_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: (
                    x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern)
                    if x is not None
                    else x
                ),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: (
                    x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern)
                    if x is not None
                    else x
                ),
                y_true,
            )
        values = await self._fn(y_true, y_pred, **self._fn_kwargs)
        return await super().update_state(values)

    def get_config(self):
        """Returns the serializable config of the metric."""
        base_config = super().get_config()
        config = {
            "fn": serialization_lib.serialize_synalinks_object(self._fn),
        }
        config.update(self._fn_kwargs)
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        fn = serialization_lib.deserialize_synalinks_object(config.pop("fn"))
        return cls(fn=fn, **config)
