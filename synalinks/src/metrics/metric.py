# Modified from: keras/src/metrics/metric.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src import initializers
from synalinks.src.api_export import synalinks_export
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name
from synalinks.src.utils.tracking import Tracker


@synalinks_export(["synalinks.Metric", "synalinks.metrics.Metric"])
class Metric(SynalinksSaveable):
    """Metric base class: all synalinks metrics inherit from this class.

    Args:
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).

    Attributes:
        direction (str | None): Class-level optimization direction. ``"up"`` if
            higher values are better (accuracy, F1, reward), ``"down"`` if
            lower is better (cost, latency, loss-like metrics). ``None``
            means unknown — consumers (EarlyStopping, Keras-Tuner inference)
            then require an explicit `mode=` / `direction=`. Concrete metric
            classes set this at class scope; ``Mean``/``MeanMetricWrapper``
            can also set it at instance scope when wrapping a reward.
    """

    direction = None

    def __init__(
        self,
        name=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        self.name = name or auto_name(self.__class__.__name__)
        self._metrics = []
        self._variables = []
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_mask_pattern = in_mask_pattern
        self.out_mask_pattern = out_mask_pattern
        self._tracker = Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    self._variables,
                ),
                "metrics": (lambda x: isinstance(x, Metric), self._metrics),
            }
        )

    def reset_state(self):
        """Reset all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        for v in self.variables:
            initializer = initializers.Empty(data_model=v._data_model)
            v.assign(initializer.get_json())

    async def update_state(self, *args, **kwargs):
        """Accumulate statistics for the metric."""
        raise NotImplementedError

    def stateless_update_state(self, metric_variables, *args, **kwargs):
        if len(metric_variables) != len(self.variables):
            raise ValueError(
                "Argument `metric_variables` must be a list of data_models "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(metric_variables)}, but "
                f"expected {len(self.variables)} variables."
            )
        # Gather variable mapping
        mapping = list(zip(self.variables, metric_variables))

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.update_state(*args, **kwargs)

        # Gather updated variables
        metric_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                metric_variables.append(new_v)
            else:
                metric_variables.append(v)
        return metric_variables

    def result(self):
        """Compute the current metric value.

        Returns:
            (float | dict): A scalar, or a dictionary of scalars.
        """
        raise NotImplementedError

    def stateless_result(self, metric_variables):
        if len(metric_variables) != len(self.variables):
            raise ValueError(
                "Argument `metric_variables` must be a list of data_models "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(metric_variables)}, but "
                f"expected {len(self.variables)} variables."
            )
        # Gather variable mapping
        mapping = list(zip(self.variables, metric_variables))

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping):
            res = self.result()
        return res

    def _obj_type(self):
        return "Metric"

    def add_variable(self, initializer=None, data_model=None, name=None):
        if initializer is None:
            initializer = initializers.Empty(data_model=data_model)
        self._check_super_called()
        with backend.name_scope(self.name.replace("/", ">"), caller=self):
            initializer = initializer
            variable = backend.Variable(
                initializer=initializer,
                data_model=data_model,
                trainable=False,
                name=name,
            )
        # Prevent double-tracking
        self._tracker.add_to_store("variables", variable)
        return variable

    @property
    def variables(self):
        variables = list(self._variables)
        for metric in self._metrics:
            variables.extend(metric.variables)
        return variables

    async def __call__(self, *args, **kwargs):
        self._check_super_called()
        await self.update_state(*args, **kwargs)
        return self.result()

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        return {
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        """Returns a metric instance from config.

        Args:
            config (dict): The config dict.

        Returns:
            (Metric): The metric instance.
        """
        return cls(**config)

    def __setattr__(self, name, value):
        # Track Variables, Layers, Metrics
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def _check_super_called(self):
        if not hasattr(self, "_tracker"):
            raise RuntimeError(
                "You forgot to call `super().__init__()` "
                "in the `__init__()` method. Go add it!"
            )

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"

    def __str__(self):
        return self.__repr__()


def ragged_add(current, fresh):
    """Element-wise add of two per-leaf state vectors that may differ in length.

    Metric state is positional (leaf ``i`` of the flattened JSON). Samples
    with variable-length structures — e.g. an extraction gold whose array
    holds N items — produce a different leaf count per update, so a plain
    ``np.add`` raises a broadcast error the moment two updates disagree on
    length. The shorter vector is zero-padded instead: aggregate sums
    ("micro") stay exact, and per-position reductions simply see no
    contribution at positions a sample doesn't have.
    """
    from synalinks.src.backend.common import numpy as np

    current = list(current)
    fresh = list(fresh)
    size = max(len(current), len(fresh))
    current = current + [0.0] * (size - len(current))
    fresh = fresh + [0.0] * (size - len(fresh))
    return np.add(np.convert_to_numpy(current), np.convert_to_numpy(fresh))
