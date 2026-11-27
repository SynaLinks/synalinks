# Modified from: keras/src/metrics/metric.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src import initializers
from synalinks.src import ops
from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name
from synalinks.src.utils.tracking import Tracker


@synalinks_export(["synalinks.Metric", "synalinks.metrics.Metric"])
class Metric(SynalinksSaveable):
    """Metric base class: all synalinks metrics inherit from this class.

    During `fit()` / `evaluate()`, an `update_state()` that also declares an `x`
    argument receives the program's inputs (per sample, or the batch's list for
    a `BatchMetric`), e.g. to grade with a judge that needs the question.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.Metric(),
        ],
    )
    ```

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
            lower is better (cost, latency, token usage, errors). Defaults to
            ``"up"``; operational (LM, embedding model, program) metrics
            default to ``"down"``. Consumers (EarlyStopping, ProgramCheckpoint,
            Keras-Tuner inference) use it to resolve ``mode="auto"``. Setting
            it to ``None`` forces an explicit `mode=` / `direction=`.
    """

    direction = "up"

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

    async def close(self):
        """Release what this metric holds. Idempotent.

        Nothing, for most metrics; a judge's program, for one wrapping a judge.
        """

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

    def result(self):
        """Compute the current metric value.

        Returns:
            (float | dict): A scalar, or a dictionary of scalars.
        """
        raise NotImplementedError

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


def align_leaves(y_true, y_pred, fill=None):
    """Pair the leaves of two data models by their path in the JSON.

    Pairing by position would shift every field after a list whose length
    differs between the two. A failed prediction (`y_pred is None`) has no
    leaves.

    Args:
        y_true (JsonDataModel): The ground truth.
        y_pred (JsonDataModel): The prediction, or `None`.
        fill (any): The value of a leaf missing on one side.

    Returns:
        (tuple): The paths (e.g. `"sources/1"`), the `y_true` leaves and the
            `y_pred` leaves, aligned.
    """

    def leaves(value, path=""):
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, list):
            items = enumerate(value)
        else:
            return {path: value}
        result = {}
        for key, item in items:
            result.update(leaves(item, f"{path}/{key}" if path else str(key)))
        return result

    true = leaves(y_true.get_json())
    pred = {} if y_pred is None else leaves(y_pred.get_json())
    paths = list(true) + [path for path in pred if path not in true]
    return paths, [true.get(p, fill) for p in paths], [pred.get(p, fill) for p in paths]


def accumulate(state, fields, **values):
    """Add per-field counts into a metric state, matching fields by name.

    The state keeps the list of fields seen so far under `"fields"`, and each
    count list in the same order, so samples with different structures (e.g.
    lists of different lengths) never mix the counts of different fields.

    Args:
        state (Variable): The metric state.
        fields (list): The name of each count, e.g. a leaf path or a label.
        **values (list): The new counts, one list per state key, aligned with
            `fields`.
    """
    known = list(state.get("fields") or [])
    known += [field for field in dict.fromkeys(fields) if field not in known]
    update = {"fields": known}
    for name, new in values.items():
        current = list(state.get(name) or [])
        current += [0.0] * (len(known) - len(current))
        for field, value in zip(fields, new):
            current[known.index(field)] += float(value)
        update[name] = current
    state.update(update)


def mask_data_models(
    value,
    in_mask=None,
    in_mask_pattern=None,
    out_mask=None,
    out_mask_pattern=None,
):
    """Convert `value` to `JsonDataModel` and apply field masks.

    Works on a single data model or a list/tree of them (`None` leaves are
    kept as is).
    """

    def apply(x):
        x = ops.convert_to_json_data_model(x)
        if x is None:
            return x
        if in_mask or in_mask_pattern:
            x = x.in_mask(mask=in_mask, pattern=in_mask_pattern)
        if out_mask or out_mask_pattern:
            x = x.out_mask(mask=out_mask, pattern=out_mask_pattern)
        return x

    return tree.map_structure(apply, value)


def apply_masks(
    y_true,
    y_pred,
    in_mask=None,
    in_mask_pattern=None,
    out_mask=None,
    out_mask_pattern=None,
):
    """Convert ``y_true``/``y_pred`` to ``JsonDataModel`` and apply field masks.

    Works on both a single sample (a leaf ``JsonDataModel``) and a list/tree
    of samples; ``tree.map_structure`` handles either uniformly.
    """
    masks = {
        "in_mask": in_mask,
        "in_mask_pattern": in_mask_pattern,
        "out_mask": out_mask,
        "out_mask_pattern": out_mask_pattern,
    }
    return mask_data_models(y_true, **masks), mask_data_models(y_pred, **masks)
