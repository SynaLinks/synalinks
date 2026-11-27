# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Optional

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend.common import numpy as np
from synalinks.src.metrics.metric import Metric
from synalinks.src.metrics.metric import accumulate
from synalinks.src.metrics.metric import align_leaves
from synalinks.src.metrics.metric import apply_masks
from synalinks.src.metrics.metrics_utils import aggregate_scores
from synalinks.src.utils import nlp_utils


class AccuracyState(DataModel):
    correct: Optional[List[float]] = None
    total: Optional[List[float]] = None
    intermediate_weights: Optional[List[float]] = None
    fields: Optional[List[str]] = None


@synalinks_export("synalinks.metrics.Accuracy")
class Accuracy(Metric):
    """Computes per-field token accuracy.

    Formula (per field, Jaccard index over normalized tokens):

    ```python
    accuracy = |y_true_tokens ∩ y_pred_tokens| / |y_true_tokens ∪ y_pred_tokens|
    ```

    Its output range is `[0, 1]`. It operates at a word level
    and can be used for **QA systems**.

    If `y_true` and `y_pred` contain multiple fields the JSON object's
    fields are flattened and the score computed for each one
    independently, then reduced
    according to `average` (a list of per-field scores by default).


    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.Accuracy(average="macro"),
        ],
    )
    ```

    Args:
        average (str): Type of averaging to be performed across per-field results
            in the multi-field case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each field.
            If `"micro"`, compute the metric globally by aggregating
            tokens across all fields.
            If `"macro"`, compute the metric for each field,
            and return their unweighted mean.
            If `"weighted"`, compute the metric for each field,
            and return their mean weighted by support
            (the number of `y_true` tokens for each field).
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
    """

    direction = "up"

    def __init__(
        self,
        average=None,
        name="accuracy",
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
        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Invalid `average` argument value. Expected one of: "
                "[None, 'micro', 'macro', 'weighted']. "
                f"Received: average={average}"
            )
        self.state = self.add_variable(
            data_model=AccuracyState,
            name="state_" + self.name,
        )
        self.average = average
        self.axis = None
        if self.average != "micro":
            self.axis = 0

    async def update_state(self, y_true, y_pred):
        y_true, y_pred = apply_masks(
            y_true,
            y_pred,
            in_mask=self.in_mask,
            in_mask_pattern=self.in_mask_pattern,
            out_mask=self.out_mask,
            out_mask_pattern=self.out_mask_pattern,
        )

        if y_true is None:
            return
        # A leaf missing on one side (a failed prediction misses them all)
        # scores as an empty string.
        fields, y_true, y_pred = align_leaves(y_true, y_pred, fill="")

        correct = []
        total = []
        intermediate_weights = []
        for yt, yp in zip(y_true, y_pred):
            y_true_tokens = nlp_utils.normalize_and_tokenize(str(yt))
            y_pred_tokens = nlp_utils.normalize_and_tokenize(str(yp))
            if not y_true_tokens and not y_pred_tokens:
                # Both blank: a correct (empty) answer, not a 0/0.
                y_true_tokens = y_pred_tokens = [""]
            common_tokens = set(y_true_tokens) & set(y_pred_tokens)
            union_tokens = set(y_true_tokens) | set(y_pred_tokens)
            correct.append(len(common_tokens))
            total.append(len(union_tokens))
            intermediate_weights.append(len(y_true_tokens))

        accumulate(
            self.state,
            fields,
            correct=correct,
            total=total,
            intermediate_weights=intermediate_weights,
        )

    def result(self):
        if self.state.get("correct") is None and self.state.get("total") is None:
            return 0.0
        correct = np.convert_to_tensor(self.state.get("correct"))
        total = np.convert_to_tensor(self.state.get("total"))

        # Keras/sklearn "micro": aggregate correct/total across all fields
        # first, *then* compute the ratio. Without this collapse, "micro"
        # would degenerate to a mean over per-field scores (i.e. macro).
        if self.average == "micro":
            correct = np.sum(correct)
            total = np.sum(total)

        score = np.divide_no_nan(correct, total)
        return aggregate_scores(
            score, self.average, self.axis, self.state.get("intermediate_weights")
        )

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        config = {
            "name": self.name,
            "average": self.average,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@synalinks_export("synalinks.metrics.BinaryAccuracy")
class BinaryAccuracy(Accuracy):
    """Computes accuracy on binary structures.

    Its output range is `[0, 1]`. It operates at a field level
    and can be used for **multi-class and multi-label classification**.

    Each field of `y_true` and `y_pred` should be a boolean or a float in
    `[0, 1]`. Float fields are thresholded against `threshold` to become
    binary.

    Per-field accuracy is `1` when the binarized values agree and `0`
    otherwise. Results are aggregated according to `average`.

    Example:

    ```python

    class MultiClassClassification(synalinks.DataModel):
        label_1: bool = synalinks.Field(
            description="The first label",
        )
        label_2: bool = synalinks.Field(
            description="The second label",
        )
        label_3: bool = synalinks.Field(
            description="The third label",
        )

    # OR you can also use floats between 0 and 1
    # The `Score`, enforce a float between 0.0 and 1.0 using constrained decoding

    class MultiClassClassification(synalinks.DataModel):
        label_1: synalinks.Score = synalinks.Field(
            description="The first label",
        )
        label_2: synalinks.Score = synalinks.Field(
            description="The second label",
        )
        label_3: synalinks.Score = synalinks.Field(
            description="The third label",
        )

    ```


    Compilation example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.BinaryAccuracy(),
        ],
    )
    ```

    Args:
        average (str): Type of averaging to be performed across per-class results
            in the multi-class case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute the metric globally by aggregating
            counts across all fields.
            If `"macro"`, compute the metric for each field, and return their
            unweighted mean.
            If `"weighted"`, compute the metric for each field, and return their
            mean weighted by support (the number of positive labels per field).
        threshold (float): (Optional) Float representing the threshold for deciding
            whether a value is `1` or `0`. Elements of `y_pred` and `y_true`
            greater than `threshold` are converted to `1`, the rest to `0`.
            Defaults to `0.5`.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
    """

    def __init__(
        self,
        average=None,
        threshold=0.5,
        name="binary_accuracy",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            average=average,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        if not isinstance(threshold, float):
            raise ValueError(
                "Invalid `threshold` argument value. "
                "It should be a Python float. "
                f"Received: threshold={threshold} "
                f"of type '{type(threshold)}'"
            )
        if threshold > 1.0 or threshold <= 0.0:
            raise ValueError(
                "Invalid `threshold` argument value. "
                "It should verify 0 < threshold <= 1. "
                f"Received: threshold={threshold}"
            )
        self.threshold = threshold

    async def update_state(self, y_true, y_pred):
        y_true, y_pred = apply_masks(
            y_true,
            y_pred,
            in_mask=self.in_mask,
            in_mask_pattern=self.in_mask_pattern,
            out_mask=self.out_mask,
            out_mask_pattern=self.out_mask_pattern,
        )

        def convert_to_binary(x):
            if isinstance(x, bool):
                return 1.0 if x is True else 0.0
            elif isinstance(x, float):
                return 1.0 if x > self.threshold else 0.0
            else:
                raise ValueError(
                    "All `y_true` and y_pred` fields should be booleans or floats. "
                    "Use `in_mask` or `out_mask` to remove the other fields."
                )

        if y_true is None:
            return
        # A leaf missing on one side (a failed prediction misses them all) is
        # None, which never equals a 0.0/1.0 leaf.
        fields, y_true, y_pred = align_leaves(y_true, y_pred)
        y_true = [None if v is None else convert_to_binary(v) for v in y_true]
        y_pred = [None if v is None else convert_to_binary(v) for v in y_pred]
        correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred)]
        total = [1.0] * len(fields)
        intermediate_weights = [yt or 0.0 for yt in y_true]

        accumulate(
            self.state,
            fields,
            correct=correct,
            total=total,
            intermediate_weights=intermediate_weights,
        )

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        config = {
            "threshold": self.threshold,
            "name": self.name,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@synalinks_export("synalinks.metrics.CategoricalAccuracy")
class CategoricalAccuracy(Accuracy):
    """Computes accuracy on list / categorical structures.

    Formula (per field, Jaccard index over label sets):

    ```python
    accuracy = |y_true_labels ∩ y_pred_labels| / |y_true_labels ∪ y_pred_labels|
    ```

    Its output range is `[0, 1]`. It operates at a label level
    and can be used for **classification** or **retrieval pipelines**.

    Unlike `Accuracy`, this metric considers each element of the list
    (or the string value) as **one label**, comparing label sets rather than
    tokenized words.

    If `labels` is provided, accumulation is performed per-label (sklearn-style):
    for each label `L`, a sample is "correct for L" when L's presence in
    `y_true` matches its presence in `y_pred`. This enables stable
    `macro`/`weighted` averaging across batches even when some labels are
    absent from a given sample, and lets `result()` return a `{label: score}`
    dict for `average=None`.

    If `labels` is `None`, a single global set-Jaccard is computed over the
    pooled label values; in that mode `average=None` returns one scalar
    (use `labels=...` for a per-label breakdown).

    Example:

    ```python

    # for single label classification

    class ListClassification(synalinks.DataModel):
        label: Literal["label", "label_1", "label_2"]

    # for multi label classification

    class ListClassification(synalinks.DataModel):
        labels: List[Literal["label", "label_1", "label_2"]]

    # or use it with retrieval pipelines, in that case make sure to mask
    # the correct fields.

    class AnswerWithReferences(synalinks.DataModel):
        sources: List[str]
        answer: str

    ```


    Compilation example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.CategoricalAccuracy(),
        ],
    )
    ```

    Args:
        average (str): Type of averaging to be performed across per-field results
            in the multi-field case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each field.
            If `"micro"`, compute the metric globally by aggregating
            label counts across all fields.
            If `"macro"`, compute the metric for each field, and return their
            unweighted mean.
            If `"weighted"`, compute the metric for each field, and return their
            mean weighted by support (the number of true labels per field).
        labels (list): (Optional) Explicit list of label names to track.
            When provided, accumulation is per-label across all batches and
            `result()` returns a `{label: score}` dict for `average=None`.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): (Optional) Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
    """

    def __init__(
        self,
        average=None,
        labels=None,
        name="categorical_accuracy",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            average=average,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        if labels is not None:
            labels = [str(label) for label in labels]
        self.labels = labels

    async def update_state(self, y_true, y_pred):
        y_true, y_pred = apply_masks(
            y_true,
            y_pred,
            in_mask=self.in_mask,
            in_mask_pattern=self.in_mask_pattern,
            out_mask=self.out_mask,
            out_mask_pattern=self.out_mask_pattern,
        )

        if y_true is None:
            return
        y_true = tree.flatten(tree.map_structure(lambda x: x, y_true.get_json()))
        # A failed prediction (`y_pred is None`) scores as an empty one.
        y_pred = (
            []
            if y_pred is None
            else tree.flatten(tree.map_structure(lambda x: x, y_pred.get_json()))
        )

        correct = []
        total = []
        intermediate_weights = []

        fields = self.labels if self.labels is not None else ["labels"]
        if self.labels is not None:
            y_true_set = {str(v) for v in y_true}
            y_pred_set = {str(v) for v in y_pred}
            for label in self.labels:
                t = label in y_true_set
                p = label in y_pred_set
                correct.append(1 if t == p else 0)
                total.append(1)
                intermediate_weights.append(1 if t else 0)
        else:
            # Set-Jaccard over the full pool of labels: position-independent,
            # so that `["a","b"]` vs `["b","a"]` scores 1.0. Produces a single
            # entry per call; per-label tracking requires `labels=...`.
            y_true_labels = [str(v) for v in y_true]
            y_pred_labels = [str(v) for v in y_pred]
            common_labels = set(y_true_labels) & set(y_pred_labels)
            union_labels = set(y_true_labels) | set(y_pred_labels)
            correct.append(len(common_labels))
            total.append(len(union_labels))
            intermediate_weights.append(len(y_true_labels))

        accumulate(
            self.state,
            fields,
            correct=correct,
            total=total,
            intermediate_weights=intermediate_weights,
        )

    def result(self):
        res = super().result()
        if self.labels is not None and self.average is None:
            # One label (or no update yet) gives a single float.
            scores = res if isinstance(res, list) else [res] * len(self.labels)
            return dict(zip(self.labels, scores))
        return res

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        config = {
            "labels": list(self.labels) if self.labels is not None else None,
            "name": self.name,
        }
        base_config = super().get_config()
        return {**base_config, **config}
