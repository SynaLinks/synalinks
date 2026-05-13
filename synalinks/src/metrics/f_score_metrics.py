# Modified from: keras/src/metrics/f_score_metrics.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from collections import Counter
from typing import List
from typing import Optional

from synalinks.src import backend
from synalinks.src import ops
from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend.common import numpy as np
from synalinks.src.metrics.metric import Metric
from synalinks.src.utils import nlp_utils


class FBetaState(DataModel):
    true_positives: Optional[List[float]] = None
    false_positives: Optional[List[float]] = None
    false_negatives: Optional[List[float]] = None
    intermediate_weights: Optional[List[float]] = None


@synalinks_export("synalinks.metrics.FBetaScore")
class FBetaScore(Metric):
    """Computes F-Beta score.

    Formula:

    ```python
    b2 = beta ** 2
    f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
    ```

    This is the weighted harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It operates at a word level
    and can be used for **QA systems**.

    If `y_true` and `y_pred` contains multiple fields
    The JSON object's fields are flattened and the score
    computed for each one independently.

    Args:
        average (str): Type of averaging to be performed across per-field results
            in the multi-field case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        beta (float): Determines the weight of given to recall
            in the harmonic mean between precision and recall (see pseudocode
            equation above). Defaults to `1`.
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
        beta=1.0,
        name="fbeta_score",
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

        if not isinstance(beta, float):
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be a Python float. "
                f"Received: beta={beta} of type '{type(beta)}'"
            )
        self.state = self.add_variable(
            data_model=FBetaState,
            name="state_" + self.name,
        )
        self.average = average
        self.beta = beta
        self.axis = None
        if self.average != "micro":
            self.axis = 0
        # Subclasses (Precision, Recall) override this to switch the result
        # formula while reusing TP/FP/FN state and update_state.
        self._formula = "fbeta"

    async def update_state(self, y_true, y_pred):
        y_pred = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_pred)
        y_true = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_true)

        if self.in_mask or self.in_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern),
                y_true,
            )
        if self.out_mask or self.out_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern),
                y_true,
            )

        y_true = tree.flatten(tree.map_structure(lambda x: str(x), y_true.get_json()))
        y_pred = tree.flatten(tree.map_structure(lambda x: str(x), y_pred.get_json()))

        true_positives = []
        false_positives = []
        false_negatives = []
        intermediate_weights = []
        # For each field of y_true and y_pred. SQuAD-style multiset (Counter)
        # intersection — needed so identical strings with repeated tokens
        # score 1.0.
        for yt, yp in zip(y_true, y_pred):
            y_true_tokens = nlp_utils.normalize_and_tokenize(str(yt))
            y_pred_tokens = nlp_utils.normalize_and_tokenize(str(yp))
            num_common = sum(
                (Counter(y_true_tokens) & Counter(y_pred_tokens)).values()
            )
            true_positives.append(num_common)
            false_positives.append(len(y_pred_tokens) - num_common)
            false_negatives.append(len(y_true_tokens) - num_common)
            intermediate_weights.append(len(y_true_tokens))

        true_positives = np.convert_to_numpy(true_positives)
        false_positives = np.convert_to_numpy(false_positives)
        false_negatives = np.convert_to_numpy(false_negatives)
        intermediate_weights = np.convert_to_numpy(intermediate_weights)

        current_true_positives = self.state.get("true_positives")
        if current_true_positives:
            true_positives = np.add(current_true_positives, true_positives)

        current_false_positives = self.state.get("false_positives")
        if current_false_positives:
            false_positives = np.add(current_false_positives, false_positives)

        current_false_negatives = self.state.get("false_negatives")
        if current_false_negatives:
            false_negatives = np.add(current_false_negatives, false_negatives)

        current_intermediate_weights = self.state.get("intermediate_weights")
        if current_intermediate_weights:
            intermediate_weights = np.add(
                current_intermediate_weights, intermediate_weights
            )

        self.state.update(
            {
                "true_positives": true_positives.tolist(),
                "false_positives": false_positives.tolist(),
                "false_negatives": false_negatives.tolist(),
                "intermediate_weights": intermediate_weights.tolist(),
            }
        )

    def result(self):
        if (
            self.state.get("true_positives") is None
            and self.state.get("false_positives") is None
            and self.state.get("false_negatives") is None
        ):
            return 0.0
        tp = np.convert_to_tensor(self.state.get("true_positives"))
        fp = np.convert_to_tensor(self.state.get("false_positives"))
        fn = np.convert_to_tensor(self.state.get("false_negatives"))

        # Keras/sklearn "micro": aggregate TP/FP/FN across all fields first,
        # *then* compute precision/recall. Without this collapse, "micro"
        # would degenerate to a mean over per-field scores (i.e. macro).
        if self.average == "micro":
            tp = np.sum(tp)
            fp = np.sum(fp)
            fn = np.sum(fn)

        precision = np.convert_to_tensor(
            np.divide(tp, np.add(tp, fp) + backend.epsilon())
        )
        recall = np.convert_to_tensor(
            np.divide(tp, np.add(tp, fn) + backend.epsilon())
        )

        formula = getattr(self, "_formula", "fbeta")
        if formula == "precision":
            score = precision
        elif formula == "recall":
            score = recall
        else:
            mul_value = precision * recall
            add_value = ((self.beta**2) * precision) + recall
            mean = np.divide(mul_value, add_value + backend.epsilon())
            score = mean * (1 + (self.beta**2))

        return self._aggregate(score)

    def _aggregate(self, score):
        """Apply `average` reduction over per-field scores."""
        score = np.convert_to_tensor(score)
        if self.average == "weighted":
            intermediate_weights = self.state.get("intermediate_weights")
            weights = np.divide(
                intermediate_weights,
                np.sum(intermediate_weights) + backend.epsilon(),
            )
            score = np.sum(score * weights)
        elif self.average is not None:  # [micro, macro]
            score = np.mean(score, self.axis)
        try:
            return float(score)
        except Exception:
            return list(score)

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        config = {
            "name": self.name,
            "average": self.average,
            "beta": self.beta,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@synalinks_export("synalinks.metrics.F1Score")
class F1Score(FBetaScore):
    """Computes F-1 Score.

    Formula:

    ```python
    f1_score = 2 * (precision * recall) / (precision + recall)
    ```

    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It operates at a word level
    and can be used for **QA systems**.

    If `y_true` and `y_pred` contains multiple fields
    The JSON object's fields are flattened and the score
    computed for each one independently before being averaged.

    Args:
        average (str): Type of averaging to be performed across per-field results
            in the multi-field case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
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
        name="f1_score",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


@synalinks_export("synalinks.metrics.BinaryFBetaScore")
class BinaryFBetaScore(FBetaScore):
    """Computes F-Beta score on binary structures.

    Formula:

    ```python
    b2 = beta ** 2
    f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
    ```

    This is the weighted harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It operates at a field level
    and can be used for **multi-class and multi-label classification**.

    Each field of `y_true` and `y_pred` should be booleans or floats between [0, 1].
    If the fields are floats, it uses the threshold parameter for deciding
    if the values are 0 or 1.

    Example:

    ```

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

    Args:
        average (str): Type of averaging to be performed across per-class results
            in the multi-class case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        beta (float): Determines the weight of given to recall
            in the harmonic mean between precision and recall (see pseudocode
            equation above). Defaults to `1`.
        threshold (float): (Optional) Float representing the threshold for deciding
            whether prediction values are 1 or 0. Elements of `y_pred` and `y_true`
            greater than `threshold` are converted to be 1, and the rest 0.
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
        beta=1.0,
        threshold=0.5,
        name="binary_fbeta_score",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            average=average,
            beta=beta,
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
        y_pred = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_pred)
        y_true = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_true)

        if self.in_mask or self.in_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern),
                y_true,
            )
        if self.out_mask or self.out_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern),
                y_true,
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

        y_true = tree.flatten(
            tree.map_structure(lambda x: convert_to_binary(x), y_true.get_json())
        )
        y_pred = tree.flatten(
            tree.map_structure(lambda x: convert_to_binary(x), y_pred.get_json())
        )
        y_true = np.convert_to_tensor(y_true)
        y_pred = np.convert_to_tensor(y_pred)

        true_positives = y_pred * y_true
        false_positives = y_pred * (1 - y_true)
        false_negatives = (1 - y_pred) * y_true
        intermediate_weights = y_true

        current_true_positives = self.state.get("true_positives")
        if current_true_positives:
            true_positives = np.add(current_true_positives, true_positives)

        current_false_positives = self.state.get("false_positives")
        if current_false_positives:
            false_positives = np.add(current_false_positives, false_positives)

        current_false_negatives = self.state.get("false_negatives")
        if current_false_negatives:
            false_negatives = np.add(current_false_negatives, false_negatives)

        current_intermediate_weights = self.state.get("intermediate_weights")
        if current_intermediate_weights:
            intermediate_weights = np.add(
                current_intermediate_weights, intermediate_weights
            )

        self.state.update(
            {
                "true_positives": true_positives.tolist(),
                "false_positives": false_positives.tolist(),
                "false_negatives": false_negatives.tolist(),
                "intermediate_weights": intermediate_weights.tolist(),
            }
        )

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        config = {
            "beta": self.beta,
            "threshold": self.threshold,
            "name": self.name,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@synalinks_export("synalinks.metrics.BinaryF1Score")
class BinaryF1Score(BinaryFBetaScore):
    """Computes F-1 Score on binary structures.

    Formula:

    ```python
    f1_score = 2 * (precision * recall) / (precision + recall)
    ```

    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It operates at a field level
    and can be used for **multi-class and multi-label classification**.

    Each field of `y_true` and `y_pred` should booleans or floats between [0, 1].
    If the fields are floats, it uses the threshold for deciding
    if the values are 0 or 1.

    Args:
        average (str): Type of averaging to be performed across per-class results
            in the multi-class case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each class.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        threshold (float): (Optional) Float representing the threshold for deciding
            whether prediction values are 1 or 0. Elements of `y_pred` and `y_true`
            greater than `threshold` are converted to be 1, and the rest 0.
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
        name="binary_f1_score",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


@synalinks_export(
    [
        "synalinks.metrics.CategoricalFBetaScore",
        "synalinks.metrics.ListFBetaScore",
    ]
)
class CategoricalFBetaScore(FBetaScore):
    """Computes F-Beta score on categorical (list / label) structures.

    Formula:

    ```python
    b2 = beta ** 2
    f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
    ```

    This is the weighted harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It operates at a label level
    and can be used for **classification** or **retrieval pipelines**.

    The difference between this metric and `F1Score` is that this one considers
    each element of the list (or the string value) as **one label**, comparing
    label sets rather than tokenized words.

    If `labels` is provided, accumulation is performed per-label (sklearn-style):
    for each label `L`, `tp[L] += 1` when `L` appears in both `y_true` and
    `y_pred`, `fp[L] += 1` when it appears only in `y_pred`, `fn[L] += 1` when
    it appears only in `y_true`. This enables stable `macro`/`weighted`
    averaging across batches even when some labels are absent from a given
    sample, and lets `result()` return a `{label: score}` dict when
    `average=None`.

    If `labels` is `None`, a single global set-based TP/FP/FN is computed
    over the pooled label values; in that mode `average=None` returns one
    scalar (use `labels=...` for a per-label breakdown).

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

    Args:
        average (str): Type of averaging to be performed across per-field results
            in the multi-field case.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
            If `None`, no averaging is performed and `result()` will return
            the score for each field.
            If `"micro"`, compute metrics globally by counting the total
            true positives, false negatives and false positives.
            If `"macro"`, compute metrics for each label,
            and return their unweighted mean.
            This does not take label imbalance into account.
            If `"weighted"`, compute metrics for each label,
            and return their average weighted by support
            (the number of true instances for each label).
            This alters `"macro"` to account for label imbalance.
            It can result in an score that is not between precision and recall.
        beta (float): Determines the weight of given to recall
            in the harmonic mean between precision and recall. Defaults to `1`.
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
        beta=1.0,
        labels=None,
        name="categorical_fbeta_score",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            average=average,
            beta=beta,
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
        y_pred = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_pred)
        y_true = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_true)

        if self.in_mask or self.in_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: x.in_mask(mask=self.in_mask, pattern=self.in_mask_pattern),
                y_true,
            )
        if self.out_mask or self.out_mask_pattern:
            y_pred = tree.map_structure(
                lambda x: x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern),
                y_pred,
            )
            y_true = tree.map_structure(
                lambda x: x.out_mask(mask=self.out_mask, pattern=self.out_mask_pattern),
                y_true,
            )

        y_true = tree.flatten(tree.map_structure(lambda x: x, y_true.get_json()))
        y_pred = tree.flatten(tree.map_structure(lambda x: x, y_pred.get_json()))

        true_positives = []
        false_positives = []
        false_negatives = []
        intermediate_weights = []

        if self.labels is not None:
            y_true_set = {str(v) for v in y_true}
            y_pred_set = {str(v) for v in y_pred}
            for label in self.labels:
                t = label in y_true_set
                p = label in y_pred_set
                true_positives.append(1 if (t and p) else 0)
                false_positives.append(1 if (p and not t) else 0)
                false_negatives.append(1 if (t and not p) else 0)
                intermediate_weights.append(1 if t else 0)
        else:
            # Set-based TP/FP/FN over the full pool of labels:
            # position-independent, so that `["a","b"]` vs `["b","a"]` scores
            # 1.0. Produces a single entry per call; per-label tracking
            # requires `labels=...`.
            y_true_labels = [str(v) for v in y_true]
            y_pred_labels = [str(v) for v in y_pred]
            common_labels = set(y_true_labels) & set(y_pred_labels)
            true_positives.append(len(common_labels))
            false_positives.append(len(set(y_pred_labels)) - len(common_labels))
            false_negatives.append(len(set(y_true_labels)) - len(common_labels))
            intermediate_weights.append(len(y_true_labels))

        true_positives = np.convert_to_numpy(true_positives)
        false_positives = np.convert_to_numpy(false_positives)
        false_negatives = np.convert_to_numpy(false_negatives)
        intermediate_weights = np.convert_to_numpy(intermediate_weights)

        current_true_positives = self.state.get("true_positives")
        if current_true_positives:
            true_positives = np.add(current_true_positives, true_positives)

        current_false_positives = self.state.get("false_positives")
        if current_false_positives:
            false_positives = np.add(current_false_positives, false_positives)

        current_false_negatives = self.state.get("false_negatives")
        if current_false_negatives:
            false_negatives = np.add(current_false_negatives, false_negatives)

        current_intermediate_weights = self.state.get("intermediate_weights")
        if current_intermediate_weights:
            intermediate_weights = np.add(
                current_intermediate_weights, intermediate_weights
            )

        self.state.update(
            {
                "true_positives": true_positives.tolist(),
                "false_positives": false_positives.tolist(),
                "false_negatives": false_negatives.tolist(),
                "intermediate_weights": intermediate_weights.tolist(),
            }
        )

    def result(self):
        res = super().result()
        if (
            self.labels is not None
            and self.average is None
            and isinstance(res, list)
        ):
            return {label: score for label, score in zip(self.labels, res)}
        return res

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        config = {
            "beta": self.beta,
            "labels": list(self.labels) if self.labels is not None else None,
            "name": self.name,
        }
        base_config = super().get_config()
        return {**base_config, **config}


# Deprecated alias for backward compatibility.
ListFBetaScore = CategoricalFBetaScore


@synalinks_export(
    [
        "synalinks.metrics.CategoricalF1Score",
        "synalinks.metrics.ListF1Score",
    ]
)
class CategoricalF1Score(CategoricalFBetaScore):
    """Computes F-1 Score on categorical (list / label) structures.

    Formula:
    ```python
        f1_score = 2 * (precision * recall) / (precision + recall)
    ```

    This is the harmonic mean of precision and recall.
    Its output range is `[0, 1]`. It operates at a label level
    and can be used for **classification** or **retrieval pipelines**.

    The difference between this metric and `F1Score` is that this one considers
    each element of the list (or the string value) as **one label**.

    If `labels` is provided, accumulation is performed per-label (sklearn-style)
    and `result()` returns a `{label: score}` dict for `average=None`. See
    `CategoricalFBetaScore` for details.

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

    Args:
        average (str): Type of averaging to be performed.
            Acceptable values are `None`, `"micro"`, `"macro"` and
            `"weighted"`. Defaults to `None`.
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
        name="categorical_f1_score",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            labels=labels,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


# Deprecated alias for backward compatibility.
ListF1Score = CategoricalF1Score
