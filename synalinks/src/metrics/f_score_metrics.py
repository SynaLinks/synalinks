# Modified from: keras/src/metrics/f_score_metrics.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

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
    """

    def __init__(
        self,
        average=None,
        beta=1.0,
        name="fbeta_score",
        in_mask=None,
        out_mask=None,
    ):
        super().__init__(
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
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

    async def update_state(self, y_true, y_pred):
        y_pred = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_pred)
        y_true = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_true)

        if self.in_mask:
            y_pred = tree.map_structure(lambda x: x.in_mask(mask=self.in_mask), y_pred)
            y_true = tree.map_structure(lambda x: x.in_mask(mask=self.in_mask), y_true)
        if self.out_mask:
            y_pred = tree.map_structure(lambda x: x.out_mask(mask=self.out_mask), y_pred)
            y_true = tree.map_structure(lambda x: x.out_mask(mask=self.out_mask), y_true)

        y_true = tree.flatten(tree.map_structure(lambda x: str(x), y_true.get_json()))
        y_pred = tree.flatten(tree.map_structure(lambda x: str(x), y_pred.get_json()))

        true_positives = []
        false_positives = []
        false_negatives = []
        intermediate_weights = []
        # For each field of y_true and y_pred
        for yt, yp in zip(y_true, y_pred):
            y_true_tokens = nlp_utils.normalize_and_tokenize(str(yt))
            y_pred_tokens = nlp_utils.normalize_and_tokenize(str(yp))
            common_tokens = set(y_true_tokens) & set(y_pred_tokens)
            true_positives.append(len(common_tokens))
            false_positives.append(len(y_pred_tokens) - len(common_tokens))
            false_negatives.append(len(y_true_tokens) - len(common_tokens))
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
        precision = np.divide(
            self.state.get("true_positives"),
            np.add(
                self.state.get("true_positives"),
                self.state.get("false_positives"),
            )
            + backend.epsilon(),
        )
        recall = np.divide(
            self.state.get("true_positives"),
            np.add(
                self.state.get("true_positives"),
                self.state.get("false_negatives"),
            )
            + backend.epsilon(),
        )
        precision = np.convert_to_tensor(precision)
        recall = np.convert_to_tensor(recall)

        mul_value = precision * recall
        add_value = ((self.beta**2) * precision) + recall
        mean = np.divide(mul_value, add_value + backend.epsilon())
        f1_score = mean * (1 + (self.beta**2))
        if self.average == "weighted":
            intermediate_weights = self.state.get("intermediate_weights")
            weights = np.divide(
                intermediate_weights,
                np.sum(intermediate_weights) + backend.epsilon(),
            )
            f1_score = np.sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = np.mean(f1_score, self.axis)

        try:
            return float(f1_score)
        except Exception:
            return list(f1_score)

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        config = {
            "name": self.name,
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
    """

    def __init__(
        self,
        average=None,
        name="f1_score",
        in_mask=None,
        out_mask=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
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
        beta (float): Determines the weight of given to recall
            in the harmonic mean between precision and recall (see pseudocode
            equation above). Defaults to `1`.
        threshold (float): (Optional) Float representing the threshold for deciding
            whether prediction values are 1 or 0. Elements of `y_pred` and `y_true`
            greater than `threshold` are converted to be 1, and the rest 0.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
    """

    def __init__(
        self,
        average=None,
        beta=1.0,
        threshold=0.5,
        name="binary_fbeta_score",
        in_mask=None,
        out_mask=None,
    ):
        super().__init__(
            average=average,
            beta=beta,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
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

        if self.in_mask:
            y_pred = tree.map_structure(lambda x: x.in_mask(mask=self.in_mask), y_pred)
            y_true = tree.map_structure(lambda x: x.in_mask(mask=self.in_mask), y_true)
        if self.out_mask:
            y_pred = tree.map_structure(lambda x: x.out_mask(mask=self.out_mask), y_pred)
            y_true = tree.map_structure(lambda x: x.out_mask(mask=self.out_mask), y_true)

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
    """

    def __init__(
        self,
        average=None,
        threshold=0.5,
        name="binary_f1_score",
        in_mask=None,
        out_mask=None,
    ):
        super().__init__(
            average=average,
            beta=1.0,
            threshold=threshold,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
        )

    def get_config(self):
        """Return the serializable config of the metric.

        Returns:
            (dict): The config dict.
        """
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
