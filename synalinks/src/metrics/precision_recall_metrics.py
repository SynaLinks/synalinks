# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.f_score_metrics import BinaryFBetaScore
from synalinks.src.metrics.f_score_metrics import CategoricalFBetaScore
from synalinks.src.metrics.f_score_metrics import FBetaScore


@synalinks_export("synalinks.metrics.Precision")
class Precision(FBetaScore):
    """Computes token-level precision for LM string outputs.

    Formula (per field, SQuAD-style multiset over normalized tokens):

    ```python
    precision = |Counter(y_true_tokens) ∩ Counter(y_pred_tokens)| / |y_pred_tokens|
    ```

    Mirrors `F1Score`: tokenization, masking and `average` modes behave
    identically — only the result formula differs. Use this when you want
    to report precision as a separate signal alongside `Recall` and
    `F1Score`.

    Args:
        average (str): Type of averaging across per-field results.
            One of `None`, `"micro"`, `"macro"`, `"weighted"`.
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
        name="precision",
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
        self._formula = "precision"

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


@synalinks_export("synalinks.metrics.Recall")
class Recall(FBetaScore):
    """Computes token-level recall for LM string outputs.

    Formula (per field, SQuAD-style multiset over normalized tokens):

    ```python
    recall = |Counter(y_true_tokens) ∩ Counter(y_pred_tokens)| / |y_true_tokens|
    ```

    Mirrors `F1Score`: tokenization, masking and `average` modes behave
    identically — only the result formula differs.

    Args:
        average (str): Type of averaging across per-field results.
            One of `None`, `"micro"`, `"macro"`, `"weighted"`.
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
        name="recall",
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
        self._formula = "recall"

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


@synalinks_export("synalinks.metrics.BinaryPrecision")
class BinaryPrecision(BinaryFBetaScore):
    """Computes precision on binary structures.

    Mirrors `BinaryF1Score`. Each field of `y_true` and `y_pred` should be
    a boolean or a float in `[0, 1]`; floats are thresholded against
    `threshold`. Per-field precision is `TP / (TP + FP)`, aggregated via
    `average`.

    Args:
        average (str): One of `None`, `"micro"`, `"macro"`, `"weighted"`.
        threshold (float): Threshold for deciding whether a float value is
            `1` or `0`. Defaults to `0.5`.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern.
        out_mask_pattern (str): (Optional) Regex pattern.
    """

    def __init__(
        self,
        average=None,
        threshold=0.5,
        name="binary_precision",
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
        self._formula = "precision"

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


@synalinks_export("synalinks.metrics.BinaryRecall")
class BinaryRecall(BinaryFBetaScore):
    """Computes recall on binary structures.

    Mirrors `BinaryF1Score`. Per-field recall is `TP / (TP + FN)`,
    aggregated via `average`.

    Args:
        average (str): One of `None`, `"micro"`, `"macro"`, `"weighted"`.
        threshold (float): Threshold for binarizing float fields.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep to compute the metric.
        out_mask (list): (Optional) list of keys to remove to compute the metric.
        in_mask_pattern (str): (Optional) Regex pattern.
        out_mask_pattern (str): (Optional) Regex pattern.
    """

    def __init__(
        self,
        average=None,
        threshold=0.5,
        name="binary_recall",
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
        self._formula = "recall"

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


@synalinks_export("synalinks.metrics.CategoricalPrecision")
class CategoricalPrecision(CategoricalFBetaScore):
    """Computes precision on categorical (list / label) structures.

    Mirrors `CategoricalF1Score`. Supports the optional `labels=` parameter:
    when provided, accumulation is per-label (sklearn-style) and
    `result()` returns a `{label: precision}` dict for `average=None`.

    Args:
        average (str): One of `None`, `"micro"`, `"macro"`, `"weighted"`.
        labels (list): (Optional) Explicit list of label names to track.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep.
        out_mask (list): (Optional) list of keys to remove.
        in_mask_pattern (str): (Optional) Regex pattern.
        out_mask_pattern (str): (Optional) Regex pattern.
    """

    def __init__(
        self,
        average=None,
        labels=None,
        name="categorical_precision",
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
        self._formula = "precision"

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


@synalinks_export("synalinks.metrics.CategoricalRecall")
class CategoricalRecall(CategoricalFBetaScore):
    """Computes recall on categorical (list / label) structures.

    Mirrors `CategoricalF1Score`. Supports the optional `labels=` parameter:
    when provided, accumulation is per-label (sklearn-style) and
    `result()` returns a `{label: recall}` dict for `average=None`.

    Args:
        average (str): One of `None`, `"micro"`, `"macro"`, `"weighted"`.
        labels (list): (Optional) Explicit list of label names to track.
        name (str): (Optional) string name of the metric instance.
        in_mask (list): (Optional) list of keys to keep.
        out_mask (list): (Optional) list of keys to remove.
        in_mask_pattern (str): (Optional) Regex pattern.
        out_mask_pattern (str): (Optional) Regex pattern.
    """

    def __init__(
        self,
        average=None,
        labels=None,
        name="categorical_recall",
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
        self._formula = "recall"

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config
