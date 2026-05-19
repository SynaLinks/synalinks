# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Metric-shaped data models — discretized scales the LM can pick from."""

from enum import Enum

from synalinks.src.api_export import synalinks_export


@synalinks_export(
    [
        "synalinks.backend.Score",
        "synalinks.Score",
    ]
)
class Score(float, Enum):
    """A discretized confidence score on an 11-level scale from 0.0 to 1.0.

    Use `Score` as the type of a `DataModel` field when you want the
    language model to pick a confidence level from a fixed set of named
    values rather than to emit an arbitrary float. Because `Score` is
    both a `float` and an `Enum`, the JSON schema constrains the model
    to one of the eleven labels, while downstream Python code can use
    the value in arithmetic directly (e.g. `0.95` is `Score.GOOD + 0.05`).

    The labels:

    | Name           | Value |
    | -------------- | ----- |
    | `VERY_BAD`     | 0.0   |
    | `POOR`         | 0.1   |
    | `BELOW_AVERAGE`| 0.2   |
    | `LOW_AVERAGE`  | 0.3   |
    | `MEDIUM_LOW`   | 0.4   |
    | `MEDIUM`       | 0.5   |
    | `MEDIUM_HIGH`  | 0.6   |
    | `ABOVE_AVERAGE`| 0.7   |
    | `HIGH_AVERAGE` | 0.8   |
    | `GOOD`         | 0.9   |
    | `VERY_GOOD`    | 1.0   |

    Example:

    ```python
    import synalinks

    class Sentiment(synalinks.DataModel):
        joy: synalinks.Score = synalinks.Field(
            description="How strongly the text expresses joy",
        )
        anger: synalinks.Score = synalinks.Field(
            description="How strongly the text expresses anger",
        )

    # Score values are real floats, usable in arithmetic.
    blended = synalinks.Score.GOOD + 0.05  # approx 0.95
    ```

    See `synalinks/src/metrics/f_score_metrics.py` and
    `examples/19_multi_objective_lm_selection.py` for end-to-end usage
    inside metrics and multi-label classification.
    """

    VERY_BAD = 0.0
    POOR = 0.1
    BELOW_AVERAGE = 0.2
    LOW_AVERAGE = 0.3
    MEDIUM_LOW = 0.4
    MEDIUM = 0.5
    MEDIUM_HIGH = 0.6
    ABOVE_AVERAGE = 0.7
    HIGH_AVERAGE = 0.8
    GOOD = 0.9
    VERY_GOOD = 1.0
