# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Metric-shaped data models: discretized scales the LM can pick from."""

from enum import Enum

from synalinks.src.api_export import synalinks_export
from synalinks.src.saving.object_registration import get_registered_name
from synalinks.src.saving.object_registration import get_registered_object


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


@synalinks_export(
    [
        "synalinks.backend.FineScore",
        "synalinks.FineScore",
    ]
)
class FineScore(float, Enum):
    """A discretized confidence score on a 21-level scale from 0.0 to 1.0.

    `FineScore` is the finer-grained sibling of `Score`: it steps by 0.05
    instead of 0.1, giving the language model twice the resolution when it
    picks a confidence level. Every `Score` value is also a `FineScore`
    value (with the same name), with one extra level slotted between each
    pair. Like `Score`, it is both a `float` and an `Enum`, so the JSON
    schema constrains the model to one of the fixed values while Python
    code can use the value in arithmetic directly.

    Reach for `FineScore` when the 11 levels of `Score` are too coarse
    (e.g. ranking many candidates that would otherwise tie), and stay with
    `Score` when you want the model to commit to broader buckets.

    The labels:

    | Name                   | Value |
    | ---------------------- | ----- |
    | `VERY_BAD`             | 0.00  |
    | `BAD`                  | 0.05  |
    | `POOR`                 | 0.10  |
    | `VERY_LOW`             | 0.15  |
    | `BELOW_AVERAGE`        | 0.20  |
    | `LOW`                  | 0.25  |
    | `LOW_AVERAGE`          | 0.30  |
    | `SLIGHTLY_LOW`         | 0.35  |
    | `MEDIUM_LOW`           | 0.40  |
    | `SLIGHTLY_BELOW_MEDIUM`| 0.45  |
    | `MEDIUM`               | 0.50  |
    | `SLIGHTLY_ABOVE_MEDIUM`| 0.55  |
    | `MEDIUM_HIGH`          | 0.60  |
    | `SLIGHTLY_HIGH`        | 0.65  |
    | `ABOVE_AVERAGE`        | 0.70  |
    | `HIGH`                 | 0.75  |
    | `HIGH_AVERAGE`         | 0.80  |
    | `VERY_HIGH`            | 0.85  |
    | `GOOD`                 | 0.90  |
    | `EXCELLENT`            | 0.95  |
    | `VERY_GOOD`            | 1.00  |

    Example:

    ```python
    import synalinks

    class Relevance(synalinks.DataModel):
        relevance: synalinks.FineScore = synalinks.Field(
            description="How relevant the document is to the query",
        )

    # FineScore values are real floats, usable in arithmetic.
    assert synalinks.FineScore.HIGH == 0.75
    assert synalinks.FineScore.GOOD == synalinks.Score.GOOD
    ```
    """

    VERY_BAD = 0.0
    BAD = 0.05
    POOR = 0.1
    VERY_LOW = 0.15
    BELOW_AVERAGE = 0.2
    LOW = 0.25
    LOW_AVERAGE = 0.3
    SLIGHTLY_LOW = 0.35
    MEDIUM_LOW = 0.4
    SLIGHTLY_BELOW_MEDIUM = 0.45
    MEDIUM = 0.5
    SLIGHTLY_ABOVE_MEDIUM = 0.55
    MEDIUM_HIGH = 0.6
    SLIGHTLY_HIGH = 0.65
    ABOVE_AVERAGE = 0.7
    HIGH = 0.75
    HIGH_AVERAGE = 0.8
    VERY_HIGH = 0.85
    GOOD = 0.9
    EXCELLENT = 0.95
    VERY_GOOD = 1.0


@synalinks_export(
    [
        "synalinks.backend.Rating",
        "synalinks.Rating",
    ]
)
class Rating(int, Enum):
    """A discretized rating on a 5-level integer scale from 1 to 5.

    `Rating` is the integer, Likert-style counterpart of `Score`: instead
    of a float between 0.0 and 1.0, the language model picks one of five
    whole numbers, 1 (worst) to 5 (best). Because `Rating` is both an `int`
    and an `Enum`, the JSON schema constrains the model to exactly those
    five values while Python code can use the value in arithmetic
    directly (e.g. averaging several ratings, or dividing by 5 to get a
    score between 0.2 and 1.0).

    Reach for `Rating` when you want the familiar "rate this from 1 to 5"
    framing (star ratings, Likert surveys, rubric grades), and for `Score`
    or `FineScore` when you want a normalized float.

    The labels are the spelled-out numbers `ONE` (1) through `FIVE` (5).

    Example:

    ```python
    import synalinks

    class Review(synalinks.DataModel):
        rating: synalinks.Rating = synalinks.Field(
            description="Overall quality of the answer, from 1 (worst) to 5 (best)",
        )

    # Rating values are real ints, usable in arithmetic.
    assert synalinks.Rating.FOUR == 4
    normalized = synalinks.Rating.FOUR / 5  # 0.8
    ```
    """

    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


@synalinks_export(
    [
        "synalinks.backend.Rating10",
        "synalinks.Rating10",
    ]
)
class Rating10(int, Enum):
    """A discretized rating on a 10-level integer scale from 1 to 10.

    `Rating10` is the 10-point variant of `Rating`: the language model
    picks one whole number from 1 (worst) to 10 (best). Like `Rating`, it
    is both an `int` and an `Enum`, so the JSON schema constrains the model
    to exactly those ten values while Python code can use the value in
    arithmetic directly (e.g. divide by 10 to get a score between 0.1 and
    1.0).

    Reach for `Rating10` when the 5 levels of `Rating` are too coarse but
    you still want the "rate this out of 10" framing; use `Rating20` for
    an even finer integer scale, and `Score` / `FineScore` when you want a
    normalized float instead.

    The labels are the spelled-out numbers `ONE` (1) through `TEN` (10).

    Example:

    ```python
    import synalinks

    class Review(synalinks.DataModel):
        rating: synalinks.Rating10 = synalinks.Field(
            description="Overall quality of the answer, from 1 (worst) to 10 (best)",
        )

    # Rating10 values are real ints, usable in arithmetic.
    assert synalinks.Rating10.SEVEN == 7
    normalized = synalinks.Rating10.SEVEN / 10  # 0.7
    ```
    """

    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10


@synalinks_export(
    [
        "synalinks.backend.Rating20",
        "synalinks.Rating20",
    ]
)
class Rating20(int, Enum):
    """A discretized rating on a 20-level integer scale from 1 to 20.

    `Rating20` is the 20-point variant of `Rating`: the language model
    picks one whole number from 1 (worst) to 20 (best). Like `Rating`, it
    is both an `int` and an `Enum`, so the JSON schema constrains the model
    to exactly those twenty values while Python code can use the value in
    arithmetic directly (e.g. divide by 20 to get a score between 0.05 and
    1.0).

    Reach for `Rating20` when you need fine integer resolution (e.g. a
    "grade out of 20" rubric, or ranking many candidates that would tie on
    a coarser scale); use `Rating` or `Rating10` for broader buckets, and
    `Score` / `FineScore` when you want a normalized float instead.

    The labels are the spelled-out numbers `ONE` (1) through `TWENTY` (20).

    Example:

    ```python
    import synalinks

    class Grade(synalinks.DataModel):
        grade: synalinks.Rating20 = synalinks.Field(
            description="Grade of the essay, from 1 (worst) to 20 (best)",
        )

    # Rating20 values are real ints, usable in arithmetic.
    assert synalinks.Rating20.FIFTEEN == 15
    normalized = synalinks.Rating20.FIFTEEN / 20  # 0.75
    ```
    """

    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    THIRTEEN = 13
    FOURTEEN = 14
    FIFTEEN = 15
    SIXTEEN = 16
    SEVENTEEN = 17
    EIGHTEEN = 18
    NINETEEN = 19
    TWENTY = 20


# Built-in discretized scales, keyed by their registered (class) name, so
# modules can accept a `score_type` as either the class or its string name
# and round-trip it through `get_config()` / `from_config()`.
_BUILTIN_SCORE_TYPES = {
    "Score": Score,
    "FineScore": FineScore,
    "Rating": Rating,
    "Rating10": Rating10,
    "Rating20": Rating20,
}


def _is_score_type(obj):
    return (
        isinstance(obj, type)
        and issubclass(obj, Enum)
        and issubclass(obj, (int, float))
        and not issubclass(obj, bool)
        and len(obj) > 0
    )


@synalinks_export("synalinks.backend.get_score_type")
def get_score_type(identifier):
    """Resolve a discretized score scale from a class or a string name.

    A "score type" is any `Enum` whose members are also `int` or `float`
    (`Score`, `FineScore`, `Rating`, `Rating10`, `Rating20`, or your own).
    Modules such as `SelfCritique` and `LMAsJudge` take a `score_type`
    argument and use this helper so that both the class itself and its
    name (as stored in a serialized config) are accepted.

    Args:
        identifier (type | str | None): The score type class, its name
            (`"Score"`, `"FineScore"`, `"Rating"`, `"Rating10"`, `"Rating20"`,
            or the registered name of a custom one), or `None` for the
            default `Score`.

    Returns:
        (type): The resolved score type class.
    """
    if identifier is None:
        return Score
    if isinstance(identifier, str):
        score_type = _BUILTIN_SCORE_TYPES.get(identifier)
        if score_type is None:
            score_type = get_registered_object(identifier)
        if score_type is None or not _is_score_type(score_type):
            raise ValueError(
                f"Unknown score type '{identifier}'. Expected one of "
                f"{list(_BUILTIN_SCORE_TYPES)} or the registered name of an "
                "`Enum` subclass whose members are `int` or `float`."
            )
        return score_type
    if _is_score_type(identifier):
        return identifier
    raise ValueError(
        "Could not interpret score type identifier: "
        f"{identifier!r}. Expected a class inheriting from `Enum` and `int` "
        "or `float` (e.g. `synalinks.Score`, `synalinks.Rating`) or its name."
    )


def serialize_score_type(score_type):
    """Return the string name under which a score type is serialized."""
    return get_registered_name(get_score_type(score_type))


def score_type_bounds(score_type):
    """Return `(minimum, maximum)` member values of a score type."""
    values = [member.value for member in get_score_type(score_type)]
    return min(values), max(values)


def score_type_json_type(score_type):
    """Return the JSON schema `type` (`"integer"` or `"number"`) of a score type."""
    score_type = get_score_type(score_type)
    return "integer" if issubclass(score_type, int) else "number"


def score_type_description(score_type):
    """Return a short, LM-facing description of the scale of a score type."""
    score_type = get_score_type(score_type)
    lo, hi = score_type_bounds(score_type)
    if issubclass(score_type, int):
        return f"an integer between {lo} and {hi}, {lo} being very bad and {hi} very good"
    return f"a float between {lo} and {hi}, {lo} being very bad and {hi} very good"


@synalinks_export("synalinks.backend.normalize_score")
def normalize_score(value, score_type):
    """Normalize a raw score to a float between 0.0 and 1.0.

    The lowest member of `score_type` maps to `0.0` and the highest to
    `1.0`, linearly in between; e.g. `Rating.THREE` (3 on a 1..5 scale)
    maps to `0.5`, `Rating10.SEVEN` to `0.667`, and any `Score` /
    `FineScore` value is returned unchanged since it already spans 0..1.
    The result is clamped to `[0.0, 1.0]` so a value slightly outside the
    scale (e.g. from a lenient provider) cannot produce an invalid reward.

    Args:
        value (int | float | Enum): The raw score, as emitted by the
            language model (a plain number or a member of `score_type`).
        score_type (type | str): The scale the value was picked from
            (see `get_score_type`).

    Returns:
        (float): The normalized score between 0.0 and 1.0.
    """
    lo, hi = score_type_bounds(score_type)
    value = float(value)
    if hi == lo:
        return 1.0
    normalized = (value - lo) / (hi - lo)
    return min(1.0, max(0.0, normalized))
