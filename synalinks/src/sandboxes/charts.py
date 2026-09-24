# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
# Mirrors the chart types of E2B's code-interpreter SDK
# (e2b_code_interpreter/charts.py, Apache 2.0, (c) FoundryLabs, Inc.).
"""The chart data of `Result.chart`, as E2B's SDK decodes it.

When a snippet displays a matplotlib figure, the sandbox extracts its data
with E2B's own extractor (``e2b_charts``) and decodes it into these types:
the plotted points of a line or scatter chart, the bars of a bar chart, the
wedges of a pie, the statistics of a box plot, or one chart per subplot in a
`SuperChart`. Useful to rebuild an interactive chart, or to let a language
model read the numbers behind a plot it cannot see.
"""

import enum
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from synalinks.src.api_export import synalinks_export


@synalinks_export("synalinks.sandboxes.ChartType")
class ChartType(str, enum.Enum):
    """The kind of a chart."""

    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    PIE = "pie"
    BOX_AND_WHISKER = "box_and_whisker"
    SUPERCHART = "superchart"
    UNKNOWN = "unknown"


@synalinks_export("synalinks.sandboxes.ScaleType")
class ScaleType(str, enum.Enum):
    """The scale of a chart axis."""

    LINEAR = "linear"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    LOG = "log"
    SYMLOG = "symlog"
    LOGIT = "logit"
    FUNCTION = "function"
    FUNCTIONLOG = "functionlog"
    ASINH = "asinh"
    UNKNOWN = "unknown"


@synalinks_export("synalinks.sandboxes.Chart")
class Chart:
    """Data extracted from a chart; `to_dict` gives it back as extracted."""

    type: ChartType
    title: str
    elements: List[Any]

    def __init__(self, **kwargs):
        self.raw_data = kwargs
        self.type = ChartType(kwargs["type"] or ChartType.UNKNOWN)
        self.title = kwargs["title"]
        self.elements = kwargs["elements"]

    def to_dict(self) -> dict:
        return self.raw_data


@synalinks_export("synalinks.sandboxes.Chart2D")
class Chart2D(Chart):
    """A chart with an x and a y axis."""

    x_label: Optional[str]
    y_label: Optional[str]
    x_unit: Optional[str]
    y_unit: Optional[str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x_label = kwargs["x_label"]
        self.y_label = kwargs["y_label"]
        self.x_unit = kwargs["x_unit"]
        self.y_unit = kwargs["y_unit"]


@synalinks_export("synalinks.sandboxes.PointData")
class PointData:
    """One series of a line or scatter chart: its label and ``(x, y)`` points."""

    label: str
    points: List[Tuple[Union[str, float], Union[str, float]]]

    def __init__(self, **kwargs):
        self.label = kwargs["label"]
        self.points = [(x, y) for x, y in kwargs["points"]]


@synalinks_export("synalinks.sandboxes.PointChart")
class PointChart(Chart2D):
    """A chart of point series, with its axes' ticks and scales."""

    x_ticks: List[Union[str, float]]
    x_tick_labels: List[str]
    x_scale: ScaleType
    y_ticks: List[Union[str, float]]
    y_tick_labels: List[str]
    y_scale: ScaleType
    elements: List[PointData]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.x_scale = ScaleType(kwargs.get("x_scale"))
        except ValueError:
            self.x_scale = ScaleType.UNKNOWN
        self.x_ticks = kwargs["x_ticks"]
        self.x_tick_labels = kwargs["x_tick_labels"]
        try:
            self.y_scale = ScaleType(kwargs.get("y_scale"))
        except ValueError:
            self.y_scale = ScaleType.UNKNOWN
        self.y_ticks = kwargs["y_ticks"]
        self.y_tick_labels = kwargs["y_tick_labels"]
        self.elements = [PointData(**d) for d in kwargs["elements"]]


@synalinks_export("synalinks.sandboxes.LineChart")
class LineChart(PointChart):
    """A line chart."""

    type = ChartType.LINE


@synalinks_export("synalinks.sandboxes.ScatterChart")
class ScatterChart(PointChart):
    """A scatter chart."""

    type = ChartType.SCATTER


@synalinks_export("synalinks.sandboxes.BarData")
class BarData:
    """One bar: its label, its group and its value."""

    label: str
    group: str
    value: str

    def __init__(self, **kwargs):
        self.label = kwargs["label"]
        self.value = kwargs["value"]
        self.group = kwargs["group"]


@synalinks_export("synalinks.sandboxes.BarChart")
class BarChart(Chart2D):
    """A bar chart."""

    type = ChartType.BAR
    elements: List[BarData]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elements = [BarData(**d) for d in kwargs["elements"]]


@synalinks_export("synalinks.sandboxes.PieData")
class PieData:
    """One wedge: its label, angle and radius."""

    label: str
    angle: float
    radius: float

    def __init__(self, **kwargs):
        self.label = kwargs["label"]
        self.angle = kwargs["angle"]
        self.radius = kwargs["radius"]


@synalinks_export("synalinks.sandboxes.PieChart")
class PieChart(Chart):
    """A pie chart."""

    type = ChartType.PIE
    elements: List[PieData]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elements = [PieData(**d) for d in kwargs["elements"]]


@synalinks_export("synalinks.sandboxes.BoxAndWhiskerData")
class BoxAndWhiskerData:
    """One box of a box plot: its label, quartiles, extremes and outliers."""

    label: str
    min: float
    first_quartile: float
    median: float
    third_quartile: float
    max: float
    outliers: List[float]

    def __init__(self, **kwargs):
        self.label = kwargs["label"]
        self.min = kwargs["min"]
        self.first_quartile = kwargs["first_quartile"]
        self.median = kwargs["median"]
        self.third_quartile = kwargs["third_quartile"]
        self.max = kwargs["max"]
        self.outliers = kwargs.get("outliers") or []


@synalinks_export("synalinks.sandboxes.BoxAndWhiskerChart")
class BoxAndWhiskerChart(Chart2D):
    """A box plot."""

    type = ChartType.BOX_AND_WHISKER
    elements: List[BoxAndWhiskerData]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elements = [BoxAndWhiskerData(**d) for d in kwargs["elements"]]


@synalinks_export("synalinks.sandboxes.SuperChart")
class SuperChart(Chart):
    """A figure with several subplots: one chart per subplot."""

    type = ChartType.SUPERCHART
    elements: List[Chart]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elements = []
        for raw_chart in kwargs["elements"]:
            chart = deserialize_chart(raw_chart)
            if chart is not None:
                self.elements.append(chart)


ChartTypes = Union[
    Chart, LineChart, ScatterChart, BarChart, PieChart, BoxAndWhiskerChart, SuperChart
]


def deserialize_chart(data: Optional[dict]) -> Optional[ChartTypes]:
    """Decode extracted chart data into its chart type (``None`` if empty)."""
    if not data:
        return None
    chart_class = {
        ChartType.LINE: LineChart,
        ChartType.SCATTER: ScatterChart,
        ChartType.BAR: BarChart,
        ChartType.PIE: PieChart,
        ChartType.BOX_AND_WHISKER: BoxAndWhiskerChart,
        ChartType.SUPERCHART: SuperChart,
    }.get(data["type"], Chart)
    return chart_class(**data)
