# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend.pydantic.metrics import FineScore
from synalinks.src.backend.pydantic.metrics import Rating
from synalinks.src.backend.pydantic.metrics import Rating10
from synalinks.src.backend.pydantic.metrics import Rating20
from synalinks.src.backend.pydantic.metrics import Score


def _enum_schema(enum_cls):
    class Holder(DataModel):
        value: enum_cls = Field(description="value")

    return Holder.get_schema()["$defs"][enum_cls.__name__]


class MetricsTest(testing.TestCase):
    def test_score_levels(self):
        values = [m.value for m in Score]
        self.assertEqual(len(values), 11)
        self.assertEqual(values, [round(i / 10, 2) for i in range(11)])
        self.assertEqual(_enum_schema(Score)["type"], "number")

    def test_fine_score_levels(self):
        values = [m.value for m in FineScore]
        self.assertEqual(len(values), 21)
        self.assertEqual(values, [round(i / 20, 2) for i in range(21)])
        schema = _enum_schema(FineScore)
        self.assertEqual(schema["type"], "number")
        self.assertEqual(schema["enum"], values)

    def test_fine_score_is_superset_of_score(self):
        for member in Score:
            self.assertEqual(FineScore[member.name].value, member.value)

    def test_score_arithmetic(self):
        self.assertAlmostEqual(Score.GOOD + 0.05, 0.95)
        self.assertAlmostEqual(FineScore.HIGH * 2, 1.5)

    def test_rating_levels(self):
        for enum_cls, n in ((Rating, 5), (Rating10, 10), (Rating20, 20)):
            values = [m.value for m in enum_cls]
            self.assertEqual(values, list(range(1, n + 1)), enum_cls.__name__)
            schema = _enum_schema(enum_cls)
            self.assertEqual(schema["type"], "integer", enum_cls.__name__)
            self.assertEqual(schema["enum"], values, enum_cls.__name__)

    def test_rating_arithmetic(self):
        self.assertEqual(Rating.FOUR / 5, 0.8)
        self.assertEqual(Rating10.SEVEN + 3, 10)
        self.assertEqual(Rating20.TWENTY // 4, 5)

    def test_rating_validation(self):
        class Holder(DataModel):
            value: Rating = Field(description="value")

        self.assertEqual(Holder(value=3).value, Rating.THREE)
        with self.assertRaises(Exception):
            Holder(value=6)
