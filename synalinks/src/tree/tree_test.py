# Modified from: keras/src/tree/tree_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import functools
from collections import OrderedDict
from collections import defaultdict
from collections import deque
from collections import namedtuple

import numpy as np
from absl.testing import parameterized

from synalinks.src import testing
from synalinks.src.tree.tree_api import MAP_TO_NONE
from synalinks.src.utils.module_utils import optree
from synalinks.src.utils.tracking import TrackedDict
from synalinks.src.utils.tracking import TrackedList
from synalinks.src.utils.tracking import TrackedSet

TEST_CASES = []

if optree.available:
    from synalinks.src.tree import optree_impl

    TEST_CASES += [
        {
            "testcase_name": "optree",
            "t": optree_impl,
        },
    ]

Empty = namedtuple("Empty", [])
Point = namedtuple("Point", ["x", "y"])
OtherPoint = namedtuple("OtherPoint", ["x", "y"])


def default_value():
    return None


class Visitor:
    def __init__(self, func):
        self.func = func
        self.visited_list = []

    def __call__(self, x):
        self.visited_list.append(x)
        return self.func(x)

    def visited(self):
        ret = self.visited_list
        self.visited_list = []
        return ret


@parameterized.named_parameters(TEST_CASES)
class TreeTest(testing.TestCase):
    # def setUp(self):
    #     if dmtree.available and optree.available:
    #         # If both are available, the annotation on the Keras tracking
    #         # wrappers will have used optree. For testing purposes, we need to
    #         # also register them with dm-tree.
    #         from synalinks.src.tree import dmtree_impl

    #         dmtree_impl.register_tree_node_class(TrackedList)
    #         dmtree_impl.register_tree_node_class(TrackedSet)
    #         dmtree_impl.register_tree_node_class(TrackedDict)
    #     super().setUp()

    def assertEqualStrict(self, a, b):
        self.assertEqual(a, b)
        self.assertEqual(type(a), type(b))
        if isinstance(a, OrderedDict):
            # Verify order.
            self.assertEqual(a.items(), b.items())
        elif isinstance(a, defaultdict):
            self.assertEqual(a.default_factory, b.default_factory)
        # Recurse
        if isinstance(a, (tuple, list, deque)):
            for sub_a, sub_b in zip(a, b):
                self.assertEqualStrict(sub_a, sub_b)
        elif isinstance(a, dict):
            for k in a:
                self.assertEqualStrict(a[k], b[k])

    def is_dmtree(self, tree_impl):
        # if dmtree.available:
        #     from synalinks.src.tree import dmtree_impl

        #     return tree_impl is dmtree_impl
        return False

    def test_is_nested(self, t):
        # Non-nested.
        self.assertFalse(t.is_nested(1))
        self.assertFalse(t.is_nested("1234"))
        self.assertFalse(t.is_nested(b"1234"))
        self.assertFalse(t.is_nested(bytearray("1234", "ascii")))
        self.assertFalse(t.is_nested(set([1, 2])))

        # Standard structures.
        self.assertTrue(t.is_nested(()))
        self.assertTrue(t.is_nested((1,)))
        self.assertTrue(t.is_nested((1, 2)))
        self.assertTrue(t.is_nested([]))
        self.assertTrue(t.is_nested([1]))
        self.assertTrue(t.is_nested([1, 2]))
        self.assertTrue(t.is_nested(deque([])))
        self.assertTrue(t.is_nested(deque([1])))
        self.assertTrue(t.is_nested(deque([1, 2])))
        self.assertTrue(t.is_nested(Empty()))
        self.assertTrue(t.is_nested(Point(x=1, y=2)))
        self.assertTrue(t.is_nested({}))
        self.assertTrue(t.is_nested({"a": 1}))
        self.assertTrue(t.is_nested({"b": 2, "a": 1}))
        self.assertTrue(t.is_nested(OrderedDict()))
        self.assertTrue(t.is_nested(OrderedDict([("a", 1)])))
        self.assertTrue(t.is_nested(OrderedDict([("b", 2), ("a", 1)])))
        self.assertTrue(t.is_nested(defaultdict(default_value)))
        self.assertTrue(t.is_nested(defaultdict(default_value, [("a", 1)])))
        self.assertTrue(t.is_nested(defaultdict(default_value, [("b", 2), ("a", 1)])))

        # Keras tracking wrappers.
        self.assertTrue(t.is_nested(TrackedList([])))
        self.assertTrue(t.is_nested(TrackedList([1])))
        self.assertTrue(t.is_nested(TrackedList([1, 2])))
        self.assertTrue(t.is_nested(TrackedSet([])))
        self.assertTrue(t.is_nested(TrackedSet([1])))
        self.assertTrue(t.is_nested(TrackedSet([1, 2])))
        self.assertTrue(t.is_nested(TrackedDict({})))
        self.assertTrue(t.is_nested(TrackedDict({"a": 1})))
        self.assertTrue(t.is_nested(TrackedDict({"b": 2, "a": 1})))

    def test_flatten(self, t):
        # Non-nested.
        self.assertEqualStrict(t.flatten(1), [1])

        # Standard structures.
        self.assertEqualStrict(t.flatten(()), [])
        self.assertEqualStrict(t.flatten((1,)), [1])
        self.assertEqualStrict(t.flatten((1, 2)), [1, 2])
        self.assertEqualStrict(t.flatten([]), [])
        self.assertEqualStrict(t.flatten([1]), [1])
        self.assertEqualStrict(t.flatten([1, 2]), [1, 2])
        self.assertEqualStrict(t.flatten(deque([])), [])
        self.assertEqualStrict(t.flatten(deque([1])), [1])
        self.assertEqualStrict(t.flatten(deque([1, 2])), [1, 2])
        self.assertEqualStrict(t.flatten(Empty()), [])
        self.assertEqualStrict(t.flatten(Point(y=2, x=1)), [1, 2])
        self.assertEqualStrict(t.flatten({}), [])
        self.assertEqualStrict(t.flatten({"a": 1}), [1])
        self.assertEqualStrict(t.flatten({"b": 2, "a": 1}), [1, 2])
        self.assertEqualStrict(
            t.flatten(OrderedDict()),
            [],
        )
        self.assertEqualStrict(
            t.flatten(OrderedDict([("a", 1)])),
            [1],
        )
        self.assertEqualStrict(
            t.flatten(OrderedDict([("b", 2), ("a", 1)])),
            [2, 1],
        )
        self.assertEqualStrict(
            t.flatten(defaultdict(default_value)),
            [],
        )
        self.assertEqualStrict(
            t.flatten(defaultdict(default_value, [("a", 1)])),
            [1],
        )
        self.assertEqualStrict(
            t.flatten(defaultdict(default_value, [("b", 2), ("a", 1)])),
            [1, 2],
        )

        # Keras tracking wrappers.
        self.assertEqualStrict(t.flatten(TrackedList([])), [])
        self.assertEqualStrict(t.flatten(TrackedList([1])), [1])
        self.assertEqualStrict(t.flatten(TrackedList([1, 2])), [1, 2])
        self.assertEqualStrict(t.flatten(TrackedSet([])), [])
        self.assertEqualStrict(t.flatten(TrackedSet([1])), [1])
        self.assertEqualStrict(sorted(t.flatten(TrackedSet([1, 2]))), [1, 2])
        self.assertEqualStrict(t.flatten(TrackedDict({})), [])
        self.assertEqualStrict(t.flatten(TrackedDict({"a": 1})), [1])
        self.assertEqualStrict(t.flatten(TrackedDict({"b": 2, "a": 1})), [1, 2])

        # Deeper nested structures.
        self.assertEqualStrict(
            t.flatten(
                (
                    {"b": [2, 3], "a": (1,)},
                    TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                    TrackedSet([7]),
                    Point(y=9, x=8),
                    np.array([10]),
                )
            ),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, np.array([10])],
        )

    def test_map_structure_with_multiple_structures(self, t):
        def f2(x, y):
            return x + y if isinstance(x, int) and isinstance(y, int) else None

        # Non-nested.
        self.assertEqualStrict(t.map_structure(f2, 1, 10), 11)

        # Standard structures.
        self.assertEqualStrict(t.map_structure(f2, ()), ())
        self.assertEqualStrict(t.map_structure(f2, (1,), (10,)), (11,))
        self.assertEqualStrict(t.map_structure(f2, (1, 2), (10, 20)), (11, 22))
        self.assertEqualStrict(t.map_structure(f2, []), [])
        self.assertEqualStrict(t.map_structure(f2, [1], [10]), [11])
        self.assertEqualStrict(t.map_structure(f2, [1, 2], [10, 20]), [11, 22])
        self.assertEqualStrict(t.map_structure(f2, deque([])), deque([]))
        self.assertEqualStrict(t.map_structure(f2, deque([1]), deque([10])), deque([11]))
        self.assertEqualStrict(
            t.map_structure(f2, deque([1, 2]), deque([10, 20])), deque([11, 22])
        )
        self.assertEqualStrict(t.map_structure(f2, Empty()), Empty())
        self.assertEqualStrict(
            t.map_structure(f2, Point(y=2, x=1), Point(x=10, y=20)),
            Point(x=11, y=22),
        )
        self.assertEqualStrict(t.map_structure(f2, {}), {})
        self.assertEqualStrict(t.map_structure(f2, {"a": 1}, {"a": 10}), {"a": 11})
        self.assertEqualStrict(
            t.map_structure(f2, {"b": 2, "a": 1}, {"a": 10, "b": 20}),
            {"a": 11, "b": 22},
        )
        self.assertEqualStrict(
            t.map_structure(f2, OrderedDict()),
            OrderedDict(),
        )
        self.assertEqualStrict(
            t.map_structure(f2, OrderedDict([("a", 1)]), OrderedDict([("a", 10)])),
            OrderedDict([("a", 11)]),
        )
        self.assertEqualStrict(
            t.map_structure(
                f2,
                OrderedDict([("b", 2), ("a", 1)]),
                OrderedDict([("b", 20), ("a", 10)]),
            ),
            OrderedDict([("b", 22), ("a", 11)]),
        )
        self.assertEqualStrict(
            t.map_structure(f2, defaultdict(default_value), defaultdict(default_value)),
            defaultdict(default_value),
        )
        self.assertEqualStrict(
            t.map_structure(
                f2,
                defaultdict(default_value, [("a", 1)]),
                defaultdict(default_value, [("a", 10)]),
            ),
            defaultdict(default_value, [("a", 11)]),
        )
        self.assertEqualStrict(
            t.map_structure(
                f2,
                defaultdict(default_value, [("b", 2), ("a", 1)]),
                defaultdict(default_value, [("a", 10), ("b", 20)]),
            ),
            defaultdict(default_value, [("a", 11), ("b", 22)]),
        )

        # Keras tracking wrappers.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                TrackedList([]),
                TrackedList([]),
            ),
            TrackedList([]),
        )
        self.assertEqualStrict(
            t.map_structure(
                f2,
                TrackedList([1]),
                TrackedList([10]),
            ),
            TrackedList([11]),
        )
        self.assertEqualStrict(
            t.map_structure(
                f2,
                TrackedList([1, 2]),
                TrackedList([10, 20]),
            ),
            TrackedList([11, 22]),
        )

        # Known limitation of the dm-tree implementation:
        # Registered classes are not handled when mapping multiple
        # structures at once. TrackedSet is the only problematic one.
        if not self.is_dmtree(t):
            self.assertEqualStrict(
                t.map_structure(
                    f2,
                    TrackedSet([]),
                    TrackedSet([]),
                ),
                TrackedSet([]),
            )
            self.assertEqualStrict(
                t.map_structure(
                    f2,
                    TrackedSet([1]),
                    TrackedSet([10]),
                ),
                TrackedSet([11]),
            )
            self.assertEqualStrict(
                t.map_structure(
                    f2,
                    TrackedSet([1, 2]),
                    TrackedSet([10, 20]),
                ),
                TrackedSet([11, 22]),
            )

        self.assertEqualStrict(
            t.map_structure(
                f2,
                TrackedDict({}),
                TrackedDict({}),
            ),
            TrackedDict({}),
        )
        self.assertEqualStrict(
            t.map_structure(
                f2,
                TrackedDict({"a": 1}),
                TrackedDict({"a": 10}),
            ),
            TrackedDict({"a": 11}),
        )
        self.assertEqualStrict(
            t.map_structure(
                f2,
                TrackedDict({"b": 2, "a": 1}),
                TrackedDict({"a": 10, "b": 20}),
            ),
            TrackedDict({"a": 11, "b": 22}),
        )

        # Deeper nested structures.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                (
                    {"b": [2, 3], "a": (1,)},
                    TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                    TrackedSet([7]),
                    Point(y=9, x=8),
                    np.array([10]),
                ),
                (
                    {"b": [20, 30], "a": (10,)},
                    TrackedDict({"x": 40, "y": TrackedList([50, 60])}),
                    TrackedSet([70]),
                    Point(y=90, x=80),
                    np.array([100]),
                ),
            ),
            (
                {"b": [22, 33], "a": (11,)},
                TrackedDict({"x": 44, "y": TrackedList([55, 66])}),
                # Known limitation of the dm-tree implementation:
                # Registered classes are not handled when mapping multiple
                # structures at once. TrackedSet is the only problematic one.
                None if self.is_dmtree(t) else TrackedSet([77]),
                Point(y=99, x=88),
                None,
            ),
        )

        # Error cases.

        # list, tuple, deque and namedtuple are not considered equivalent.
        # Test all 6 combinations:
        # tuple, list.
        with self.assertRaisesRegex(ValueError, "tuple"):
            t.map_structure(f2, (), [])
        # tuple, deque.
        with self.assertRaisesRegex(ValueError, "tuple"):
            t.map_structure(f2, (), deque())
        # tuple, namedtuple.
        with self.assertRaisesRegex(ValueError, "tuple"):
            t.map_structure(f2, (), Empty())
        # list, deque.
        with self.assertRaisesRegex(ValueError, "list"):
            t.map_structure(f2, [], deque())
        # list, namedtuple.
        with self.assertRaisesRegex(ValueError, "list"):
            t.map_structure(f2, [], Empty())
        # deque, namedtuple.
        with self.assertRaisesRegex(ValueError, "deque"):
            t.map_structure(f2, deque(), Empty())

        # Equivalent namedtuples don't match.
        with self.assertRaisesRegex(ValueError, "namedtuple"):
            t.map_structure(f2, Point(x=1, y=2), OtherPoint(x=10, y=20))

        # Mismatched counts.
        with self.assertRaisesRegex(ValueError, "(number|[Aa]rity)"):
            t.map_structure(f2, (1, 2), (1,))
        with self.assertRaisesRegex(ValueError, "(number|[Aa]rity)"):
            t.map_structure(f2, [1, 2], [1])
        with self.assertRaisesRegex(ValueError, "(number|[Aa]rity)"):
            t.map_structure(f2, deque([1, 2]), deque([1]))

        # dict, OrderedDict, defaultdict are considered equivalent, but the
        # returned type is the first one. Test all 6 combinations (3 type
        # combinations plus the order).
        # dict, OrderedDict yields dict.
        self.assertEqualStrict(
            t.map_structure(f2, {"a": 1, "b": 2}, OrderedDict([("b", 20), ("a", 10)])),
            {"a": 11, "b": 22},
        )
        # OrderedDict, dict yields OrderedDict with same order.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                OrderedDict([("b", 2), ("a", 1)]),
                {"a": 10, "b": 20},
            ),
            OrderedDict([("b", 22), ("a", 11)]),
        )
        # dict, defaultdict yields dict.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                {"a": 1, "b": 2},
                defaultdict(default_value, [("b", 20), ("a", 10)]),
            ),
            {"a": 11, "b": 22},
        )
        # defaultdict, dict yields defaultdict.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                defaultdict(default_value, [("b", 2), ("a", 1)]),
                {"a": 10, "b": 20},
            ),
            defaultdict(default_value, [("a", 11), ("b", 22)]),
        )
        # defaultdict, OrderedDict yields defaultdict.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                defaultdict(default_value, [("a", 1), ("b", 2)]),
                OrderedDict([("b", 20), ("a", 10)]),
            ),
            defaultdict(default_value, [("a", 11), ("b", 22)]),
        )
        # OrderedDict, defaultdict yields OrderedDict with same order.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                OrderedDict([("b", 2), ("a", 1)]),
                defaultdict(default_value, [("a", 10), ("b", 20)]),
            ),
            OrderedDict([("b", 22), ("a", 11)]),
        )

        # Multiple OrderedDicts with same keys but different orders, the order
        # of the first one prevails.
        self.assertEqualStrict(
            t.map_structure(
                f2,
                OrderedDict([("b", 2), ("a", 1)]),
                OrderedDict([("a", 10), ("b", 20)]),
            ),
            OrderedDict([("b", 22), ("a", 11)]),
        )

        # Mismatched keys
        with self.assertRaisesRegex(ValueError, "key"):
            t.map_structure(f2, {"a": 1, "b": 2}, {"a": 1})
        with self.assertRaisesRegex(ValueError, "key"):
            t.map_structure(
                f2,
                defaultdict(default_value, [("a", 1), ("b", 2)]),
                defaultdict(default_value, [("a", 10)]),
            )
        with self.assertRaisesRegex(ValueError, "key"):
            t.map_structure(
                f2, OrderedDict([("a", 1), ("b", 2)]), OrderedDict([("a", 10)])
            )

    def test_assert_same_structure(self, t):
        # Non-nested.
        t.assert_same_structure(1, 10)

        # Standard structures.
        t.assert_same_structure((), ())
        t.assert_same_structure((1,), (10,))
        t.assert_same_structure((1, 2), (10, 20))
        t.assert_same_structure([], [])
        t.assert_same_structure([1], [10])
        t.assert_same_structure([1, 2], [10, 20])
        t.assert_same_structure(deque([]), deque([]))
        t.assert_same_structure(deque([1]), deque([1]))
        t.assert_same_structure(deque([1, 2]), deque([10, 20]))
        t.assert_same_structure(Empty(), Empty())
        t.assert_same_structure(Point(y=1, x=2), Point(x=10, y=20))
        t.assert_same_structure({}, {})
        t.assert_same_structure({"a": 1}, {"a": 10})
        t.assert_same_structure({"b": 2, "a": 1}, {"a": 10, "b": 20})
        t.assert_same_structure(OrderedDict(), OrderedDict())
        t.assert_same_structure(OrderedDict([("a", 1)]), OrderedDict([("a", 10)]))
        t.assert_same_structure(
            OrderedDict([("b", 1), ("a", 2)]),
            OrderedDict([("b", 10), ("a", 20)]),
        )
        t.assert_same_structure(defaultdict(default_value), defaultdict(default_value))
        t.assert_same_paths(
            defaultdict(default_value, [("a", 1)]),
            defaultdict(default_value, [("a", 10)]),
        )
        t.assert_same_paths(
            defaultdict(default_value, [("b", 1), ("a", 2)]),
            defaultdict(default_value, [("a", 10), ("b", 20)]),
        )

        # Keras tracking wrappers.
        t.assert_same_structure(
            TrackedList([]),
            TrackedList([]),
        )
        t.assert_same_structure(
            TrackedList([1]),
            TrackedList([10]),
        )
        t.assert_same_structure(
            TrackedList([1, 2]),
            TrackedList([10, 20]),
        )
        t.assert_same_structure(
            TrackedSet([]),
            TrackedSet([]),
        )
        t.assert_same_structure(
            TrackedSet([1]),
            TrackedSet([10]),
        )
        t.assert_same_structure(
            TrackedSet([1, 2]),
            TrackedSet([10, 20]),
        )
        t.assert_same_structure(
            TrackedDict({}),
            TrackedDict({}),
        )
        t.assert_same_structure(
            TrackedDict({"a": 1}),
            TrackedDict({"a": 10}),
        )
        t.assert_same_structure(
            TrackedDict({"b": 2, "a": 1}),
            TrackedDict({"a": 10, "b": 20}),
        )

        # Deeper nested structures.
        t.assert_same_structure(
            (
                {"b": [2, 3], "a": (1,)},
                TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                TrackedSet([7]),
                Point(y=9, x=8),
                np.array([10]),
            ),
            (
                {"b": [20, 30], "a": (10,)},
                TrackedDict({"x": 40, "y": TrackedList([50, 60])}),
                TrackedSet([70]),
                Point(y=90, x=80),
                np.array([100]),
            ),
        )

        # Error cases.

        # Non-nested vs. nested.
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, ())
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*tuple"):
            t.assert_same_structure((), 1)
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, [])
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*list"):
            t.assert_same_structure([], 1)
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, deque([]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*deque"):
            t.assert_same_structure(deque([]), 1)
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, Empty())
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*(Empty|tuple)"):
            t.assert_same_structure(Empty(), 1)
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, Point(x=1, y=2))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*(Point|tuple)"):
            t.assert_same_structure(Point(x=1, y=2), 1)
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, {})
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*dict"):
            t.assert_same_structure({}, 1)
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, OrderedDict())
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*OrderedDict"):
            t.assert_same_structure(OrderedDict(), 1)
        with self.assertRaisesRegex(ValueError, "don't have the same nested"):
            t.assert_same_structure(1, defaultdict(default_value))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*defaultdict"):
            t.assert_same_structure(defaultdict(default_value), 1)

        # Non-nested vs. Keras tracking wrappers.
        with self.assertRaisesRegex(ValueError, "(nested|TrackedList)"):
            t.assert_same_structure(1, TrackedList([]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*TrackedList"):
            t.assert_same_structure(TrackedList([]), 1)
        with self.assertRaisesRegex(ValueError, "(nested|TrackedSet)"):
            t.assert_same_structure(1, TrackedSet([]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*TrackedSet"):
            t.assert_same_structure(TrackedSet([]), 1)
        with self.assertRaisesRegex(ValueError, "(nested|TrackedDict)"):
            t.assert_same_structure(1, TrackedDict([]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*TrackedDict"):
            t.assert_same_structure(TrackedDict([]), 1)

        # list, tuple, deque and namedtuple are not considered equivalent.
        # Test all 6 combinations:
        # tuple, list.
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*tuple"):
            t.assert_same_structure((), [])
        # tuple, deque.
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*tuple"):
            t.assert_same_structure((), deque())
        # tuple, namedtuple.
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*tuple"):
            t.assert_same_structure((), Empty())
        # list, deque.
        with self.assertRaisesRegex(ValueError, "list"):
            t.assert_same_structure([], deque())
        # list, namedtuple.
        with self.assertRaisesRegex(ValueError, "list"):
            t.assert_same_structure([], Empty())
        # deque, namedtuple.
        with self.assertRaisesRegex(ValueError, "deque"):
            t.assert_same_structure(deque(), Empty())

        # Equivalent namedtuples don't match.
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*[. ]Point"):
            t.assert_same_structure(Point(x=1, y=2), OtherPoint(x=10, y=20))

        # Mismatched counts.
        with self.assertRaisesRegex(ValueError, "[Aa]rity mismatch"):
            t.assert_same_structure((1, 2), (1,))
        with self.assertRaisesRegex(ValueError, "[Aa]rity mismatch"):
            t.assert_same_structure([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "[Aa]rity mismatch"):
            t.assert_same_structure(deque([1, 2]), deque([1]))

        # Mismatched counts with Keras tracking wrappers.
        with self.assertRaisesRegex(ValueError, "[Aa]rity mismatch"):
            t.assert_same_structure(TrackedList([1, 2]), TrackedList([1]))
        with self.assertRaisesRegex(ValueError, "[Aa]rity mismatch"):
            t.assert_same_structure(TrackedSet([1, 2]), TrackedSet([1]))

        # dict, OrderedDict, defaultdict are considered equivalent.
        # Test all 6 combinations (3 type combinations plus the order).
        # dict, OrderedDict.
        t.assert_same_structure({"a": 1, "b": 2}, OrderedDict([("b", 20), ("a", 10)]))
        # OrderedDict, dict.
        t.assert_same_structure(OrderedDict([("b", 20), ("a", 10)]), {"a": 1, "b": 2})
        # dict, defaultdict.
        t.assert_same_structure(
            {"a": 1, "b": 2},
            defaultdict(default_value, [("b", 20), ("a", 10)]),
        )
        # defaultdict, dict.
        t.assert_same_structure(
            defaultdict(default_value, [("b", 20), ("a", 10)]),
            {"a": 1, "b": 2},
        )
        # defaultdict, OrderedDict.
        t.assert_same_structure(
            defaultdict(default_value, [("a", 1), ("b", 2)]),
            OrderedDict([("b", 20), ("a", 10)]),
        )
        # OrderedDict, defaultdict.
        t.assert_same_structure(
            OrderedDict([("b", 2), ("a", 1)]),
            defaultdict(default_value, [("a", 10), ("b", 20)]),
        )

        # Two OrderedDicts with same keys but different orders.
        t.assert_same_structure(
            OrderedDict([("b", 2), ("a", 1)]),
            OrderedDict([("a", 10), ("b", 20)]),
        )

        # Keras tracking wrappers are not equivalent to the raw structures.
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*TrackedList"):
            t.assert_same_structure(TrackedList([1, 2]), list([10, 20]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*list"):
            t.assert_same_structure(list([1, 2]), TrackedList([10, 20]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*TrackedSet"):
            t.assert_same_structure(TrackedSet([1, 2]), list([10, 20]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*list"):
            t.assert_same_structure(list([1, 2]), TrackedSet([10, 20]))
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*TrackedDict"):
            t.assert_same_structure(TrackedDict({"b": 2, "a": 1}), {"a": 10, "b": 20})
        with self.assertRaisesRegex(ValueError, "[Ee]xpected.*dict"):
            t.assert_same_structure({"b": 2, "a": 1}, TrackedDict({"a": 10, "b": 20}))

        # Mismatched key count.
        with self.assertRaisesRegex(ValueError, "[Dd]ictionary key mismatch"):
            t.assert_same_structure(
                {"a": 1, "b": 2},
                {"a": 1},
            )
        with self.assertRaisesRegex(ValueError, "[Dd]ictionary key mismatch"):
            t.assert_same_structure(
                defaultdict(default_value, [("a", 1), ("b", 2)]),
                defaultdict(default_value, [("a", 10)]),
            )
        with self.assertRaisesRegex(ValueError, "[Dd]ictionary key mismatch"):
            t.assert_same_structure(
                OrderedDict([("a", 1), ("b", 2)]),
                OrderedDict([("a", 10)]),
            )

        # Mismatched keys.
        with self.assertRaisesRegex(ValueError, "[Dd]ictionary key mismatch"):
            t.assert_same_structure(
                {"a": 1},
                {"b": 2},
            )
        with self.assertRaisesRegex(ValueError, "[Dd]ictionary key mismatch"):
            t.assert_same_structure(
                defaultdict(default_value, [("a", 1)]),
                defaultdict(default_value, [("b", 2)]),
            )
        with self.assertRaisesRegex(ValueError, "[Dd]ictionary key mismatch"):
            t.assert_same_structure(
                OrderedDict([("a", 1)]),
                OrderedDict([("b", 2)]),
            )

        # Mismatched key count and keys with TrackedDict.
        with self.assertRaisesRegex(ValueError, "Mismatch custom node data"):
            t.assert_same_structure(
                TrackedDict({"a": 1, "b": 2}),
                TrackedDict({"a": 1}),
            )
        with self.assertRaisesRegex(ValueError, "Mismatch custom node data"):
            t.assert_same_structure(
                TrackedDict({"a": 1}),
                TrackedDict({"b": 2}),
            )

    def test_assert_same_paths(self, t):
        # Non-nested.
        t.assert_same_paths(1, 10)

        # Standard structures.
        t.assert_same_paths((), ())
        t.assert_same_paths((1,), (10,))
        t.assert_same_paths((1, 2), (10, 20))
        t.assert_same_paths([], [])
        t.assert_same_paths([1], [10])
        t.assert_same_paths([1, 2], [10, 20])
        t.assert_same_paths(deque([]), deque([]))
        t.assert_same_paths(deque([1]), deque([10]))
        t.assert_same_paths(deque([1, 2]), deque([10, 20]))
        t.assert_same_paths(Empty(), Empty())
        t.assert_same_paths(Point(y=2, x=1), Point(x=10, y=20))
        t.assert_same_paths({}, {})
        t.assert_same_paths({"a": 1}, {"a": 10})
        t.assert_same_paths({"b": None, "a": None}, {"a": 10, "b": 20})
        t.assert_same_paths(OrderedDict(), OrderedDict())
        t.assert_same_paths(OrderedDict([("a", 1)]), OrderedDict([("a", 10)]))
        t.assert_same_paths(
            OrderedDict([("b", 2), ("a", 1)]),
            OrderedDict([("a", 10), ("b", 20)]),
        )
        t.assert_same_paths(defaultdict(default_value), defaultdict(default_value))
        t.assert_same_paths(
            defaultdict(default_value, [("a", 1)]),
            defaultdict(default_value, [("a", 10)]),
        )
        t.assert_same_paths(
            defaultdict(default_value, [("b", 2), ("a", 1)]),
            defaultdict(default_value, [("a", 1), ("b", 2)]),
        )

        # Keras tracking wrappers.
        t.assert_same_paths(
            TrackedList([]),
            TrackedList([]),
        )
        t.assert_same_paths(
            TrackedList([1]),
            TrackedList([10]),
        )
        t.assert_same_paths(
            TrackedList([1, 2]),
            TrackedList([10, 20]),
        )
        t.assert_same_paths(
            TrackedSet([]),
            TrackedSet([]),
        )
        t.assert_same_paths(
            TrackedSet([1]),
            TrackedSet([10]),
        )
        t.assert_same_paths(
            TrackedSet([1, 2]),
            TrackedSet([10, 20]),
        )
        t.assert_same_paths(
            TrackedDict({}),
            TrackedDict({}),
        )
        t.assert_same_paths(
            TrackedDict({"a": 1}),
            TrackedDict({"a": 10}),
        )
        t.assert_same_paths(
            TrackedDict({"b": 2, "a": 1}),
            TrackedDict({"a": 10, "b": 20}),
        )

        # Deeper nested structures.
        t.assert_same_paths(
            (
                {"b": [2, 3], "a": (1,)},
                TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                TrackedSet([7]),
                Point(y=9, x=8),
                np.array([10]),
            ),
            (
                {"b": [20, 30], "a": (10,)},
                TrackedDict({"x": 40, "y": TrackedList([50, 60])}),
                TrackedSet([70]),
                Point(y=90, x=80),
                np.array([100]),
            ),
        )

        # list, tuple, deque and namedtuple have the same paths.
        # Test all 6 combinations:
        # tuple, list.
        t.assert_same_paths((), [])
        t.assert_same_paths([1, 2], (10, 20))
        # tuple, deque.
        t.assert_same_paths((), deque())
        t.assert_same_paths(deque([1, 2]), (10, 20))
        # tuple, namedtuple.
        t.assert_same_paths((), Empty())
        t.assert_same_paths(Point(x=1, y=2), (10, 20))
        # list, deque.
        t.assert_same_paths([], deque())
        t.assert_same_paths(deque([1, 2]), [10, 20])
        # list, namedtuple.
        t.assert_same_paths([], Empty())
        t.assert_same_paths(Point(x=None, y=20), [1, 2])
        # deque, namedtuple.
        t.assert_same_paths(deque(), Empty())
        t.assert_same_paths(Point(x=None, y=20), deque([1, 2]))

        # Equivalent namedtuples.
        t.assert_same_paths(Point(x=1, y=2), OtherPoint(x=None, y=20))

        # Mismatched counts.
        with self.assertRaisesRegex(ValueError, "don't have the same paths"):
            t.assert_same_paths((1, 2), (1,))
        with self.assertRaisesRegex(ValueError, "don't have the same paths"):
            t.assert_same_paths([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "don't have the same paths"):
            t.assert_same_paths(deque([1, 2]), deque([1]))

        # dict, OrderedDict, defaultdict are considered equivalent. Test all 6
        # combinations (3 type combinations plus the order).
        # dict, OrderedDict.
        t.assert_same_paths({"a": 1, "b": 2}, OrderedDict([("b", 20), ("a", 10)]))
        # OrderedDict, dict.
        t.assert_same_paths(OrderedDict([("b", 20), ("a", 10)]), {"a": 1, "b": 2})
        # dict, defaultdict.
        t.assert_same_paths(
            {"a": 1, "b": 2},
            defaultdict(default_value, [("b", 20), ("a", 10)]),
        )
        # defaultdict, dict.
        t.assert_same_paths(
            defaultdict(default_value, [("b", 20), ("a", 10)]),
            {"a": 1, "b": 2},
        )
        # defaultdict, OrderedDict.
        t.assert_same_paths(
            defaultdict(default_value, [("a", 1), ("b", 2)]),
            OrderedDict([("b", 20), ("a", 10)]),
        )
        # OrderedDict, defaultdict.
        t.assert_same_paths(
            OrderedDict([("b", 2), ("a", 1)]),
            defaultdict(default_value, [("a", 10), ("b", 20)]),
        )

        # Two OrderedDicts with same keys but different orders.
        t.assert_same_paths(
            OrderedDict([("b", 2), ("a", 1)]),
            OrderedDict([("a", 10), ("b", 20)]),
        )

        # Keras tracking wrappers are equivalent to the raw structures.
        t.assert_same_paths(TrackedList([1, 2]), list([10, 20]))
        t.assert_same_paths(list([1, 2]), TrackedList([10, 20]))
        t.assert_same_paths(TrackedSet([1, 2]), list([10, 20]))
        t.assert_same_paths(list([1, 2]), TrackedSet([10, 20]))
        t.assert_same_paths(TrackedDict({"b": 2, "a": 1}), {"a": 10, "b": 20})
        t.assert_same_paths({"b": 2, "a": 1}, TrackedDict({"a": 10, "b": 20}))

        # Mismatched keys
        with self.assertRaisesRegex(ValueError, "don't have the same paths"):
            t.assert_same_paths({"a": 1, "b": 2}, {"a": 1})
        with self.assertRaisesRegex(ValueError, "don't have the same paths"):
            t.assert_same_paths(
                defaultdict(default_value, [("a", 1), ("b", 2)]),
                defaultdict(default_value, [("a", 10)]),
            )
        with self.assertRaisesRegex(ValueError, "don't have the same paths"):
            t.assert_same_paths(
                OrderedDict([("a", 1), ("b", 2)]), OrderedDict([("a", 10)])
            )

    def test_traverse_top_down(self, t):
        v = Visitor(lambda x: None if t.is_nested(x) else x + 10)

        # Non-nested.
        self.assertEqualStrict(t.traverse(v, 1), 11)
        self.assertEqualStrict(v.visited(), [1])

        # Standard structures.
        self.assertEqualStrict(t.traverse(v, ()), ())
        self.assertEqualStrict(v.visited(), [()])

        self.assertEqualStrict(t.traverse(v, (1,)), (11,))
        self.assertEqualStrict(v.visited(), [(1,), 1])

        self.assertEqualStrict(t.traverse(v, (1, 2)), (11, 12))
        self.assertEqualStrict(v.visited(), [(1, 2), 1, 2])

        self.assertEqualStrict(t.traverse(v, []), [])
        self.assertEqualStrict(v.visited(), [[]])

        self.assertEqualStrict(t.traverse(v, [1]), [11])
        self.assertEqualStrict(v.visited(), [[1], 1])

        self.assertEqualStrict(t.traverse(v, [1, 2]), [11, 12])
        self.assertEqualStrict(v.visited(), [[1, 2], 1, 2])

        self.assertEqualStrict(t.traverse(v, deque([])), deque([]))
        self.assertEqualStrict(v.visited(), [deque([])])

        self.assertEqualStrict(t.traverse(v, deque([1])), deque([11]))
        self.assertEqualStrict(v.visited(), [deque([1]), 1])

        self.assertEqualStrict(t.traverse(v, deque([1, 2])), deque([11, 12]))
        self.assertEqualStrict(v.visited(), [deque([1, 2]), 1, 2])

        self.assertEqualStrict(t.traverse(v, Empty()), Empty())
        self.assertEqualStrict(v.visited(), [Empty()])

        self.assertEqualStrict(t.traverse(v, Point(y=2, x=1)), Point(x=11, y=12))
        self.assertEqualStrict(v.visited(), [Point(x=1, y=2), 1, 2])

        self.assertEqualStrict(t.traverse(v, {}), {})
        self.assertEqualStrict(v.visited(), [{}])

        self.assertEqualStrict(t.traverse(v, {"a": 1}), {"a": 11})
        self.assertEqualStrict(v.visited(), [{"a": 1}, 1])

        self.assertEqualStrict(t.traverse(v, {"b": 2, "a": 1}), {"a": 11, "b": 12})
        self.assertEqualStrict(v.visited(), [{"a": 1, "b": 2}, 1, 2])

        self.assertEqualStrict(t.traverse(v, OrderedDict()), OrderedDict())
        self.assertEqualStrict(v.visited(), [OrderedDict()])

        self.assertEqualStrict(
            t.traverse(v, OrderedDict([("a", 1)])), OrderedDict([("a", 11)])
        )
        self.assertEqualStrict(v.visited(), [OrderedDict([("a", 1)]), 1])

        self.assertEqualStrict(
            t.traverse(v, OrderedDict([("b", 2), ("a", 1)])),
            OrderedDict([("b", 12), ("a", 11)]),
        )
        self.assertEqualStrict(v.visited(), [OrderedDict([("b", 2), ("a", 1)]), 2, 1])

        self.assertEqualStrict(
            t.traverse(v, defaultdict(default_value)),
            defaultdict(default_value),
        )
        self.assertEqualStrict(v.visited(), [defaultdict(default_value)])

        self.assertEqualStrict(
            t.traverse(v, defaultdict(default_value, [("a", 1)])),
            defaultdict(default_value, [("a", 11)]),
        )
        self.assertEqualStrict(v.visited(), [defaultdict(default_value, [("a", 1)]), 1])

        self.assertEqualStrict(
            t.traverse(v, defaultdict(default_value, [("b", 2), ("a", 1)])),
            defaultdict(default_value, [("a", 11), ("b", 12)]),
        )
        self.assertEqualStrict(
            v.visited(),
            [defaultdict(default_value, [("a", 1), ("b", 2)]), 1, 2],
        )

        # Keras tracking wrappers.
        self.assertEqualStrict(t.traverse(v, TrackedList([])), TrackedList([]))
        self.assertEqualStrict(v.visited(), [TrackedList([])])

        self.assertEqualStrict(t.traverse(v, TrackedList([1])), TrackedList([11]))
        self.assertEqualStrict(v.visited(), [TrackedList([1]), 1])

        self.assertEqualStrict(t.traverse(v, TrackedList([1, 2])), TrackedList([11, 12]))
        self.assertEqualStrict(v.visited(), [TrackedList([1, 2]), 1, 2])

        self.assertEqualStrict(t.traverse(v, TrackedSet([])), TrackedSet([]))
        self.assertEqualStrict(v.visited(), [TrackedSet([])])

        self.assertEqualStrict(t.traverse(v, TrackedSet([1])), TrackedSet([11]))
        self.assertEqualStrict(v.visited(), [TrackedSet([1]), 1])

        self.assertEqualStrict(t.traverse(v, TrackedSet([1, 2])), TrackedSet([11, 12]))
        visited = v.visited()
        self.assertEqualStrict(visited[0], TrackedSet([1, 2]))
        self.assertEqualStrict(sorted(visited[1:]), [1, 2])

        self.assertEqualStrict(
            t.traverse(v, TrackedDict()),
            TrackedDict(),
        )
        self.assertEqualStrict(v.visited(), [TrackedDict()])

        self.assertEqualStrict(
            t.traverse(v, TrackedDict({"a": 1})),
            TrackedDict({"a": 11}),
        )
        self.assertEqualStrict(v.visited(), [TrackedDict({"a": 1}), 1])

        self.assertEqualStrict(
            t.traverse(v, TrackedDict({"b": 2, "a": 1})),
            TrackedDict({"a": 11, "b": 12}),
        )
        self.assertEqualStrict(v.visited(), [TrackedDict({"a": 1, "b": 2}), 1, 2])

        # Deeper nested structures.
        self.assertEqualStrict(
            t.traverse(
                v,
                (
                    {"b": [2, 3], "a": (1,)},
                    TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                    TrackedSet([7]),
                    Point(y=9, x=8),
                    np.array([10]),
                ),
            ),
            (
                {"b": [12, 13], "a": (11,)},
                TrackedDict({"x": 14, "y": TrackedList([15, 16])}),
                TrackedSet([17]),
                Point(y=19, x=18),
                np.array([20]),
            ),
        )
        self.assertEqualStrict(
            v.visited(),
            [
                (
                    {"b": [2, 3], "a": (1,)},
                    TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                    TrackedSet([7]),
                    Point(y=9, x=8),
                    np.array([10]),
                ),
                {"b": [2, 3], "a": (1,)},
                (1,),
                1,
                [2, 3],
                2,
                3,
                TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                4,
                TrackedList([5, 6]),
                5,
                6,
                TrackedSet([7]),
                7,
                Point(x=8, y=9),
                8,
                9,
                np.array([10]),
            ],
        )

        # Error cases.
        with self.assertRaisesRegex(TypeError, "callable"):
            t.traverse("bad", [1, 2])

        # Children are not explored if structure is replaced with a leaf.
        v = Visitor(lambda x: "X" if isinstance(x, tuple) else None)
        self.assertEqualStrict(
            t.traverse(v, [(1, [2]), [3, (4, 5, 6)]]),
            ["X", [3, "X"]],
        )
        self.assertEqualStrict(
            v.visited(),
            [
                [(1, [2]), [3, (4, 5, 6)]],
                (1, [2]),
                [3, (4, 5, 6)],
                3,
                (4, 5, 6),
            ],
        )

        # Children are not explored if structure is replaced with structure.
        v = Visitor(lambda x: ("a", "b") if isinstance(x, tuple) else None)
        self.assertEqualStrict(
            t.traverse(v, [(1, [2]), [3, (4, 5, 6)]]),
            [("a", "b"), [3, ("a", "b")]],
        )
        self.assertEqualStrict(
            v.visited(),
            [
                [(1, [2]), [3, (4, 5, 6)]],
                (1, [2]),
                [3, (4, 5, 6)],
                3,
                (4, 5, 6),
            ],
        )

        # MAP_TO_NONE.
        v = Visitor(lambda x: MAP_TO_NONE if isinstance(x, tuple) else None)
        self.assertEqualStrict(
            t.traverse(v, [(1, [2]), [3, (4, 5, 6)]]),
            [None, [3, None]],
        )
        self.assertEqualStrict(
            v.visited(),
            [
                [(1, [2]), [3, (4, 5, 6)]],
                (1, [2]),
                [3, (4, 5, 6)],
                3,
                (4, 5, 6),
            ],
        )

    def test_traverse_bottom_up(self, t):
        v = Visitor(lambda x: None if t.is_nested(x) else x + 10)
        traverse_u = functools.partial(t.traverse, top_down=False)

        # Non-nested.
        self.assertEqualStrict(traverse_u(v, 1), 11)
        self.assertEqualStrict(v.visited(), [1])

        # Standard structures.
        self.assertEqualStrict(traverse_u(v, ()), ())
        self.assertEqualStrict(v.visited(), [()])

        self.assertEqualStrict(traverse_u(v, (1,)), (11,))
        self.assertEqualStrict(v.visited(), [1, (11,)])

        self.assertEqualStrict(traverse_u(v, (1, 2)), (11, 12))
        self.assertEqualStrict(v.visited(), [1, 2, (11, 12)])

        self.assertEqualStrict(traverse_u(v, []), [])
        self.assertEqualStrict(v.visited(), [[]])

        self.assertEqualStrict(traverse_u(v, [1]), [11])
        self.assertEqualStrict(v.visited(), [1, [11]])

        self.assertEqualStrict(traverse_u(v, [1, 2]), [11, 12])
        self.assertEqualStrict(v.visited(), [1, 2, [11, 12]])

        self.assertEqualStrict(traverse_u(v, deque([])), deque([]))
        self.assertEqualStrict(v.visited(), [deque([])])

        self.assertEqualStrict(traverse_u(v, deque([1])), deque([11]))
        self.assertEqualStrict(v.visited(), [1, deque([11])])

        self.assertEqualStrict(traverse_u(v, deque([1, 2])), deque([11, 12]))
        self.assertEqualStrict(v.visited(), [1, 2, deque([11, 12])])

        self.assertEqualStrict(traverse_u(v, Empty()), Empty())
        self.assertEqualStrict(v.visited(), [Empty()])

        self.assertEqualStrict(traverse_u(v, Point(y=2, x=1)), Point(x=11, y=12))
        self.assertEqualStrict(v.visited(), [1, 2, Point(x=11, y=12)])

        self.assertEqualStrict(traverse_u(v, {}), {})
        self.assertEqualStrict(v.visited(), [{}])

        self.assertEqualStrict(traverse_u(v, {"a": 1}), {"a": 11})
        self.assertEqualStrict(v.visited(), [1, {"a": 11}])

        self.assertEqualStrict(traverse_u(v, {"b": 2, "a": 1}), {"a": 11, "b": 12})
        self.assertEqualStrict(v.visited(), [1, 2, {"a": 11, "b": 12}])

        self.assertEqualStrict(traverse_u(v, OrderedDict()), OrderedDict())
        self.assertEqualStrict(v.visited(), [OrderedDict()])

        self.assertEqualStrict(
            traverse_u(v, OrderedDict([("a", 1)])), OrderedDict([("a", 11)])
        )
        self.assertEqualStrict(v.visited(), [1, OrderedDict([("a", 11)])])

        self.assertEqualStrict(
            traverse_u(v, OrderedDict([("b", 2), ("a", 1)])),
            OrderedDict([("b", 12), ("a", 11)]),
        )
        self.assertEqualStrict(v.visited(), [2, 1, OrderedDict([("b", 12), ("a", 11)])])

        self.assertEqualStrict(
            traverse_u(v, defaultdict(default_value)),
            defaultdict(default_value),
        )
        self.assertEqualStrict(v.visited(), [defaultdict(default_value)])

        self.assertEqualStrict(
            traverse_u(v, defaultdict(default_value, [("a", 1)])),
            defaultdict(default_value, [("a", 11)]),
        )
        self.assertEqualStrict(v.visited(), [1, defaultdict(default_value, [("a", 11)])])

        self.assertEqualStrict(
            traverse_u(v, defaultdict(default_value, [("b", 2), ("a", 1)])),
            defaultdict(default_value, [("a", 11), ("b", 12)]),
        )
        self.assertEqualStrict(
            v.visited(),
            [1, 2, defaultdict(default_value, [("a", 11), ("b", 12)])],
        )

        # Keras tracking wrappers.
        self.assertEqualStrict(traverse_u(v, TrackedList([])), TrackedList([]))
        self.assertEqualStrict(v.visited(), [TrackedList([])])

        self.assertEqualStrict(traverse_u(v, TrackedList([1])), TrackedList([11]))
        self.assertEqualStrict(v.visited(), [1, TrackedList([11])])

        self.assertEqualStrict(traverse_u(v, TrackedList([1, 2])), TrackedList([11, 12]))
        self.assertEqualStrict(v.visited(), [1, 2, TrackedList([11, 12])])

        self.assertEqualStrict(traverse_u(v, TrackedSet([])), TrackedSet([]))
        self.assertEqualStrict(v.visited(), [TrackedSet([])])

        self.assertEqualStrict(traverse_u(v, TrackedSet([1])), TrackedSet([11]))
        self.assertEqualStrict(v.visited(), [1, TrackedSet([11])])

        self.assertEqualStrict(traverse_u(v, TrackedSet([1, 2])), TrackedSet([11, 12]))
        visited = v.visited()
        self.assertEqualStrict(visited[-1], TrackedSet([11, 12]))
        self.assertEqualStrict(sorted(visited[:-1]), [1, 2])

        self.assertEqualStrict(
            traverse_u(v, TrackedDict()),
            TrackedDict(),
        )
        self.assertEqualStrict(v.visited(), [TrackedDict()])

        self.assertEqualStrict(
            traverse_u(v, TrackedDict({"a": 1})),
            TrackedDict({"a": 11}),
        )
        self.assertEqualStrict(v.visited(), [1, TrackedDict({"a": 11})])

        self.assertEqualStrict(
            traverse_u(v, TrackedDict({"b": 2, "a": 1})),
            TrackedDict({"a": 11, "b": 12}),
        )
        self.assertEqualStrict(v.visited(), [1, 2, TrackedDict({"a": 11, "b": 12})])

        # Deeper nested structures.
        self.assertEqualStrict(
            traverse_u(
                v,
                (
                    {"b": [2, 3], "a": (1,)},
                    TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                    TrackedSet([7]),
                    Point(y=9, x=8),
                    np.array([10]),
                ),
            ),
            (
                {"b": [12, 13], "a": (11,)},
                TrackedDict({"x": 14, "y": TrackedList([15, 16])}),
                TrackedSet([17]),
                Point(y=19, x=18),
                np.array([20]),
            ),
        )
        self.assertEqualStrict(
            v.visited(),
            [
                1,
                (11,),
                2,
                3,
                [12, 13],
                {"b": [12, 13], "a": (11,)},
                4,
                5,
                6,
                TrackedList([15, 16]),
                TrackedDict({"x": 14, "y": TrackedList([15, 16])}),
                7,
                TrackedSet([17]),
                8,
                9,
                Point(x=18, y=19),
                np.array([10]),
                (
                    {"b": [12, 13], "a": (11,)},
                    TrackedDict({"x": 14, "y": TrackedList([15, 16])}),
                    TrackedSet([17]),
                    Point(y=19, x=18),
                    np.array([20]),
                ),
            ],
        )

        # Error cases.
        with self.assertRaisesRegex(TypeError, "callable"):
            traverse_u("bad", [1, 2])

        # Children are not explored if structure is replaced with a leaf.
        v = Visitor(lambda x: "X" if isinstance(x, tuple) else None)
        self.assertEqualStrict(
            traverse_u(v, [(1, [2]), [3, (4, 5, 6)]]),
            ["X", [3, "X"]],
        )
        self.assertEqualStrict(
            v.visited(),
            [
                1,
                2,
                [2],
                (1, [2]),
                3,
                4,
                5,
                6,
                (4, 5, 6),
                [3, "X"],
                ["X", [3, "X"]],
            ],
        )

        # Children are not explored if structure is replaced with structure.
        v = Visitor(lambda x: ("a", "b") if isinstance(x, tuple) else None)
        self.assertEqualStrict(
            traverse_u(v, [(1, [2]), [3, (4, 5, 6)]]),
            [("a", "b"), [3, ("a", "b")]],
        )
        self.assertEqualStrict(
            v.visited(),
            [
                1,
                2,
                [2],
                (1, [2]),
                3,
                4,
                5,
                6,
                (4, 5, 6),
                [3, ("a", "b")],
                [("a", "b"), [3, ("a", "b")]],
            ],
        )

        # MAP_TO_NONE.
        v = Visitor(lambda x: MAP_TO_NONE if isinstance(x, tuple) else None)
        self.assertEqualStrict(
            traverse_u(v, [(1, [2]), [3, (4, 5, 6)]]),
            [None, [3, None]],
        )
        self.assertEqualStrict(
            v.visited(),
            [
                1,
                2,
                [2],
                (1, [2]),
                3,
                4,
                5,
                6,
                (4, 5, 6),
                [3, None],
                [None, [3, None]],
            ],
        )

    def test_lists_to_tuples(self, t):
        self.assertEqualStrict(
            t.lists_to_tuples([1, 2, 3]),
            (1, 2, 3),
        )
        self.assertEqualStrict(
            t.lists_to_tuples([[1], [2, 3]]),
            ((1,), (2, 3)),
        )

        # Deeper nested structures.
        self.assertEqualStrict(
            t.lists_to_tuples(
                (
                    {"b": [2, 3], "a": (1,)},
                    TrackedDict({"x": 4, "y": TrackedList([5, 6])}),
                    TrackedSet([(7, 8, 9)]),
                ),
            ),
            (
                {"b": (2, 3), "a": (1,)},
                TrackedDict({"x": 4, "y": (5, 6)}),
                TrackedSet([(7, 8, 9)]),
            ),
        )
