# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy

from synalinks.src import testing
from synalinks.src.utils.tracking import TrackedDict
from synalinks.src.utils.tracking import TrackedList
from synalinks.src.utils.tracking import TrackedSet
from synalinks.src.utils.tracking import Tracker


class TrackingTest(testing.TestCase):
    def test_deepcopy_drops_the_tracker(self):
        # Once a module is built its tracker holds the module graph, so a
        # deep copy that reconstructs the tracker recurses back into the
        # container and dies. Copies are plain containers instead.
        tracker = Tracker({"items": (lambda x: False, [])})
        nested = TrackedDict({"schema": {"properties": {"a": [1, 2]}}}, tracker)
        tracker.config["items"][1].append(nested)

        plain = copy.deepcopy(nested)
        self.assertIs(type(plain), dict)
        self.assertIs(type(plain["schema"]), dict)
        self.assertEqual(plain, {"schema": {"properties": {"a": [1, 2]}}})
        plain["schema"]["properties"]["a"].append(3)
        self.assertEqual(nested["schema"]["properties"]["a"], [1, 2])

        self.assertIs(type(copy.deepcopy(TrackedList([1, [2]], tracker))), list)
        self.assertIs(type(copy.deepcopy(TrackedSet({1, 2}, tracker))), set)
