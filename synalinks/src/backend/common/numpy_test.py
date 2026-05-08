# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as raw_np

from synalinks.src import testing
from synalinks.src.backend import floatx
from synalinks.src.backend.common import numpy as np


class DtypeHarmonizationTest(testing.TestCase):
    """All float-returning primitives must coerce results to ``floatx()``."""

    def _assert_floatx(self, value):
        self.assertEqual(raw_np.asarray(value).dtype, raw_np.dtype(floatx()))

    def test_convert_to_tensor_int_input_becomes_floatx(self):
        out = np.convert_to_tensor([1, 2, 3])
        self._assert_floatx(out)

    def test_exp(self):
        out = np.exp([0.0, 1.0])
        self._assert_floatx(out)
        self.assertAlmostEqual(float(out[0]), 1.0)

    def test_log(self):
        out = np.log([1.0, raw_np.e])
        self._assert_floatx(out)
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[1]), 1.0, places=5)

    def test_sqrt(self):
        out = np.sqrt([4.0, 9.0])
        self._assert_floatx(out)
        self.assertAlmostEqual(float(out[0]), 2.0)
        self.assertAlmostEqual(float(out[1]), 3.0)

    def test_abs(self):
        out = np.abs([-1.5, 2.0])
        self._assert_floatx(out)
        self.assertAlmostEqual(float(out[0]), 1.5)

    def test_max(self):
        out = np.max([0.1, 0.7, 0.3])
        self._assert_floatx(out)
        self.assertAlmostEqual(float(out), 0.7)

    def test_min(self):
        out = np.min([0.1, 0.7, 0.3])
        self._assert_floatx(out)
        self.assertAlmostEqual(float(out), 0.1)

    def test_std(self):
        out = np.std([0.0, 1.0])
        self._assert_floatx(out)
        self.assertAlmostEqual(float(out), 0.5)

    def test_std_ddof(self):
        out = np.std([0.0, 1.0], ddof=1)
        self._assert_floatx(out)
        # sample std with ddof=1 of [0, 1] is sqrt(0.5).
        self.assertAlmostEqual(float(out), 0.7071067, places=5)

    def test_clip(self):
        out = np.clip([-2.0, 0.5, 5.0], 0.0, 1.0)
        self._assert_floatx(out)
        self.assertEqual(list(out), [0.0, 0.5, 1.0])

    def test_softmax_pattern(self):
        # Recreates the optimizer softmax pipeline: every intermediate must
        # stay in floatx, no dtype upcasts, no surprises.
        rewards = np.convert_to_tensor([0.1, 0.5, 0.4])
        scaled = rewards / 1.0
        exp = np.exp(scaled - np.max(scaled))
        probabilities = exp / np.sum(exp)
        self._assert_floatx(probabilities)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=5)
