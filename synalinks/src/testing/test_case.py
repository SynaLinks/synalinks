# Modified from: keras/src/testing/test_case.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import shutil
import tempfile
import unittest

from absl.testing import parameterized

from synalinks.src.backend.common.global_state import clear_session
from synalinks.src.backend.config import disable_telemetry


class TestCase(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase, unittest.TestCase
):
    maxDiff = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        # clear global state so that test cases are independent
        clear_session(free_memory=False)
        disable_telemetry()

    def get_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir))
        return temp_dir
