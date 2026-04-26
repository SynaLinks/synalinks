# Modified from: keras/src/backend/common/global_state_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.backend.common import global_state
from synalinks.src.testing import test_case
from synalinks.src.utils.naming import auto_name


class GlobalStateTest(test_case.TestCase):
    def test_clear_session(self):
        name0 = auto_name("somename")
        self.assertEqual(name0, "somename")
        name1 = auto_name("somename")
        self.assertEqual(name1, "somename_1")
        global_state.clear_session()
        name0 = auto_name("somename")
        self.assertEqual(name0, "somename")
