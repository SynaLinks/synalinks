# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.saving import serialization_lib


class SafeModeTest(testing.TestCase):
    def test_safe_mode_refuses_non_synalinks_modules(self):
        config = {
            "module": "subprocess",
            "class_name": "function",
            "config": "getoutput",
            "registered_name": "getoutput",
        }
        with self.assertRaisesRegex(TypeError, "safe_mode"):
            serialization_lib.deserialize_synalinks_object(config)
        fn = serialization_lib.deserialize_synalinks_object(config, safe_mode=False)
        self.assertEqual(fn.__name__, "getoutput")
