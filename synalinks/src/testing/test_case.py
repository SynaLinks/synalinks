# Modified from: keras/src/testing/test_case.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import shutil
import tempfile
import unittest

from absl.testing import parameterized
from dotenv import load_dotenv

from synalinks.src.backend import config as _config
from synalinks.src.backend.common.global_state import clear_session


class TestCase(
    unittest.IsolatedAsyncioTestCase, parameterized.TestCase, unittest.TestCase
):
    maxDiff = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        # Load environment variables from .env file
        load_dotenv()
        # clear global state so that test cases are independent
        clear_session(free_memory=False)
        # Reset default LM/EM in memory only, without touching the
        # persisted `~/.synalinks/synalinks.json` so tests don't clobber a
        # user's saved defaults.
        _config._DEFAULT_LANGUAGE_MODEL = None
        _config._DEFAULT_LANGUAGE_MODEL_IDENTIFIER = None
        _config._DEFAULT_EMBEDDING_MODEL = None
        _config._DEFAULT_EMBEDDING_MODEL_IDENTIFIER = None
        _config._DEFAULT_DECISION_MODEL = None
        _config._DEFAULT_DECISION_MODEL_IDENTIFIER = None
        self._zero_retry_backoff()
        self._close_ladybug_adapters_after_test()

    def _close_ladybug_adapters_after_test(self):
        """Close every `LadybugAdapter` a test opens when that test ends.

        Most tests never close their knowledge base, and each open Ladybug
        database reserves 8 TiB of address space until it is released. Left to
        the garbage collector, the suite can run out of it between two
        collections.
        """
        from unittest import mock

        from synalinks.src.knowledge_bases.graph_database_adapters import (
            ladybug_adapter,
        )

        adapters = []
        original_init = ladybug_adapter.LadybugAdapter.__init__

        def tracking_init(adapter, *args, **kwargs):
            adapters.append(adapter)
            original_init(adapter, *args, **kwargs)

        patcher = mock.patch.object(
            ladybug_adapter.LadybugAdapter, "__init__", tracking_init
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: [adapter.close() for adapter in adapters])

    def _zero_retry_backoff(self):
        """Make tenacity retry waits instant for the duration of each test.

        The LM/EM wrappers retry failed `litellm` calls with exponential
        backoff (default `retry=5` -> `1+2+4+8 = 15s`). In tests `litellm` is
        mocked, so a call that falls into the retry path (e.g. an
        under-provided `side_effect` list) would otherwise sleep the full real
        backoff and stack across calls, turning sub-second tests into 15s+
        ones. Retry *logic* is untouched (attempt counts, fallback selection);
        only the *wait* is zeroed, and only in-process. Production defaults are
        unaffected.
        """
        from unittest import mock

        import tenacity

        from synalinks.src.modules.decision_models import decision_model
        from synalinks.src.modules.embedding_models import embedding_model
        from synalinks.src.modules.language_models import language_model

        def _instant_wait(max_wait=60.0):
            return tenacity.wait_fixed(0)

        for module in (language_model, embedding_model, decision_model):
            patcher = mock.patch.object(module, "rate_limit_aware_wait", _instant_wait)
            patcher.start()
            self.addCleanup(patcher.stop)

    def get_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir))
        return temp_dir
