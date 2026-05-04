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
        # Reset default LM/EM in memory only — without touching the
        # persisted `~/.synalinks/synalinks.json` so tests don't clobber a
        # user's saved defaults.
        _config._DEFAULT_LANGUAGE_MODEL = None
        _config._DEFAULT_LANGUAGE_MODEL_IDENTIFIER = None
        _config._DEFAULT_EMBEDDING_MODEL = None
        _config._DEFAULT_EMBEDDING_MODEL_IDENTIFIER = None

    def get_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir))
        return temp_dir
