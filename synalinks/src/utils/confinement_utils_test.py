# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.utils import confinement_utils


@unittest.skipUnless(sys.platform.startswith("linux"), "the Linux user-namespace probe")
class ConfinementProbeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        confinement_utils.confine_smoke_cache = None

    def tearDown(self):
        confinement_utils.confine_smoke_cache = None
        super().tearDown()

    def test_probe_ignores_this_process_at_fork_hooks(self):
        # LanceDB restarts its runtime threads in a forked child: a probe run
        # in a fork saw a multithreaded child and wrongly said "cannot confine".
        expected = confinement_utils.confinement_available()
        confinement_utils.confine_smoke_cache = None
        active = [True]  # an at-fork hook cannot be unregistered: switch it off

        def start_a_thread():
            if active[0]:
                threading.Thread(target=threading.Event().wait, daemon=True).start()

        os.register_at_fork(after_in_child=start_a_thread)
        try:
            self.assertEqual(confinement_utils.confinement_available(), expected)
        finally:
            active[0] = False

    def test_a_refused_handshake_cannot_confine(self):
        refused = subprocess.CompletedProcess(args=[], returncode=1)
        with patch.object(confinement_utils.subprocess, "run", return_value=refused):
            ok, reason = confinement_utils.confinement_available()
        if confinement_utils.confine_smoke_cache is None:
            self.skipTest(f"an earlier check already fails here: {reason}")
        self.assertFalse(ok)
        self.assertIn("user-namespace", reason)
