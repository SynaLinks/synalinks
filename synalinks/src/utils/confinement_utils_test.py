# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock

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

    def probe(self, *exit_codes):
        """Run `confinement_available` once per probe exit code.

        The host's other checks are faked to pass.
        """
        codes = iter(exit_codes)

        def no_userns_switch(path, *args, **kwargs):
            raise OSError(path)  # as on distros without the switch

        with (
            mock.patch.object(confinement_utils.platform, "system", return_value="Linux"),
            mock.patch.object(confinement_utils.os.path, "exists", return_value=True),
            mock.patch.object(confinement_utils.os, "unshare", create=True),
            mock.patch.dict(sys.modules, {"mfusepy": types.ModuleType("mfusepy")}),
            mock.patch.object(confinement_utils, "open", no_userns_switch, create=True),
            mock.patch.object(
                confinement_utils.subprocess,
                "run",
                side_effect=lambda *a, **k: subprocess.CompletedProcess(a, next(codes)),
            ),
        ):
            return [confinement_utils.confinement_available() for _ in exit_codes]

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
        [(ok, reason)] = self.probe(1)
        self.assertFalse(ok)
        self.assertIn("user-namespace", reason)

    def test_a_refusal_is_probed_again_and_a_later_pass_counts(self):
        refused, passed, cached = self.probe(92, 0, 92)
        self.assertFalse(refused[0])
        self.assertIn("writing /proc/self/uid_map was refused", refused[1])
        self.assertEqual(passed, (True, ""))
        # Once confinement worked it is not probed again.
        self.assertEqual(cached, (True, ""))

    def test_the_refused_step_is_named(self):
        for code, step in confinement_utils.PROBE_STEPS.items():
            [(ok, reason)] = self.probe(code)
            self.assertFalse(ok)
            self.assertIn(step, reason)

    def test_a_probe_killed_by_a_signal_is_a_refusal(self):
        [(ok, reason)] = self.probe(-9)
        self.assertFalse(ok)
        self.assertIn("the probe was refused", reason)
