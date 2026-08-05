# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import gc
import os
import sys
import tempfile
import unittest

from synalinks.src import testing
from synalinks.src.sandboxes.mirage_sandbox import _MACOS_CONFINE_SRC
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox
from synalinks.src.sandboxes.mirage_sandbox import _confinement_available
from synalinks.src.sandboxes.mirage_sandbox import confinement_kind
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import Sandbox

_TIMEOUT = 30.0
_CONFINE_OK, _CONFINE_REASON = _confinement_available()
# The tests below assert namespace-specific behaviour (PID namespaces, seccomp,
# the pivot_root filesystem swap), so they gate on the *backend* rather than on
# confinement being available at all. macOS now reports confinement available
# too, via Seatbelt, which enforces a deliberately different and weaker set;
# `MacosSeatbeltConfineTest` covers what it does guarantee.
_KIND = confinement_kind()
_NAMESPACES_OK = _CONFINE_OK and _KIND == "namespaces"
_SEATBELT_OK = _CONFINE_OK and _KIND == "seatbelt"
_NS_REASON = f"namespace confinement unavailable: {_CONFINE_REASON or _KIND}"


class _SandboxTestCase(testing.TestCase):
    """Base for sandbox tests.

    Most tests build a ``MirageSandbox`` (and forks) as locals and never call
    ``close()``. Each live sandbox holds a FUSE mount, so without cleanup the
    suite accumulates mounts until ``mount_max`` (default 1000) is hit and new
    mounts fail *silently*; the confinement tests then read empty output and
    fail. Forcing a GC pass after every test runs each dropped sandbox's
    finalizer, which releases its mount deterministically between tests.
    """

    def tearDown(self):
        super().tearDown()
        gc.collect()


class MirageSandboxTest(_SandboxTestCase):
    async def test_is_sandbox_subclass(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        self.assertIsInstance(sandbox, Sandbox)

    async def test_run_returns_execution_result(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run("print('hello')")
        self.assertIsInstance(result, ExecutionResult)
        self.assertIn("hello", result.stdout)
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)

    async def test_run_captures_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run("1 / 0")
        self.assertFalse(result.ok)
        self.assertIn("ZeroDivisionError", result.error)

    async def test_traceback_is_trimmed_to_user_frames(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run("1 / 0")
        # Bootstrap/launcher frames are hidden; only the user frame remains.
        self.assertIn('"<sandbox>"', result.stderr)
        self.assertNotIn("dill", result.stderr)
        self.assertNotIn("base64", result.stderr)

    async def test_state_persists_across_run_calls(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run("x = 7")
        result = await sandbox.run("print(x * 6)")
        self.assertIn("42", result.stdout)

    async def test_functions_and_classes_persist(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run("def sq(n):\n    return n * n")
        await sandbox.run("import math\nclass P:\n    v = 3")
        result = await sandbox.run("print(sq(4), math.floor(2.7), P.v)")
        self.assertIn("16 2 3", result.stdout)

    async def test_state_survives_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run("keep = 11")
        await sandbox.run("raise ValueError('boom')")
        result = await sandbox.run("print(keep)")
        self.assertIn("11", result.stdout)

    async def test_inputs_are_bound(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run("print(given + 1)", inputs={"given": 41})
        self.assertIn("42", result.stdout)
        self.assertTrue(result.ok)

    async def test_last_expression_is_captured_as_result(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run("a = 10\nb = 32\na + b")
        self.assertEqual(result.result, 42)

    async def test_result_variable_convention(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run("result = {'k': 5}\nresult")
        self.assertEqual(result.result, {"k": 5})

    async def test_external_functions_bridge_into_sandbox(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)

        async def adder(x, y):
            return {"sum": x + y}

        # Bound tools are called *synchronously* inside the sandbox: no
        # `await` / `asyncio.run(...)` ceremony.
        code = "out = adder(x=3, y=4)\nprint(out['sum'])\n"
        result = await sandbox.run(code, external_functions={"adder": adder})
        self.assertTrue(result.ok, msg=result.error)
        self.assertIn("7", result.stdout)

    async def test_sync_tool_call_returns_value_directly(self):
        # A bare synchronous call returns the host tool's value directly, usable
        # in the same expression, and it actually invokes the host callable.
        called = {"n": 0}

        async def adder(x, y):
            called["n"] += 1
            return {"sum": x + y}

        sandbox = MirageSandbox(timeout=_TIMEOUT)
        # Last expression is the bare sync call result (the `result` convention).
        result = await sandbox.run(
            "adder(x=2, y=5)['sum']",
            external_functions={"adder": adder},
        )
        self.assertTrue(result.ok, msg=result.error)
        self.assertEqual(result.result, 7)
        self.assertEqual(called["n"], 1)

    async def test_bound_functions_persist_across_runs(self):
        captured = {}

        async def submit(result):
            captured["value"] = result
            return {"submitted": result}

        sandbox = MirageSandbox(timeout=_TIMEOUT, external_functions={"submit": submit})
        result = await sandbox.run("submit(result={'answer': 'done'})")
        self.assertTrue(result.ok, msg=result.error)
        self.assertEqual(captured["value"], {"answer": "done"})

    async def test_run_bash(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_bash("echo hi > /t.txt && cat /t.txt")
        self.assertTrue(out["ok"])
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("hi", out["stdout"])

    async def test_run_bash_failure_returns_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_bash("python3 -c 'import sys; sys.exit(7)'")
        self.assertFalse(out["ok"])
        self.assertEqual(out["exit_code"], 7)

    async def test_bash_session_state_persists(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_bash("mkdir -p /work && cd /work")
        out = await sandbox.run_bash("pwd")
        self.assertIn("/work", out["stdout"])

    async def test_run_python_code_tool(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_python_code("print(2 + 2)")
        self.assertTrue(out["ok"])
        self.assertIn("4", out["stdout"])
        self.assertIsNone(out["error"])

    async def test_run_python_code_tool_invalid_code_reports_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_python_code("def broken(")
        self.assertFalse(out["ok"])
        self.assertIsNotNone(out["error"])
        self.assertIn("SyntaxError", out["error"])

    async def test_history_records_runs(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run("a = 1")
        await sandbox.run("print(a)")
        history = sandbox.history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["code"], "a = 1")
        self.assertTrue(history[1]["ok"])

    async def test_reset_clears_state_and_history(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run("secret = 123")
        sandbox.reset()
        self.assertEqual(sandbox.history(), [])
        result = await sandbox.run("print(secret)")
        self.assertFalse(result.ok)
        self.assertIn("NameError", result.error)

    async def test_timeout_surfaces_as_error(self):
        sandbox = MirageSandbox(timeout=1.0)
        result = await sandbox.run("import time\ntime.sleep(5)")
        self.assertFalse(result.ok)
        self.assertIn("TimeoutError", result.error)

    async def test_require_confinement_needs_confine(self):
        # require_confinement is incompatible with an explicit confine=False.
        with self.assertRaises(ValueError):
            MirageSandbox(timeout=_TIMEOUT, require_confinement=True, confine=False)

    async def test_require_confinement_fails_closed_when_unavailable(self):
        # When confinement cannot be established, require_confinement turns the
        # silent unconfined fallback into a hard error.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch.object(
            mirage_sandbox,
            "_confinement_available",
            return_value=(False, "simulated: unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                MirageSandbox(timeout=_TIMEOUT, confine=True, require_confinement=True)

    # -- security surface ---------------------------------------------------

    def test_host_allowlist_matching(self):
        from synalinks.src.sandboxes.mirage_sandbox import _host_allowed

        self.assertTrue(_host_allowed("api.openai.com", ["api.openai.com"]))
        self.assertFalse(_host_allowed("evil.com", ["api.openai.com"]))
        # wildcard matches subdomains and the apex, case/trailing-dot insensitive
        self.assertTrue(_host_allowed("a.b.example.com", ["*.example.com"]))
        self.assertTrue(_host_allowed("EXAMPLE.com.", ["*.example.com"]))
        self.assertFalse(_host_allowed("notexample.com", ["*.example.com"]))
        # an empty allowlist denies everything
        self.assertFalse(_host_allowed("anything.com", []))

    async def test_egress_tool_refuses_offlist_host(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, allowed_hosts=["api.example.com"])
        fetch = sandbox.bound_functions["http_fetch"]
        with self.assertRaises(PermissionError):
            await fetch(url="http://evil.com/")
        with self.assertRaises(PermissionError):
            await fetch(url="file:///etc/passwd")

    def test_reject_private_blocks_internal_addresses(self):
        from synalinks.src.sandboxes.mirage_sandbox import _reject_private

        # loopback / private / link-local (incl. cloud-metadata IP) are refused
        for host in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254"):
            with self.assertRaises(PermissionError):
                _reject_private(host, None, "http")
        # a public address passes and is returned (numeric, so no DNS lookup)
        self.assertEqual(_reject_private("8.8.8.8", None, "https"), "8.8.8.8")

    async def test_egress_pins_validated_ip(self):
        # The connection must dial the IP validated by _reject_private, not
        # re-resolve the hostname (which would reopen the DNS-rebinding window).
        # We allowlist a non-resolving `.invalid` host and pin it to a local
        # server: the fetch only succeeds if the pinned IP is used.
        import http.server
        import threading
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"pinned-ok")

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            sandbox = MirageSandbox(
                timeout=_TIMEOUT, allowed_hosts=["pinned.invalid"], confine=False
            )
            fetch = sandbox.bound_functions["http_fetch"]
            with mock.patch.object(
                mirage_sandbox, "_reject_private", return_value="127.0.0.1"
            ):
                resp = await fetch(url=f"http://pinned.invalid:{port}/")
            self.assertEqual(resp["status"], 200)
            self.assertEqual(resp["body"], "pinned-ok")
        finally:
            server.shutdown()
            server.server_close()

    async def test_egress_tool_blocks_private_target_by_default(self):
        # An allowlisted host that resolves to a private IP is still refused.
        sandbox = MirageSandbox(
            timeout=_TIMEOUT, allowed_hosts=["127.0.0.1"], confine=False
        )
        fetch = sandbox.bound_functions["http_fetch"]
        with self.assertRaises(PermissionError):
            await fetch(url="http://127.0.0.1/")

    async def test_block_private_egress_false_bypasses_ssrf_gate(self):
        # With the SSRF gate off, a private target gets past the check and fails
        # at the connection instead (URLError), not with a PermissionError.
        import urllib.error

        sandbox = MirageSandbox(
            timeout=_TIMEOUT,
            allowed_hosts=["127.0.0.1"],
            block_private_egress=False,
            confine=False,
        )
        fetch = sandbox.bound_functions["http_fetch"]
        with self.assertRaises(urllib.error.URLError):
            await fetch(url="http://127.0.0.1:1/")  # port 1: connection refused

    def test_confinement_unavailable_points_to_wsl_on_windows(self):
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch("platform.system", return_value="Windows"):
            ok, reason = mirage_sandbox._confinement_available()
        self.assertFalse(ok)
        self.assertIn("WSL2", reason)

    def test_native_windows_hint_silent_off_windows(self):
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        mirage_sandbox._WINDOWS_HINTED = False
        with mock.patch("platform.system", return_value="Linux"):
            self.assertIsNone(mirage_sandbox._maybe_warn_native_windows())

    def test_native_windows_hint_warns_once(self):
        import os
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        mirage_sandbox._WINDOWS_HINTED = False
        with mock.patch("platform.system", return_value="Windows"):
            with mock.patch.dict("os.environ", {}, clear=False):
                os.environ.pop("SYNALINKS_NO_WINDOWS_HINT", None)
                with self.assertWarns(RuntimeWarning):
                    msg = mirage_sandbox._maybe_warn_native_windows()
                self.assertIn("WSL2", msg)
                # second call is a no-op (already hinted this process)
                self.assertIsNone(mirage_sandbox._maybe_warn_native_windows())

    def test_native_windows_hint_suppressed_by_env_var(self):
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        mirage_sandbox._WINDOWS_HINTED = False
        with mock.patch("platform.system", return_value="Windows"):
            with mock.patch.dict(
                "os.environ", {"SYNALINKS_NO_WINDOWS_HINT": "1"}, clear=False
            ):
                self.assertIsNone(mirage_sandbox._maybe_warn_native_windows())

    async def test_allowed_hosts_binds_egress_tool(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, allowed_hosts=["api.example.com"])
        self.assertIn("http_fetch", sandbox.bound_functions)

    async def test_granted_capabilities_unconfined(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        caps = sandbox.granted_capabilities()
        self.assertFalse(caps["confined"])
        self.assertFalse(caps["seccomp"])
        self.assertEqual(caps["network"], {"mode": "host"})
        self.assertEqual(caps["tools"], [])
        # Unconfined, the default 'local' runtime is host-reaching by design and
        # is reported as such rather than hidden.
        self.assertEqual(caps["host_runtimes"], ["local"])

    async def test_non_root_mounts_default_to_read_only(self):
        from mirage import RAMResource

        sandbox = MirageSandbox(
            timeout=_TIMEOUT,
            confine=False,
            resources={"/": RAMResource(), "/data": RAMResource()},
        )
        caps = sandbox.granted_capabilities()
        # root scratch stays writable+exec; an external mount is read-only.
        self.assertEqual(caps["mounts"]["/"], "EXEC")
        self.assertEqual(caps["mounts"]["/data"], "READ")

    async def test_explicit_mount_mode_tuple_is_honored(self):
        from mirage import MountMode
        from mirage import RAMResource

        sandbox = MirageSandbox(
            timeout=_TIMEOUT,
            confine=False,
            resources={"/": RAMResource(), "/rw": (RAMResource(), MountMode.WRITE)},
        )
        self.assertEqual(sandbox.granted_capabilities()["mounts"]["/rw"], "WRITE")

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_unvouched_runtime_stripped_under_confine(self):
        # A runtime the sandbox cannot vouch for could execute code on the host
        # outside the confinement prologue, so it is dropped (with a warning)
        # when confinement is active. Requires real confinement: where it is
        # unavailable, confine=True falls back to unconfined and the entry is
        # left intact, so this assertion only holds on a confine-capable host.
        with self.assertWarns(RuntimeWarning):
            sandbox = MirageSandbox(
                timeout=_TIMEOUT,
                confine=True,
                workspace_kwargs={"runtimes": ["vfs", "not-a-vouched-runtime"]},
            )
        self.assertEqual(sandbox.granted_capabilities()["host_runtimes"], [])
        # The vouched entry survives; only the unknown one is removed.
        self.assertEqual(sandbox._workspace_kwargs["runtimes"], ["vfs"])
        sandbox.close()

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_unvouched_runtime_rejected_under_require_confinement(self):
        with self.assertRaises(ValueError):
            MirageSandbox(
                timeout=_TIMEOUT,
                confine=True,
                require_confinement=True,
                workspace_kwargs={"runtimes": ["not-a-vouched-runtime"]},
            )

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_local_runtime_allowed_under_confinement_when_patched(self):
        # 'local' reaches the host, but the run-python patch prepends the
        # confinement prologue to everything it runs, so it is the one
        # host-reaching runtime the sandbox vouches for. This is the default,
        # and it must not trip the guard.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            self.assertTrue(sandbox._run_python_patched)
            self.assertEqual(sandbox.granted_capabilities()["host_runtimes"], [])
        finally:
            sandbox.close()

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_local_runtime_rejected_when_patch_unavailable(self):
        # Fail-closed: if the confinement prologue cannot be installed, 'local'
        # is an unguarded hole and require_confinement must refuse to build.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch.object(
            mirage_sandbox, "_install_run_python_patch", return_value=False
        ):
            with self.assertRaises(ValueError):
                MirageSandbox(timeout=_TIMEOUT, confine=True, require_confinement=True)

    def test_seccomp_filter_builds_for_known_arch(self):
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch("platform.machine", return_value="x86_64"):
            blob = mirage_sandbox._build_seccomp_filter()
        self.assertIsInstance(blob, str)
        import base64

        raw = base64.b64decode(blob)
        # a valid classic-BPF program: whole 8-byte sock_filter instructions
        self.assertEqual(len(raw) % 8, 0)
        # unknown arch -> no filter (confinement still applies without one)
        with mock.patch("platform.machine", return_value="riscv128"):
            self.assertIsNone(mirage_sandbox._build_seccomp_filter())

    # -- filesystem tools ---------------------------------------------------

    async def test_write_then_read_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        written = await sandbox.write_file("/proj/main.py", "print('hi')\nX = 2\n")
        self.assertEqual(written["written"], "/proj/main.py")
        read = await sandbox.read_file("/proj/main.py")
        self.assertEqual(read["content"], "print('hi')\nX = 2\n")
        self.assertEqual(read["total_lines"], 2)

    async def test_read_missing_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        read = await sandbox.read_file("/nope.txt")
        self.assertIn("error", read)

    async def test_read_file_pagination(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/lines.txt", "a\nb\nc\nd\n")
        page = await sandbox.read_file("/lines.txt", offset=2, limit=2)
        self.assertEqual(page["content"], "b\nc\n")
        self.assertEqual(page["start_line"], 2)
        self.assertEqual(page["end_line"], 3)
        self.assertTrue(page["truncated"])

    async def test_list_files_glob(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/src/a.py", "1")
        await sandbox.write_file("/src/sub/b.py", "2")
        await sandbox.write_file("/notes.md", "3")
        listing = await sandbox.list_files("**/*.py")
        self.assertEqual(listing["files"], ["/src/a.py", "/src/sub/b.py"])
        self.assertEqual(listing["total"], 2)

    async def test_list_files_glob_no_matches(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/src/a.py", "1")
        listing = await sandbox.list_files("**/*.md")
        self.assertEqual(listing["files"], [])
        self.assertEqual(listing["total"], 0)

    async def test_list_files_glob_invalid_pattern(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/src/a.py", "1")
        listing = await sandbox.list_files("[")
        self.assertIsInstance(listing, dict)
        if "error" not in listing:
            self.assertEqual(listing["files"], [])
            self.assertEqual(listing["total"], 0)

    async def test_edit_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/c.py", "value = 1\n")
        edit = await sandbox.edit_file("/c.py", "value = 1", "value = 99")
        self.assertEqual(edit["replacements"], 1)
        read = await sandbox.read_file("/c.py")
        self.assertEqual(read["content"], "value = 99\n")

    async def test_edit_file_not_unique(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/d.py", "x\nx\n")
        edit = await sandbox.edit_file("/d.py", "x", "y")
        self.assertIn("error", edit)
        edit_all = await sandbox.edit_file("/d.py", "x", "y", replace_all=True)
        self.assertEqual(edit_all["replacements"], 2)

    async def test_search_files(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/s/a.py", "import os\nprint('found')\n")
        await sandbox.write_file("/s/b.py", "x = 1\n")
        result = await sandbox.search_files("print", glob="**/*.py")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["matches"][0]["path"], "/s/a.py")
        self.assertEqual(result["matches"][0]["line"], 2)

    async def test_search_files_no_matches(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/s/a.py", "import os\n")
        result = await sandbox.search_files("print", glob="**/*.py")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["matches"], [])

    async def test_search_files_invalid_glob(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/s/a.py", "print('found')\n")
        result = await sandbox.search_files("print", glob="[")
        self.assertIsInstance(result, dict)
        if "error" in result:
            self.assertTrue(result["error"])
        else:
            self.assertEqual(result["total"], 0)
            self.assertEqual(result["matches"], [])

    async def test_run_python_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/script.py", "print(3 * 3)\n")
        out = await sandbox.run_python_file("/script.py")
        self.assertTrue(out["ok"])
        self.assertIn("9", out["stdout"])

    async def test_run_python_file_syntax_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/bad_syntax.py", "def broken(:\n    pass\n")
        out = await sandbox.run_python_file("/bad_syntax.py")
        self.assertFalse(out["ok"])
        self.assertTrue("stderr" in out or "error" in out)

    async def test_run_python_file_runtime_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/runtime_error.py", "raise ValueError('boom')\n")
        out = await sandbox.run_python_file("/runtime_error.py")
        self.assertFalse(out["ok"])
        self.assertTrue("stderr" in out or "error" in out)

    async def test_run_python_file_missing(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_python_file("/missing.py")
        self.assertIn("error", out)

    # -- serialization & branching ------------------------------------------

    async def test_dump_load_round_trip(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run("secret = 1234")
        await sandbox.write_file("/keep.txt", "persisted")
        blob = sandbox.dump()
        restored = MirageSandbox.load(blob)
        result = await restored.run("print(secret)")
        self.assertIn("1234", result.stdout)
        read = await restored.read_file("/keep.txt")
        self.assertEqual(read["content"], "persisted")

    async def test_get_config_from_config_round_trip(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, name="cfg")
        await sandbox.run("token = 'abc'")
        config = sandbox.get_config()
        restored = MirageSandbox.from_config(config)
        self.assertEqual(restored.name, "cfg")
        result = await restored.run("print(token)")
        self.assertIn("abc", result.stdout)

    async def test_fork_isolates_filesystem(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/f.txt", "parent")
        child = sandbox.fork()
        await child.write_file("/f.txt", "child")
        await child.write_file("/only_child.txt", "x")
        parent_read = await sandbox.read_file("/f.txt")
        child_read = await child.read_file("/f.txt")
        self.assertEqual(parent_read["content"], "parent")
        self.assertEqual(child_read["content"], "child")
        parent_only = await sandbox.read_file("/only_child.txt")
        self.assertIn("error", parent_only)

    async def test_fork_copy_repl_inherits_namespace(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run("base = 5")
        child = sandbox.fork(copy_repl=True)
        result = await child.run("print(base * 2)")
        self.assertIn("10", result.stdout)
        # Child mutations do not leak back to the parent.
        await child.run("base = 999")
        parent_result = await sandbox.run("print(base)")
        self.assertIn("5", parent_result.stdout)

    async def test_diff_and_changes(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/created.txt", "new")
        diff = sandbox.diff()
        self.assertEqual(
            diff["written"], [{"path": "/created.txt", "kind": "create", "size": 3}]
        )
        changes = sandbox.changes()
        self.assertEqual(changes["written"], ["/created.txt"])

    async def test_save_writes_workdir_as_zip(self):
        import os
        import tempfile
        import zipfile

        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/models.py", "class A:\n    pass\n")
        await sandbox.write_file("/pkg/util.py", "x = 1\n")
        with tempfile.TemporaryDirectory() as tmp:
            # Suffix is appended when missing.
            result = sandbox.save(os.path.join(tmp, "out"))
            self.assertTrue(result["path"].endswith("out.zip"))
            self.assertTrue(os.path.exists(result["path"]))
            self.assertEqual(result["files"], 2)
            with zipfile.ZipFile(result["path"]) as archive:
                names = set(archive.namelist())
                # Members are virtual paths without the leading slash.
                self.assertEqual(names, {"models.py", "pkg/util.py"})
                self.assertEqual(
                    archive.read("models.py").decode(), "class A:\n    pass\n"
                )

    async def test_save_unsupported_on_base_sandbox(self):
        with self.assertRaises(NotImplementedError):
            Sandbox().save("out.zip")

    async def test_patch_renders_git_style_unified_diff(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/app/main.py", "def f():\n    return 1\n")
        await sandbox.write_file("/old.txt", "delete me\n")
        child = sandbox.fork()
        await child.write_file("/app/main.py", "def f():\n    return 2\n")
        await child.write_file("/new.txt", "new")
        await child._delete("/old.txt")
        patch = child.patch()
        # Modify: a real hunk against the fork-point text.
        self.assertIn("diff --git a/app/main.py b/app/main.py", patch)
        self.assertIn("-    return 1", patch)
        self.assertIn("+    return 2", patch)
        # Create and delete use the /dev/null git convention.
        self.assertIn("new file mode 100644", patch)
        self.assertIn("--- /dev/null", patch)
        self.assertIn("deleted file mode 100644", patch)
        self.assertIn("+++ /dev/null", patch)
        # No trailing newline is flagged the way git does.
        self.assertIn("\\ No newline at end of file", patch)
        # ``paths`` restricts output to the selected file.
        only = child.patch(paths=["/new.txt"])
        self.assertIn("b/new.txt", only)
        self.assertNotIn("main.py", only)

    async def test_merge_applies_child_changes(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/base.txt", "base")
        child = sandbox.fork()
        await child.write_file("/new.txt", "from_child")
        report = sandbox.merge(child)
        self.assertIn("/new.txt", report["written"])
        self.assertEqual(report["conflicts"], [])
        read = await sandbox.read_file("/new.txt")
        self.assertEqual(read["content"], "from_child")

    async def test_merge_conflict_refused_without_force(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/shared.txt", "original")
        child = sandbox.fork()
        await child.write_file("/shared.txt", "child_version")
        # Parent diverges on the same path after the fork.
        await sandbox.write_file("/shared.txt", "parent_version")
        report = sandbox.merge(child)
        self.assertIn("/shared.txt", report["conflicts"])
        self.assertIn("/shared.txt", report["skipped"])
        read = await sandbox.read_file("/shared.txt")
        self.assertEqual(read["content"], "parent_version")
        # force applies the child's version.
        forced = sandbox.merge(child, force=True)
        self.assertIn("/shared.txt", forced["written"])
        read = await sandbox.read_file("/shared.txt")
        self.assertEqual(read["content"], "child_version")

    async def test_workdir_seeds_filesystem_host_safe(self):
        import os
        import shutil
        import tempfile

        workdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(workdir, "pkg"))
            with open(os.path.join(workdir, "main.py"), "w") as fh:
                fh.write("print('orig')\n")
            with open(os.path.join(workdir, "pkg", "mod.py"), "w") as fh:
                fh.write("VALUE = 1\n")
            sandbox = MirageSandbox(timeout=_TIMEOUT, workdir=workdir)
            listing = await sandbox.list_files("**/*.py")
            self.assertEqual(listing["files"], ["/main.py", "/pkg/mod.py"])
            # Editing in the sandbox never touches the host directory.
            await sandbox.write_file("/main.py", "print('edited')\n")
            with open(os.path.join(workdir, "main.py")) as fh:
                self.assertEqual(fh.read(), "print('orig')\n")
            self.assertEqual(sandbox.changes()["written"], ["/main.py"])
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


@unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
class MirageSandboxConfineTest(_SandboxTestCase):
    """In-process confinement (FUSE + user-namespace pivot), Linux-only."""

    async def test_confine_extra_binds_imports_host_dir(self):
        # A host directory bound via extra_binds is visible read-only inside the
        # confined sandbox and importable once added to sys.path.
        import os
        import shutil
        import tempfile

        host_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(host_dir, "hostlibxyz.py"), "w") as fh:
                fh.write("VALUE = 4242\n")
            sandbox = MirageSandbox(
                timeout=_TIMEOUT, confine=True, extra_binds=[host_dir]
            )
            try:
                result = await sandbox.run(
                    f"import sys\nsys.path.insert(0, {host_dir!r})\n"
                    "import hostlibxyz\nprint(hostlibxyz.VALUE)\n"
                )
                self.assertTrue(result.ok, msg=result.error)
                self.assertIn("4242", result.stdout)
                # ...and it's read-only (host import-poisoning guard holds).
                ro = await sandbox.run(
                    f"try:\n"
                    f"    open({os.path.join(host_dir, 'hostlibxyz.py')!r}, 'a')"
                    f".write('x')\n"
                    f"    print('WRITABLE')\n"
                    f"except OSError:\n"
                    f"    print('READONLY')\n"
                )
                self.assertIn("READONLY", ro.stdout)
            finally:
                sandbox.close()
        finally:
            shutil.rmtree(host_dir, ignore_errors=True)

    async def test_confine_runs_and_persists_state(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            self.assertTrue(sandbox._confine)
            await sandbox.run("x = 41")
            result = await sandbox.run("print(x + 1)")
            self.assertTrue(result.ok, msg=result.error)
            self.assertIn("42", result.stdout)
            # third-party / stdlib imports still resolve inside the pivot
            imp = await sandbox.run("import json, math\nprint(math.floor(2.5))")
            self.assertTrue(imp.ok, msg=imp.error)
            self.assertIn("2", imp.stdout)
        finally:
            sandbox.close()

    async def test_confine_hides_host_filesystem(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            result = await sandbox.run("import os\nprint(os.path.exists('/etc/passwd'))")
            self.assertIn("False", result.stdout)
        finally:
            sandbox.close()

    async def test_confine_pid_namespace_hides_host_processes(self):
        # The PID namespace means the confined /proc shows only namespaced PIDs:
        # the snippet is PID 1 and no host processes are visible (defense in
        # depth so host /proc/<pid>/root and /environ aren't even reachable).
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            result = await sandbox.run(
                "import os\n"
                "pids = [int(p) for p in os.listdir('/proc') if p.isdigit()]\n"
                "print('mypid', os.getpid())\n"
                "print('maxpid', max(pids))\n"
                "print('count', len(pids))\n"
            )
            self.assertTrue(result.ok, msg=result.error)
            # PID 1 in the new namespace, and only a handful of namespaced PIDs
            # (no sprawling host process table).
            self.assertIn("mypid 1", result.stdout)
            lines = dict(ln.split(" ", 1) for ln in result.stdout.strip().splitlines())
            self.assertLessEqual(int(lines["maxpid"]), 50)
        finally:
            sandbox.close()

    async def test_confine_cuts_network(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            result = await sandbox.run(
                "import socket\n"
                "try:\n"
                "    socket.create_connection(('1.1.1.1', 80), 2)\n"
                "    print('REACHABLE')\n"
                "except OSError:\n"
                "    print('CUT')\n"
            )
            self.assertIn("CUT", result.stdout)
        finally:
            sandbox.close()

    async def test_confine_python_shares_virtual_filesystem(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            # a file written via the tool is visible to python at the same path
            await sandbox.write_file("/data.txt", "virtual-content")
            read = await sandbox.run("print(open('/data.txt').read().strip())")
            self.assertIn("virtual-content", read.stdout)
            # a file python writes is visible to the tool
            await sandbox.run("open('/out.txt', 'w').write('from-python')")
            tool = await sandbox.read_file("/out.txt")
            self.assertEqual(tool["content"], "from-python")
        finally:
            sandbox.close()

    async def test_confine_host_tool_bridge_still_works(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)

        async def adder(x, y):
            return {"sum": x + y}

        try:
            # The host-tool bridge must also work synchronously under confinement.
            code = "print(adder(x=2, y=3)['sum'])\n"
            result = await sandbox.run(code, external_functions={"adder": adder})
            self.assertTrue(result.ok, msg=result.error)
            self.assertIn("5", result.stdout)
        finally:
            sandbox.close()

    async def test_confine_covers_run_bash_python(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            await sandbox.write_file("/v.txt", "virtual")
            await sandbox.write_file(
                "/probe.py",
                "import os\n"
                "print('passwd', os.path.exists('/etc/passwd'))\n"
                "print('vtxt', open('/v.txt').read())\n",
            )
            out = await sandbox.run_bash("python3 /probe.py")
            self.assertTrue(out["ok"], msg=out["stderr"])
            # host filesystem hidden, virtual filesystem visible at the same path
            self.assertIn("passwd False", out["stdout"])
            self.assertIn("vtxt virtual", out["stdout"])
            # a file the confined run_bash python writes is visible to the tools
            await sandbox.write_file(
                "/w.py", "open('/from_bash.txt', 'w').write('written')\n"
            )
            await sandbox.run_bash("python3 /w.py")
            self.assertEqual(
                (await sandbox.read_file("/from_bash.txt"))["content"], "written"
            )
        finally:
            sandbox.close()

    async def test_unconfined_run_bash_python_is_not_patched(self):
        # The _run_python patch is global once a confined sandbox exists, but it
        # must be a no-op for an unconfined sandbox.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        try:
            await sandbox.write_file(
                "/p.py", "import os\nprint(os.path.exists('/etc/passwd'))\n"
            )
            out = await sandbox.run_bash("python3 /p.py")
            self.assertIn("True", out["stdout"])
        finally:
            sandbox.close()

    async def test_confine_runtime_binds_are_read_only(self):
        # The Python runtime (venv / stdlib) is bound read-only, so confined
        # code cannot poison files the host later imports outside the sandbox.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            import os as _os

            target = _os.path.join(_os.path.dirname(_os.__file__), "os.py")
            result = await sandbox.run(
                "import os\n"
                f"try:\n"
                f"    open({target!r}, 'a').write('x')\n"
                f"    print('WRITABLE')\n"
                f"except OSError as exc:\n"
                f"    print('READONLY')\n"
            )
            self.assertTrue(result.ok, msg=result.error)
            self.assertIn("READONLY", result.stdout)
        finally:
            sandbox.close()

    async def test_confine_seccomp_blocks_denied_syscall(self):
        # ``unshare`` is on the denylist; under seccomp it returns EPERM, so
        # os.unshare raises instead of succeeding (the no-op flags=0 call).
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True, seccomp=True)
        try:
            self.assertTrue(sandbox.granted_capabilities()["seccomp"])
            result = await sandbox.run(
                "import os\n"
                "try:\n"
                "    os.unshare(0)\n"
                "    print('ALLOWED')\n"
                "except OSError:\n"
                "    print('BLOCKED')\n"
            )
            self.assertTrue(result.ok, msg=result.error)
            self.assertIn("BLOCKED", result.stdout)
        finally:
            sandbox.close()

    async def test_confine_without_seccomp_allows_syscall(self):
        # With seccomp disabled the same call is not blocked by a filter; proves
        # the block above comes from seccomp, not the namespace.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True, seccomp=False)
        try:
            self.assertFalse(sandbox.granted_capabilities()["seccomp"])
            result = await sandbox.run(
                "import os\n"
                "try:\n"
                "    os.unshare(0)\n"
                "    print('ALLOWED')\n"
                "except OSError:\n"
                "    print('BLOCKED')\n"
            )
            self.assertTrue(result.ok, msg=result.error)
            self.assertIn("ALLOWED", result.stdout)
        finally:
            sandbox.close()

    async def test_confine_granted_capabilities(self):
        sandbox = MirageSandbox(
            timeout=_TIMEOUT, confine=True, allowed_hosts=["api.example.com"]
        )
        try:
            caps = sandbox.granted_capabilities()
            self.assertTrue(caps["confined"])
            self.assertTrue(caps["seccomp"])
            self.assertTrue(caps["read_only_runtime"])
            self.assertEqual(caps["network"]["mode"], "allowlist")
            self.assertIn("http_fetch", caps["tools"])
        finally:
            sandbox.close()

    async def test_require_confinement_runs_when_available(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True, require_confinement=True)
        try:
            self.assertTrue(sandbox._confine)
            result = await sandbox.run("print(40 + 2)")
            self.assertTrue(result.ok, msg=result.error)
            self.assertIn("42", result.stdout)
        finally:
            sandbox.close()

    async def test_fork_confine_none_inherits_parent(self):
        # confine=None (the subagent default) inherits the parent's confinement:
        # confined parent -> confined fork; unconfined parent -> unconfined fork.
        confined = MirageSandbox(timeout=_TIMEOUT, confine=True)
        unconfined = MirageSandbox(timeout=_TIMEOUT, confine=False)
        c_child = u_child = None
        try:
            c_child = confined.fork(confine=None)
            u_child = unconfined.fork(confine=None)
            self.assertTrue(c_child._confine)
            self.assertFalse(u_child._confine)
        finally:
            for sandbox in (confined, unconfined, c_child, u_child):
                if sandbox is not None:
                    sandbox.close()

    async def test_confined_fork_inherits_parent_security_posture(self):
        # A subagent's confined fork transparently carries the parent's whole
        # security config (egress allowlist, SSRF guard, extra binds, mount
        # modes, fail-closed), so confinement isn't something the subagent opts
        # into; it inherits it.
        parent = MirageSandbox(
            timeout=_TIMEOUT,
            confine=True,
            require_confinement=True,
            allowed_hosts=["api.example.com"],
            extra_binds=["/usr/share"],
        )
        child = None
        try:
            child = parent.fork(name="sub", confine=True)
            self.assertTrue(child._confine)
            pcaps = parent.granted_capabilities()
            ccaps = child.granted_capabilities()
            for key in ("confined", "seccomp", "read_only_runtime", "network"):
                self.assertEqual(ccaps[key], pcaps[key])
            self.assertIn("http_fetch", ccaps["tools"])  # egress tool inherited
            self.assertTrue(child._require_confinement)  # fail-closed propagates
            self.assertEqual(child._extra_binds, ["/usr/share"])
            self.assertTrue(child._block_private_egress)
            # and it is genuinely confined to ITS OWN fork (host hidden)
            r = await child.run("import os\nprint(os.path.exists('/etc/passwd'))")
            self.assertIn("False", r.stdout)
        finally:
            parent.close()
            if child is not None:
                child.close()

    async def test_fork_confine_isolates_child_to_its_fork(self):
        # An unconfined parent can hand a subagent a confined fork: the child is
        # locked to its own forked filesystem; the parent stays unconfined.
        parent = MirageSandbox(timeout=_TIMEOUT, confine=False)
        child = None
        try:
            await parent.write_file("/shared.txt", "parent-data")
            child = parent.fork(confine=True, name="sub")
            self.assertTrue(child._confine)
            # child sees the forked copy at the same path, host hidden
            r = await child.run(
                "import os\n"
                "print('shared', open('/shared.txt').read().strip())\n"
                "print('passwd', os.path.exists('/etc/passwd'))\n"
            )
            self.assertIn("shared parent-data", r.stdout)
            self.assertIn("passwd False", r.stdout)
            # child writes stay in the child fork; the parent never sees them
            await child.run("open('/childonly.txt', 'w').write('x')")
            self.assertIn("error", await parent.read_file("/childonly.txt"))
            # the parent itself is unconfined
            rp = await parent.run("import os\nprint(os.path.exists('/etc/passwd'))")
            self.assertIn("True", rp.stdout)
        finally:
            parent.close()
            if child is not None:
                child.close()


class RunPythonPatchTest(_SandboxTestCase):
    """`_install_run_python_patch` must locate Mirage's python runner across
    versions (it moved modules in mirage-ai 0.0.2) so `run_bash`'s `python3`
    can be confined rather than silently left unconfined."""

    async def test_patch_resolves_runner_on_installed_mirage(self):
        import importlib
        import warnings

        from synalinks.src.sandboxes import mirage_sandbox as ms

        # At least one known runner location must exist in the installed Mirage.
        # A target's qualname may name a class method (0.0.4+ runtime table) or
        # a module-level function, so resolve it the same way the patch does.
        resolved = None
        for mod_name, qualname in ms._RUN_PYTHON_TARGETS:
            try:
                owner = importlib.import_module(mod_name)
            except Exception:
                continue
            *parents, attr_name = qualname.split(".")
            for parent in parents:
                owner = getattr(owner, parent, None)
                if owner is None:
                    break
            if owner is not None and getattr(owner, attr_name, None) is not None:
                resolved = (owner, attr_name)
                break
        if resolved is None:
            self.skipTest("no known mirage python-runner internal is importable")

        mod, attr_name = resolved
        # Snapshot so we can restore the runner (and the install flag) and not
        # leak a double-wrap into other tests.
        original_attr = getattr(mod, attr_name)
        was_patched = ms._run_python_patched
        ms._run_python_patched = False
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ok = ms._install_run_python_patch()
            self.assertTrue(ok)
            self.assertFalse(
                [w for w in caught if "stays unconfined" in str(w.message)],
                msg="patch should not warn when a runner is found",
            )
            self.assertTrue(hasattr(getattr(mod, attr_name), "_mirage_sandbox_original"))
        finally:
            setattr(mod, attr_name, original_attr)
            ms._run_python_patched = was_patched


class InfraSelfHealTest(_SandboxTestCase):
    """A dead FUSE mount / confinement-bootstrap failure makes every `run`
    repeat the same infra error; an agent would loop on it until timeout. `run`
    must detect this, rebuild the workspace, and retry once instead."""

    def test_is_infra_failure_classifies_markers(self):
        from synalinks.src.sandboxes.mirage_sandbox import _is_infra_failure

        # confine bootstrap abort (exit 99) and a dead FUSE backing.
        self.assertTrue(_is_infra_failure("confine-error: OSError(13, ...)", 99))
        self.assertTrue(
            _is_infra_failure("OSError(107, 'Transport endpoint is not connected')", 1)
        )
        # an ordinary snippet error is NOT infrastructure.
        self.assertFalse(_is_infra_failure("Traceback ...\nValueError: nope", 1))
        # success never counts, even if a marker appears in echoed text.
        self.assertFalse(_is_infra_failure("confine-error: (echoed)", 0))

    async def test_run_heals_workspace_and_retries_on_infra_failure(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None):
            calls["exec"] += 1
            if calls["exec"] == 1:
                return ("", "confine-error: OSError(107, 'Transport endpoint ...')", 99)
            return ("healed\n", "", 0)

        sandbox._execute = fake_execute
        sandbox._rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run("print('x')")
        finally:
            sandbox.close()
        # exactly one rebuild, two execute attempts, and the healed run wins.
        self.assertEqual(calls["rebuild"], 1)
        self.assertEqual(calls["exec"], 2)
        self.assertTrue(result.ok, msg=result.error)
        self.assertIn("healed", result.stdout)

    async def test_run_gives_up_after_one_heal(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None):
            calls["exec"] += 1
            return ("", "confine-error: still broken", 99)

        sandbox._execute = fake_execute
        sandbox._rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run("print('x')")
        finally:
            sandbox.close()
        # one heal, two attempts, then the infra error surfaces (no infinite loop).
        self.assertEqual(calls["rebuild"], 1)
        self.assertEqual(calls["exec"], 2)
        self.assertFalse(result.ok)
        self.assertIn("confine-error", result.error)

    async def test_run_bash_heals_workspace_and_retries_on_infra_failure(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None):
            calls["exec"] += 1
            if calls["exec"] == 1:
                return ("", "OSError(107, 'Transport endpoint is not connected')", 99)
            return ("ok\n", "", 0)

        sandbox._execute = fake_execute
        sandbox._rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run_bash("ls /")
        finally:
            sandbox.close()
        self.assertEqual(calls["rebuild"], 1)
        self.assertEqual(calls["exec"], 2)
        self.assertTrue(result["ok"])
        self.assertIn("ok", result["stdout"])

    async def test_run_does_not_heal_on_ordinary_snippet_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None):
            calls["exec"] += 1
            return ("", "Traceback ...\nValueError: boom", 1)

        sandbox._execute = fake_execute
        sandbox._rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run("raise ValueError('boom')")
        finally:
            sandbox.close()
        # a genuine snippet error must not trigger a rebuild/retry.
        self.assertEqual(calls["rebuild"], 0)
        self.assertEqual(calls["exec"], 1)
        self.assertFalse(result.ok)
        self.assertIn("ValueError", result.error)


class SbplProfileTest(testing.TestCase):
    """Unit tests for the generated Seatbelt profile.

    Pure string generation, so these run on every platform. They are the only
    part of the macOS backend that can be checked off a Darwin host, which
    makes them the first line of defence against a profile that silently grants
    more than intended.
    """

    def _build(self, **overrides):
        cfg = {
            "read_paths": ["/usr", "/System"],
            "rw_paths": ["/private/tmp/sbx"],
            "network": False,
        }
        cfg.update(overrides)
        ns = {}
        exec(_MACOS_CONFINE_SRC, ns)
        return ns["_build_sbpl"](cfg)

    def test_profile_is_default_deny(self):
        # Everything else in the profile is an exception to this line; without
        # it the profile grants the world.
        profile = self._build()
        self.assertIn("(version 1)", profile)
        self.assertIn("(deny default)", profile)

    def test_read_paths_are_read_only(self):
        profile = self._build()
        self.assertIn('(allow file-read* (subpath "/usr"))', profile)
        # The runtime set must never become writable: that is what stops the
        # snippet rewriting the interpreter it is about to run under.
        self.assertNotIn('(allow file-read* file-write* (subpath "/usr"))', profile)

    def test_rw_paths_are_writable(self):
        profile = self._build()
        self.assertIn(
            '(allow file-read* file-write* (subpath "/private/tmp/sbx"))', profile
        )

    def test_network_denied_by_default(self):
        profile = self._build()
        # `(deny default)` covers everything; nothing re-opens it.
        self.assertNotIn("(allow network*)", profile)

    def test_host_loopback_is_never_granted(self):
        # macOS has no network namespace, so `(local ip)` would be the *host's*
        # 127.0.0.1: every service bound to localhost would be reachable from
        # code that is supposed to have no network at all. Assert both
        # directions, and under both network settings, since the blanket
        # `(allow network*)` is the only intended way to get out.
        for network in (False, True):
            profile = self._build(network=network)
            self.assertNotIn("(local ip)", profile)
            self.assertNotIn("network-inbound", profile)

    def test_network_allowed_when_requested(self):
        profile = self._build(network=True)
        self.assertIn("(allow network*)", profile)

    def test_rpc_socket_dir_is_allowed_even_without_network(self):
        # The host-tool bridge is a unix socket, which Seatbelt treats as
        # network; without this the bridge would break under the denial above.
        profile = self._build(rpc_socket_dir="/private/tmp/sbx")
        self.assertIn('(allow network-outbound (subpath "/private/tmp/sbx"))', profile)

    def test_process_info_is_denied(self):
        profile = self._build()
        self.assertIn("(deny process-info*)", profile)

    def test_mach_lookup_is_never_unrestricted(self):
        # A bare `(allow mach-lookup)` lets the process ask any daemon to work
        # on its behalf, and the daemon's child does not inherit this profile,
        # so the filesystem and network rules would not apply to it. Every
        # grant must name a specific service.
        profile = self._build()
        self.assertNotIn("(allow mach-lookup)\n", profile)
        for line in profile.splitlines():
            if "mach-lookup" in line:
                self.assertIn("global-name", line)

    def test_no_mach_service_is_granted_by_default(self):
        # The allowlist is empty: what libSystem needs, it looks up during
        # startup, before the profile applies. Each entry is IPC into a
        # privileged daemon, so the list grows only on evidence.
        profile = self._build()
        self.assertNotIn("mach-lookup", profile)
        # Process-spawning brokers are the escape route the allowlist exists to
        # close, so they must never appear whatever else is added.
        self.assertNotIn("launchservicesd", profile)
        self.assertNotIn("com.apple.launchd", profile)

    def test_exec_is_limited_to_the_read_only_runtime_set(self):
        # Host tools stay reachable, as they are on Linux, and a child inherits
        # the profile. What must not be executable is anything the snippet can
        # write: the writable area is not in the exec set.
        profile = self._build(rw_paths=["/private/tmp/sbx"])
        self.assertIn('(allow process-exec (subpath "/usr"))', profile)
        self.assertNotIn('(allow process-exec (subpath "/private/tmp/sbx"))', profile)

    def test_writable_area_is_not_executable(self):
        # Write-xor-execute: a snippet must not be able to drop a dylib into its
        # own scratch dir and `dlopen` it. SBPL is last-match-wins, so the deny
        # has to come *after* the grant it carves into.
        profile = self._build(rw_paths=["/private/tmp/sbx"])
        deny = '(deny file-map-executable (subpath "/private/tmp/sbx"))'
        allow = '(allow file-read* file-write* (subpath "/private/tmp/sbx"))'
        self.assertIn(deny, profile)
        self.assertGreater(profile.index(deny), profile.index(allow))

    def test_symlinked_ancestors_are_traversable(self):
        # Reaching an allowed path means resolving every symlink on the way to
        # it, which `(deny default)` also denies. macOS puts the sandbox's own
        # dirs under `/var/folders/...`, where `/var` is a symlink, so without
        # this the `file-write*` grant below can never match and writes into the
        # sandbox's own area fail while the profile looks correct.
        profile = self._build(rw_paths=["/private/var/folders/ab/cd/T/sbx"])
        for path in ("/var", "/private/var/folders/ab/cd/T", "/private/var/folders"):
            self.assertIn('(allow file-read-metadata (literal "%s"))' % path, profile)

    def test_traversal_grant_is_metadata_only(self):
        # The traversal grants must not become read access: `file-read*` on an
        # ancestor would hand the snippet the contents of everything under it.
        profile = self._build(rw_paths=["/private/var/folders/ab/cd/T/sbx"])
        for line in profile.splitlines():
            if "/private/var/folders/ab/cd" in line and "sbx" not in line:
                self.assertIn("file-read-metadata", line)
        self.assertNotIn('(allow file-read* (subpath "/"))', profile)
        self.assertNotIn('(allow file-read* (subpath "/var"))', profile)

    def test_working_directory_is_traversable(self):
        # `getcwd` is a path operation, so a cwd outside the granted set fails
        # EPERM, and CPython calls it on the first import after confinement
        # (`python3 -c` puts "" at the head of `sys.path`), catching only
        # `FileNotFoundError`. Without this the failure arrives as a
        # `PermissionError` from the import machinery, nowhere near a file.
        profile = self._build()
        cwd = os.getcwd()
        self.assertIn('(allow file-read-metadata (literal "%s"))' % cwd, profile)
        # Metadata, not contents: `listdir` stays denied, which the import
        # machinery handles as an empty directory and moves on.
        self.assertNotIn('(allow file-read* (subpath "%s"))' % cwd, profile)

    def test_rpc_socket_literal_is_allowed_when_known(self):
        # `subpath` filters on `network-outbound` are far less attested than
        # `literal`, and the host knows the socket path by the time the profile
        # is rendered, so the exact path is granted as well.
        profile = self._build(
            rpc_socket_dir="/private/tmp/sbx", sock="/private/tmp/sbx/rpc_1.sock"
        )
        self.assertIn(
            '(allow network-outbound (literal "/private/tmp/sbx/rpc_1.sock"))', profile
        )

    def test_paths_with_quotes_are_escaped(self):
        # SBPL is s-expression syntax: an unescaped quote in a path would end
        # the string early and could terminate the rule, so a crafted directory
        # name must not be able to edit the profile.
        profile = self._build(rw_paths=['/tmp/a"b'])
        self.assertIn(r'"/tmp/a\"b"', profile)
        self.assertNotIn('"/tmp/a"b"', profile)


class MacosProcessPrologueTest(testing.TestCase):
    """The pre-profile half of the macOS bootstrap, which is plain POSIX.

    Runs on every platform, like `SbplProfileTest`: it is the part of the
    Seatbelt path that decides *where the process stands* when the profile
    lands, and standing in the wrong place is what a profile cannot fix.
    """

    def _prepare(self, cfg):
        ns = {}
        exec(_MACOS_CONFINE_SRC, ns)
        cwd = os.getcwd()
        path = list(sys.path)
        env = os.environ.get("TMPDIR")
        tempdir = tempfile.tempdir

        def _restore():
            os.chdir(cwd)
            sys.path[:] = path
            tempfile.tempdir = tempdir
            if env is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = env

        self.addCleanup(_restore)
        ns["_prepare_macos_process"](cfg)

    def test_tempdir_points_into_the_sandbox(self):
        # The host temp dir is deliberately not granted, so a snippet reaching
        # for scratch space has to land inside the sandbox's own area.
        scratch = self.get_temp_dir()
        self._prepare({"tmpdir": scratch})
        self.assertEqual(os.environ["TMPDIR"], scratch)
        self.assertEqual(tempfile.gettempdir(), scratch)

    def test_working_directory_moves_into_the_sandbox(self):
        # `getcwd` is a path operation: a cwd outside the granted set fails
        # EPERM under the profile, and metadata on it is not enough. Being
        # somewhere the profile grants outright is.
        scratch = self.get_temp_dir()
        self._prepare({"tmpdir": scratch})
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(scratch))

    def test_empty_sys_path_entry_is_dropped(self):
        # CPython resolves "" through `getcwd` on the first import after the
        # profile applies, catching only `FileNotFoundError`, so an EPERM there
        # kills an import that a granted `sys.path` entry would have served.
        sys.path.insert(0, "")
        self._prepare({"tmpdir": self.get_temp_dir()})
        self.assertNotIn("", sys.path)
        # ...and only that entry: the runtime set has to survive intact.
        self.assertIn(os.path.dirname(os.__file__), sys.path)

    def test_prologue_without_a_scratch_dir_is_harmless(self):
        # `tmpdir` is absent for an unconfined run and on the `run_bash` path
        # before a scratch dir exists; the prologue must not throw.
        here = os.getcwd()
        self._prepare({})
        self.assertEqual(os.getcwd(), here)


@unittest.skipUnless(_SEATBELT_OK, f"seatbelt confinement unavailable: {_KIND}")
class MacosSeatbeltConfineTest(_SandboxTestCase):
    """Behavioural checks for the macOS backend, on a real Darwin host.

    These assert the boundary actually holds, rather than that the profile
    merely renders. Seatbelt gives no PID isolation and no syscall filter, so
    there is deliberately no counterpart here to the namespace suite's
    `test_confine_pid_namespace_hides_host_processes` or
    `test_confine_seccomp_blocks_denied_syscall`.
    """

    @staticmethod
    def _diag(result):
        """Failure message carrying the sandbox's own stderr.

        A profile that denies something it should not usually surfaces as a
        *downstream* symptom (a `NameError` for state that never persisted, a
        `None` result), while the bootstrap's `persist-warn` / `result-warn`
        line naming the denied path goes to stderr. These run on CI hosts
        nobody can attach to, so the message has to carry it.
        """
        return "error=%s\nstderr=%s" % (result.error, result.stderr)

    async def test_confinement_kind_is_seatbelt(self):
        self.assertEqual(confinement_kind(), "seatbelt")

    async def test_confined_run_still_works(self):
        # The boundary is worthless if it also breaks execution: the snippet
        # must still import, run and keep state across calls.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        first = await sandbox.run("import json\nkeep = json.dumps({'a': 1})")
        # Assert on the *first* run too: it is the one that has to write the
        # dill state into the sandbox's own dir, so a profile that cannot reach
        # that dir fails here rather than as a NameError one run later.
        self.assertTrue(first.ok, msg=self._diag(first))
        result = await sandbox.run("print(keep)")
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertIn('{"a": 1}', result.stdout)

    async def test_confined_run_captures_result(self):
        # The last-expression value travels back through a file in the sandbox
        # dir, the same writable surface the state file uses, so it is a second
        # independent check that writes there actually land.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        result = await sandbox.run("6 * 7")
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertEqual(result.result, 42, msg=self._diag(result))

    async def test_confine_blocks_write_outside_sandbox(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        result = await sandbox.run(
            "import os\n"
            "target = os.path.expanduser('~/synalinks_seatbelt_escape.txt')\n"
            "try:\n"
            "    open(target, 'w').write('escaped')\n"
            "    outcome = 'WROTE'\n"
            "except Exception as exc:\n"
            "    outcome = type(exc).__name__\n"
            "print(outcome)\n"
        )
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertNotIn("WROTE", result.stdout)
        self.assertIn("Error", result.stdout)

    async def test_confine_blocks_writes_to_the_seed_workdir(self):
        # `workdir` is a seed, copied into the virtual filesystem host-side, and
        # the documented contract is that the agent's writes never reach the
        # real directory. Linux gets that from the pivot: the host filesystem is
        # gone. macOS is the one platform where the directory is still *there*,
        # so it is the one platform where this has to be asserted.
        import shutil

        workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workdir, True)
        host_file = os.path.join(workdir, "main.py")
        with open(host_file, "w") as fh:
            fh.write("print('orig')\n")

        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True, workdir=workdir)
        result = await sandbox.run(
            "try:\n"
            "    open(%r, 'w').write('escaped')\n"
            "    outcome = 'WROTE'\n"
            "except Exception as exc:\n"
            "    outcome = type(exc).__name__\n"
            "print(outcome)\n" % host_file
        )
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertNotIn("WROTE", result.stdout)
        with open(host_file) as fh:
            self.assertEqual(fh.read(), "print('orig')\n")

    async def test_confine_blocks_read_outside_runtime_set(self):
        # Reads are restricted, unlike the Codex/Gemini Seatbelt profiles which
        # allow the whole disk. /etc/passwd is readable on any unconfined mac.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        result = await sandbox.run(
            "try:\n"
            "    open('/etc/passwd').read()\n"
            "    outcome = 'READ'\n"
            "except Exception as exc:\n"
            "    outcome = type(exc).__name__\n"
            "print(outcome)\n"
        )
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertNotIn("READ", result.stdout)

    async def test_subprocesses_run_but_inherit_the_profile(self):
        # Shelling out to a host tool works here as it does on Linux, and that
        # is safe for one reason only: the child inherits the profile. Assert
        # both halves together, because the first without the second is exactly
        # the hole that would make exec worth denying.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        result = await sandbox.run(
            "import os, subprocess\n"
            "ran = subprocess.run(\n"
            "    ['/bin/sh', '-c', 'echo alive'], capture_output=True, text=True\n"
            ")\n"
            "print(ran.stdout.strip())\n"
            "escape = os.path.expanduser('~/synalinks_subprocess_escape.txt')\n"
            "out = subprocess.run(\n"
            "    ['/bin/sh', '-c', 'echo escaped > %s' % escape],\n"
            "    capture_output=True, text=True,\n"
            ")\n"
            "print('ESCAPED' if os.path.exists(escape) else 'CONFINED')\n"
        )
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertIn("alive", result.stdout)
        self.assertIn("CONFINED", result.stdout)
        self.assertNotIn("ESCAPED", result.stdout)

    async def test_confine_cuts_network(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        result = await sandbox.run(
            "import socket\n"
            "s = socket.socket()\n"
            "s.settimeout(5)\n"
            "try:\n"
            "    s.connect(('1.1.1.1', 80))\n"
            "    outcome = 'CONNECTED'\n"
            "except Exception as exc:\n"
            "    outcome = type(exc).__name__\n"
            "print(outcome)\n"
        )
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertNotIn("CONNECTED", result.stdout)

    async def test_confine_writes_inside_sandbox_still_work(self):
        # The complement of the escape test: the sandbox's own writable area
        # must stay writable, or the boundary is just breakage. `TMPDIR` is
        # pointed into that area precisely so the most ordinary thing a snippet
        # does with a scratch file works; the host temp dir it would otherwise
        # name is *not* granted, and must not be.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        result = await sandbox.run(
            "import os, tempfile\n"
            "p = os.path.join(tempfile.gettempdir(), 'inside.txt')\n"
            "open(p, 'w').write('ok')\n"
            "print(open(p).read())\n"
            "print('WROTE_INSIDE')\n"
        )
        self.assertTrue(result.ok, msg=self._diag(result))
        self.assertIn("WROTE_INSIDE", result.stdout)
        self.assertIn("ok", result.stdout)
