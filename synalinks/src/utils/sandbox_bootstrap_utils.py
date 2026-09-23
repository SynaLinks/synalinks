# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""The program each ``run_code`` executes inside Mirage's ``python3``."""

from synalinks.src.utils.confinement_utils import CONFINE_PROLOGUE_SRC

# The launcher is what ``python3 -c`` actually runs: it decodes and execs the
# real bootstrap (argv[1]) so the bootstrap source never has to survive shell
# quoting. argv[2] is the dill session-state path; argv[3] is the path of the
# per-run JSON config file (per-call ``inputs`` blob, host-tool RPC socket +
# tool names, and the path to write the result value to). Data always travels
# by file, never on argv: the kernel caps a single exec argument at
# MAX_ARG_STRLEN (128KiB), far below real ``inputs`` payloads, and a path
# needs no encoding to survive quoting. Every arg is non-empty on purpose:
# Mirage's shell drops empty ``''`` tokens, which would shift ``argv``.
LAUNCHER = "import base64,sys;exec(base64.b64decode(sys.argv[1]))"

# The bootstrap runs inside Mirage's real CPython subprocess. Mirage spawns a
# fresh ``python3`` per command, so to make variables/imports/functions persist
# across ``run_code`` calls we serialize the user namespace with ``dill`` after each
# snippet and restore it before the next: true REPL state without replaying
# earlier snippets (and their side effects). The namespace lives in a dedicated
# dict (``ns``) the snippet runs in, pickled with explicit file I/O:
# ``dill.dump_module`` is deliberately avoided because it embeds its origin path
# in the pickle, so a re-dump of a *copied* state file (a fork) would write back
# to the original, breaking isolation. Unpicklable values are skipped per-key.
#
# Host callables bound to the sandbox can't be injected into this isolated
# subprocess directly, so each is exposed as an async stub that RPCs to a host
# Unix-socket server (length-prefixed JSON, one request/response per call); the
# host runs the real (possibly LM-calling) tool and returns the result. The
# snippet is exec'd under the filename ``<sandbox>`` so tracebacks trim to user
# frames; its last *expression* (the ``result`` convention) is JSON-encoded
# to the result file so the host can surface it as ``ExecutionResult.result``.

BOOTSTRAP = (
    r"""
import os, sys, base64, json, ast
import dill
state = sys.argv[2]
config = {}
if len(sys.argv) > 3 and sys.argv[3]:
    try:
        with open(sys.argv[3], "r") as _fh:
            config = json.load(_fh)
    except Exception as exc:
        print("config-warn: " + repr(exc), file=sys.stderr)
"""
    + CONFINE_PROLOGUE_SRC
    + r"""
if config.get("confine"):
    try:
        _confine_platform(config)
    except Exception as exc:
        print("confine-error: " + repr(exc), file=sys.stderr)
        sys.exit(99)
ns = {"__name__": "__main__", "__builtins__": __builtins__}
if os.path.exists(state):
    try:
        with open(state, "rb") as fh:
            ns.update(dill.load(fh))
    except Exception as exc:
        print("restore-warn: " + repr(exc), file=sys.stderr)
# Pinned names re-assert themselves on every run, from the persisted
# ``__rlm_pinned__`` dict, BEFORE the per-run ``inputs=`` binding (so an
# explicit binding still wins). A caller that persists an environment
# variable for the agent's code to read (the RLM binds the module input as
# ``inputs`` once per call) pins it so that a snippet assigning over the
# name — LLM-written code does — only breaks that one snippet, instead of
# poisoning every later snippet of the call: the next run restores the pin.
try:
    ns.update(ns.get("__rlm_pinned__") or {})
except Exception as exc:
    print("pinned-warn: " + repr(exc), file=sys.stderr)
_inputs = config.get("inputs")
if _inputs:
    try:
        ns.update(dill.loads(base64.b64decode(_inputs)))
    except Exception as exc:
        print("inputs-warn: " + repr(exc), file=sys.stderr)
# Re-home restored functions onto the live ``ns``. dill pickles a function
# together with a *private copy* of the globals it referenced, so a restored
# function's ``__globals__`` is a ghost of the namespace as of the run that
# defined it — while this run's snippet execs into ``ns``. Without re-homing,
# a function defined in an earlier run can never see a name defined in a
# later one (``def f(): return helper()`` then ``def helper(): ...`` next run
# leaves ``f`` raising NameError forever), and every dump/restore round-trip
# nests another ghost copy into the state file. Rebuilding each function on
# ``ns`` restores true REPL semantics: one shared global namespace.
#
# Only functions *defined in the sandbox* may be re-homed. An imported
# function closes over its own module's globals, and rebuilding it on ``ns``
# strips the module internals it reaches at call time: re-homing
# ``collections.Counter``'s methods makes ``Counter('aa')`` raise
# ``NameError: name '_collections_abc' is not defined``, and re-homing
# ``os.path.join`` loses ``sep``. Sandbox-defined functions are the ones whose
# (ghost) globals are a copy of ``ns``, which carries ``__name__ ==
# "__main__"``; an imported one carries its own module name.
import types as _types
def _rehome(value):
    if (
        isinstance(value, _types.FunctionType)
        and value.__globals__ is not ns
        and value.__globals__.get("__name__") == "__main__"
    ):
        fixed = _types.FunctionType(
            value.__code__, ns, value.__name__, value.__defaults__, value.__closure__
        )
        fixed.__kwdefaults__ = value.__kwdefaults__
        fixed.__qualname__ = value.__qualname__
        fixed.__dict__.update(value.__dict__)
        fixed.__module__ = value.__module__
        return fixed
    return value
for _key in list(ns):
    _value = ns[_key] = _rehome(ns[_key])
    if isinstance(_value, type):
        # Methods of sandbox-defined classes carry the same ghost globals.
        for _attr, _member in list(vars(_value).items()):
            _fixed = _rehome(_member)
            if _fixed is not _member:
                try:
                    setattr(_value, _attr, _fixed)
                except (AttributeError, TypeError):
                    pass
_sock = config.get("sock")
if _sock and config.get("sock_dir"):
    # Inside the microVM a host socket is reached through a guest-local proxy
    # of the same name (a unix socket does not cross the shared filesystem).
    _sock = os.path.join(config["sock_dir"], os.path.basename(_sock))
if _sock and config.get("tools"):
    import asyncio, struct, threading
    async def _rpc(_name, *args, **kwargs):
        reader, writer = await asyncio.open_unix_connection(_sock)
        payload = json.dumps(
            {"name": _name, "args": args, "kwargs": kwargs}
        ).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()
        header = await reader.readexactly(4)
        (length,) = struct.unpack(">I", header)
        body = await reader.readexactly(length)
        writer.close()
        resp = json.loads(body.decode("utf-8"))
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "tool call failed"))
        return resp.get("result")
    # Tools are exposed as PLAIN SYNC functions: `result = submit(...)`, with no
    # `await` / `asyncio.run(...)`. Each call's RPC coroutine runs on a dedicated
    # background event-loop thread and the result is returned synchronously, so
    # this works from a flat script and even from inside a snippet that happens
    # to run its own event loop.
    _tool_loop = asyncio.new_event_loop()
    threading.Thread(target=_tool_loop.run_forever, daemon=True).start()
    def _make_stub(_name):
        def _stub(*args, **kwargs):
            return asyncio.run_coroutine_threadsafe(
                _rpc(_name, *args, **kwargs), _tool_loop
            ).result()
        return _stub
    for _tool_name in config["tools"]:
        ns[_tool_name] = _make_stub(_tool_name)
user = sys.stdin.buffer.read().decode("utf-8")
value = None
error = None
try:
    tree = ast.parse(user, "<sandbox>", "exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        if tree.body:
            exec(compile(tree, "<sandbox>", "exec"), ns)
        value = eval(compile(ast.Expression(last.value), "<sandbox>", "eval"), ns)
    else:
        exec(compile(tree, "<sandbox>", "exec"), ns)
except SystemExit:
    raise
except BaseException as exc:
    # Reported as a structured error (E2B's name / value / traceback) rather
    # than printed, so stderr keeps only what the snippet itself wrote. The
    # traceback starts at the snippet's own frames, not this bootstrap's.
    import traceback as _traceback
    _tb = exc.__traceback__
    while _tb is not None and _tb.tb_frame.f_code.co_filename != "<sandbox>":
        _tb = _tb.tb_next
    error = {
        "name": type(exc).__name__,
        "value": str(exc),
        "traceback": "".join(_traceback.format_exception(type(exc), exc, _tb)),
    }
finally:
    # Persist the namespace with ONE dill.dumps. Pickling item by item is
    # quadratic: every sandbox-defined function is pickled together with a
    # copy of its (shared) globals, so N functions cost N full-namespace
    # pickles (25 functions ~3 s, 100 ~14 s, ~200 hits RecursionError). A
    # single dump memoizes the shared globals once. Only when that fails do
    # we fall back to filtering out the unpicklable items one by one.
    keep = {k: v for k, v in ns.items() if k != "__builtins__"}
    try:
        blob = dill.dumps(keep)
    except Exception:
        # Something in the namespace is unpicklable (an open file, a
        # generator, ...). Drop those items, and pickle functions with
        # ``recurse=True`` so they carry only the globals they reference
        # rather than the whole namespace: otherwise every sandbox-defined
        # function would be lost along with the offending item. Restored
        # functions are re-homed onto the live namespace on the next run
        # anyway, so the reduced globals are never observable.
        keep = {}
        for key, item in list(ns.items()):
            if key == "__builtins__":
                continue
            try:
                dill.dumps(item, recurse=True)
                keep[key] = item
            except Exception:
                pass
        try:
            blob = dill.dumps(keep, recurse=True)
        except Exception as exc:
            blob = None
            print("persist-warn: " + repr(exc), file=sys.stderr)
    if blob is not None:
        # Write to a sibling temp file and rename so a run killed mid-write
        # (host timeout) or a concurrent run can never leave the state file
        # truncated or half-written.
        tmp = state + ".tmp." + str(os.getpid())
        try:
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, state)
        except Exception as exc:
            print("persist-warn: " + repr(exc), file=sys.stderr)
            try:
                os.unlink(tmp)
            except OSError:
                pass
    result_path = config.get("result")
    if result_path:
        # The last expression's value as E2B reports a main result: its repr
        # as ``text`` and, when JSON-serializable, the value as ``json``.
        report = {"error": error, "text": None, "json": None}
        if value is not None:
            try:
                report["text"] = repr(value)
            except Exception:
                report["text"] = object.__repr__(value)
            try:
                json.dumps(value)
                report["json"] = value
            except Exception:
                pass
        try:
            with open(result_path, "w") as fh:
                json.dump(report, fh)
        except Exception as exc:
            print("result-warn: " + repr(exc), file=sys.stderr)
if error is not None:
    sys.exit(1)
"""
)
