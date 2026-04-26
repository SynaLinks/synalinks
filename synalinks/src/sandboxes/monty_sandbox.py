# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import base64
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import pydantic_monty

from synalinks.src.api_export import synalinks_export
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import Sandbox
from synalinks.src.saving.object_registration import register_synalinks_serializable


def _split_streams(collector: "pydantic_monty.CollectStreams"):
    stdout_parts = []
    stderr_parts = []
    for stream, text in collector.output:
        if stream == "stderr":
            stderr_parts.append(text)
        else:
            stdout_parts.append(text)
    return "".join(stdout_parts), "".join(stderr_parts)


@register_synalinks_serializable()
@synalinks_export(
    [
        "synalinks.sandboxes.MontySandbox",
        "synalinks.MontySandbox",
    ]
)
class MontySandbox(Sandbox):
    """A restricted Python sandbox backed by ``pydantic_monty``.

    Wraps a `Monty <https://github.com/pydantic/monty>`_ REPL: a
    restricted Python interpreter with a small stdlib subset, no
    filesystem / environment / network access, and no ``class`` or
    ``match`` statements. Safe for executing LM-authored Python snippets.

    State (variables, imports, user-defined functions) accumulates across
    ``run`` calls. ``dump()`` serializes the full REPL namespace to bytes;
    ``load()`` restores it. Full state round-trips through
    ``get_config()`` / ``from_config()`` as well, so a sandbox can be
    persisted alongside a conversation trajectory.

    Example:

    ```python
    import synalinks

    sandbox = synalinks.MontySandbox(timeout=5)
    await sandbox.run("x = 42")
    result = await sandbox.run("print(x)")
    print(result.stdout)                    # -> "42\\n"

    # Snapshot + restore
    blob = sandbox.dump()
    restored = synalinks.MontySandbox.load(blob)
    ```

    Args:
        timeout (float): Per-snippet execution budget in seconds (Default 5).
            Each ``run`` call is guaranteed a fresh ``timeout`` of actual
            in-sandbox execution time; the sandbox internally resets
            Monty's cumulative clock between snippets via dump/load, so
            long idle gaps and prior snippets do not eat into the budget
            of the current one.
        name (str): Optional. Human-readable name for the sandbox.
    """

    description: str = (
        "Code runs inside a Monty sandbox: a restricted Python "
        "interpreter. Only this stdlib subset is importable: sys, os, "
        "typing, asyncio, re, datetime, json, math, pathlib. No "
        "third-party libraries, no `class` or `match` statements. The "
        "filesystem, environment and network are unreachable: `open()`, "
        "`os.system`, `os.listdir`, `os.environ`, `os.path`, `sys.argv` "
        "and `Path.read_text` are not available even though `os`, `sys` "
        "and `pathlib` import. `asyncio` is also a stub: only "
        "`asyncio.run` and `asyncio.gather` exist (no `asyncio.sleep`, "
        "`wait_for`, `Future`, `create_task` or `TaskGroup`), and there "
        "are no time primitives. Any tools bound to the module are "
        "exposed as global async callables; call them inside an "
        "`async def main(): ...` using `await tool_name(...)` and run "
        "the coroutine with `asyncio.run(main())`. Every tool call "
        "returns a **dict**: a tool wrapping `async def f(x) -> int` "
        "yields `{'result': <value>}`, a tool that already returns a "
        "dict yields that dict directly — index the field you need "
        "before using it."
    )

    def __init__(
        self,
        timeout: float = 5.0,
        name: Optional[str] = None,
    ):
        super().__init__(timeout=timeout, name=name)
        self._repl = self._new_repl()

    def _new_repl(self) -> pydantic_monty.MontyRepl:
        return pydantic_monty.MontyRepl(
            limits=pydantic_monty.ResourceLimits(
                max_duration_secs=self.timeout,
            )
        )

    async def run(
        self,
        code: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ) -> ExecutionResult:
        # Monty's `max_duration_secs` is a **cumulative** budget across all
        # `feed_run_async` calls on a single REPL — once exceeded, every
        # subsequent call fails, even trivial ones. Users expect ``timeout``
        # to be per-snippet, so before each run we refresh the internal REPL
        # via a dump/load round-trip. The namespace (variables, imports,
        # function definitions) is preserved; only the elapsed-time counter
        # is reset. Dump+load is sub-millisecond even with meaningful state.
        blob = self._repl.dump()
        self._repl = pydantic_monty.MontyRepl.load(blob)

        collector = pydantic_monty.CollectStreams()
        kwargs: Dict[str, Any] = {"print_callback": collector}
        if inputs is not None:
            kwargs["inputs"] = inputs
        if external_functions is not None:
            kwargs["external_functions"] = external_functions

        result = None
        error = None
        try:
            result = await self._repl.feed_run_async(code, **kwargs)
        except pydantic_monty.MontyError as e:
            error = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 — sandboxed code can raise anything
            error = f"{type(e).__name__}: {e}"

        stdout, stderr = _split_streams(collector)
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            result=result,
            error=error,
        )

    def reset(self) -> None:
        self._repl = self._new_repl()

    def dump(self) -> bytes:
        return self._repl.dump()

    @classmethod
    def load(
        cls,
        data: bytes,
        *,
        timeout: float = 5.0,
        name: Optional[str] = None,
    ) -> "MontySandbox":
        """Restore a sandbox from bytes produced by ``dump()``.

        Monty does not persist the original resource limits, so the
        caller re-supplies ``timeout`` for the restored sandbox.
        """
        instance = cls.__new__(cls)
        Sandbox.__init__(instance, timeout=timeout, name=name)
        instance._repl = pydantic_monty.MontyRepl.load(data)
        return instance

    def _obj_type(self):
        return "MontySandbox"

    def get_config(self):
        """Serialize the sandbox (config + full REPL state) to a dict.

        The REPL state is captured via ``dump()`` and base64-encoded so
        the resulting config is JSON-safe.
        """
        return {
            "timeout": self.timeout,
            "name": self.name,
            "state": base64.b64encode(self.dump()).decode("ascii"),
        }

    @classmethod
    def from_config(cls, config):
        state_b64 = config.pop("state", None)
        timeout = config.pop("timeout", 5.0)
        name = config.pop("name", None)
        if state_b64:
            return cls.load(
                base64.b64decode(state_b64),
                timeout=timeout,
                name=name,
            )
        return cls(timeout=timeout, name=name)
