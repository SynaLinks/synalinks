# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable


@synalinks_export(
    [
        "synalinks.sandboxes.ExecutionResult",
        "synalinks.ExecutionResult",
    ]
)
class ExecutionResult(DataModel):
    """Structured result of running a snippet in a ``Sandbox``."""

    stdout: str = Field(
        default="",
        description="Concatenated bytes written to stdout by the snippet.",
    )
    stderr: str = Field(
        default="",
        description="Concatenated bytes written to stderr by the snippet.",
    )
    result: Optional[Any] = Field(
        default=None,
        description="Value of the snippet's last expression, or null.",
    )
    error: Optional[str] = Field(
        default=None,
        description=(
            "Formatted error message when execution failed, else null. "
            "A non-null `error` means stdout / stderr / result captured "
            "whatever happened before the failure."
        ),
    )

    @property
    def ok(self) -> bool:
        """True when the snippet ran to completion without an error."""
        return self.error is None


@synalinks_export(
    [
        "synalinks.sandboxes.Sandbox",
        "synalinks.Sandbox",
    ]
)
class Sandbox(SynalinksSaveable):
    """Abstract base class for code execution sandboxes.

    A sandbox is a **stateful**, **restricted** Python environment:
    subsequent ``run`` calls see variables, imports and function
    definitions from previous runs. State can be captured via ``dump()``
    as an opaque byte string and restored via ``load()`` — sandboxes also
    round-trip through ``get_config()`` / ``from_config()`` so they can
    flow through Synalinks' normal serialization pipeline.

    Ownership is the **caller's** responsibility: construct a sandbox,
    hand it to a code-executing module (e.g. ``CodeModeAgent``) across
    successive interactive turns, and build a new one for a fresh
    conversation. The consuming module stays stateless.

    Subclasses implement the backend — Monty REPL, Pyodide, Docker,
    subprocess — and must override ``run``, ``reset``, ``dump``,
    ``load``, plus the ``SynalinksSaveable`` hooks (``_obj_type``,
    ``get_config``, ``from_config``).

    Args:
        timeout (float): Per-snippet execution budget in seconds
            (Default 5). Backends that cannot enforce this should treat
            it as advisory; modules that instantiate sandboxes (e.g.
            ``CodeModeAgent``) pass this through.
        name (str): Optional. Human-readable name for the sandbox.
    """

    # A natural-language description of the sandbox's constraints,
    # intended for inclusion in LM prompts. Consumers (e.g.
    # ``CodeModeAgent``, ``PythonSynthesis``) compose this text into
    # their instructions or schema descriptions so the language model /
    # optimizer knows which stdlib subset, builtins, and language
    # features are available. Subclasses override with a concise,
    # prompt-friendly description of what code they can run.
    description: str = ""

    def __init__(
        self,
        timeout: float = 5.0,
        name: Optional[str] = None,
    ):
        self.timeout = float(timeout)
        self.name = name

    async def run(
        self,
        code: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ) -> ExecutionResult:
        """Execute ``code`` and return a structured result.

        Args:
            code (str): The Python source to run.
            inputs (dict): Optional. Variables bound into the sandbox
                namespace before execution. Backend-specific rules apply
                (some backends may only honour this on the first call).
            external_functions (dict): Optional. Mapping of name → async
                callable exposed as global functions inside the sandbox
                for the duration of the call.

        Returns:
            ExecutionResult: stdout, stderr, last-expression result and
            (if any) an error string.
        """
        raise NotImplementedError("Sandbox subclasses must implement `run`.")

    def reset(self) -> None:
        """Wipe all state and start over with an empty sandbox."""
        raise NotImplementedError("Sandbox subclasses must implement `reset`.")

    def dump(self) -> bytes:
        """Serialize the current sandbox state to bytes.

        The blob returned must be self-contained enough that a subsequent
        ``load(blob)`` reconstructs an equivalent sandbox, including its
        namespace (variables, imports, user-defined functions).
        """
        raise NotImplementedError("Sandbox subclasses must implement `dump`.")

    @classmethod
    def load(cls, data: bytes, **kwargs) -> "Sandbox":
        """Restore a sandbox from bytes produced by ``dump()``."""
        raise NotImplementedError("Sandbox subclasses must implement `load`.")

    def _obj_type(self):
        return "Sandbox"
