# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_strictly_chat_messages
from synalinks.src.modules.agents.agent_utils import InputsSummary
from synalinks.src.modules.agents.agent_utils import summarize_inputs
from synalinks.src.modules.agents.agent_utils import unique_inputs_path
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.sandboxes.monty_sandbox import MontySandbox
from synalinks.src.saving import serialization_lib


def get_default_instructions(workdir: Optional[str]) -> str:
    """Default system instructions for the deep agent.

    Args:
        workdir: Absolute path of the agent's working directory, or
            ``None`` for an empty in-memory workspace. Embedded in the
            prompt so the LM knows where it's operating.

    Returns:
        A prompt string describing the tool plan.
    """
    if workdir:
        workdir_line = f"Workdir: {workdir}"
    else:
        workdir_line = (
            "Workdir: (none) — an empty in-memory workspace; "
            "create files with `write_file`."
        )
    return f"""
You are a software engineering assistant working inside a sandboxed,
copy-on-write filesystem.

{workdir_line}
Available tools: read_file, list_files, search_files, write_file,
edit_file, run_python_code, run_python_file

Plan:
1. Use `list_files` to discover files (glob, e.g. `**/*.py`).
2. Use `search_files` to grep file contents by regex across a glob.
3. Use `read_file` to read a file; it returns the requested lines with
   1-based `start_line` / `end_line`. Page through long files with
   `offset` / `limit` (raise `offset` to read further in).
4. Use `edit_file` for surgical changes (preferred over `write_file`).
5. Use `run_python_code` to run a Python snippet directly, or
   `write_file` a self-contained script into the overlay then
   `run_python_file(path)` to execute it (a script cannot import other
   overlay files).
6. Once you have the answer, stop calling tools and respond.

Notes:
- The filesystem is copy-on-write: edits and new files land in an
  in-memory overlay and never modify the real workspace on disk.
- Paths are rooted at the workdir; `..` cannot escape it.
- If the conversation includes an `InputsSummary`, you only see field
  previews and sizes — the full, untruncated input values are saved as
  the JSON file named in its `inputs_file` field. Read that file instead
  of retyping values from the preview: either call the `read_file` tool,
  or inside a `run_python_code` snippet parse it with
  `json.loads(pathlib.Path(inputs_file).read_text())`. The sandbox has no
  `open()`, and `json` provides only `loads` / `dumps` (no `json.load`),
  so `json.load(open(...))` will not work.""".strip()


def get_subagent_tools_guidance() -> str:
    """Guidance appended to the instructions when subagents are enabled."""
    return """
You can delegate work to parallel subagents, each on its own isolated
branch (a copy-on-write fork) of the filesystem:
- `spawn_subagents(tasks)`: launch one subagent per task string. Each
  subagent sees the files you see *now* and may freely read/write/edit/
  delete them, but its changes stay on its own branch — they never touch
  your filesystem. Subagents run concurrently, so use this to parallelize
  independent exploration or edits. Returns a `handle` and a `diff`
  (its pending changes) per subagent.
- `merge_subagent(handle, paths=None, force=False)`: after reviewing a
  subagent's `diff`, fold its changes into your filesystem (pass `paths` to
  take only a subset). A path you also changed since spawning is a conflict
  and is refused (reported in `conflicts` / `skipped`); pass `force=True`
  to apply it anyway (the subagent's version wins).
- `discard_subagent(handle)`: drop a subagent's branch unmerged.
Nothing a subagent does affects your files until you `merge_subagent` it.
""".strip()


def get_subagent_instructions() -> str:
    """System instructions for a spawned subagent (depth >= 1)."""
    return """
You are a subagent working on a private, isolated branch of a sandboxed,
copy-on-write filesystem. The files you see were inherited from the parent
agent at the moment you were spawned; your edits stay on your branch and
affect no one else. Available tools: read_file, list_files, search_files,
write_file, edit_file, run_python_code, run_python_file.

Plan:
1. Explore with `list_files` / `search_files` / `read_file`.
2. Make the changes your task requires with `edit_file` / `write_file`,
   or run code with `run_python_code` / `run_python_file`.
3. Stop and report concisely what you did and what you changed — the parent
   agent reviews your branch and decides whether to keep it.

Notes:
- The filesystem is copy-on-write: edits land in an in-memory overlay.
- Paths are rooted at the workspace; `..` cannot escape it.
""".strip()


def _final_answer_text(output) -> str:
    """Extract a subagent's final answer text from its (no-schema) output.

    A no-schema agent returns a ``ChatMessages`` data model; its last
    message's ``content`` is the final answer. Falls back to a JSON dump
    for any other shape so the parent always gets a string.
    """
    if output is None:
        return ""
    data = output.get_json() if hasattr(output, "get_json") else output
    if isinstance(data, dict):
        messages = data.get("messages")
        if messages:
            content = (
                messages[-1].get("content") if isinstance(messages[-1], dict) else None
            )
            if isinstance(content, str):
                return content
            return "" if content is None else json.dumps(content, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)
    return str(data)


@synalinks_export(
    [
        "synalinks.modules.DeepAgent",
        "synalinks.DeepAgent",
    ]
)
class DeepAgent(Module):
    """A coding agent whose tools are a sandboxed copy of a workdir.

    DeepAgent is a thin specialization of :class:`FunctionCallingAgent`
    that mounts the workdir in a :class:`MontySandbox` and exposes the
    sandbox's tool methods to the LM:

    - ``read_file``: read a file by 1-based line range (paginated).
    - ``list_files``: list files matching a glob.
    - ``search_files``: glob for files and grep their contents (regex).
    - ``write_file``: create/overwrite a file.
    - ``edit_file``: exact-string replacement.
    - ``run_python_code``: run a Python snippet directly in the sandbox.
    - ``run_python_file``: run a self-contained script the agent wrote
      into the overlay.

    Every tool is backed by the sandbox's copy-on-write overlay and the
    Monty interpreter, so the agent is **host-safe by construction**:
    reads fall through to the real ``workdir`` but writes, edits and code
    execution can never modify it or reach the host — so there is nothing
    to gate, and all tools are always available. Inspect what the agent
    did through ``agent.sandbox`` — ``changes()``, ``journal()``,
    ``read_overlay()`` — and persist any of it yourself if desired.

    The constructor mirrors :class:`FunctionCallingAgent` — every
    parameter on that class is accepted here with identical semantics.
    The additions are ``workdir`` (required) and the sandbox ``timeout``.
    User-supplied ``tools`` are appended to the built-in ones.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():
        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await synalinks.DeepAgent(
            workdir="/tmp/my_project",
            language_model=lm,
        )(inputs)
        agent = synalinks.Program(inputs=inputs, outputs=outputs)

        messages = synalinks.ChatMessages(messages=[
            synalinks.ChatMessage(
                role="user",
                content="What's in this directory?",
            )
        ])
        result = await agent(messages)
        print(result.get("messages")[-1].get("content"))

    asyncio.run(main())
    ```

    Args:
        workdir (str): Optional working directory the agent operates on.
            When given it must exist and is mounted read-through in the
            sandbox (the LM's writes/edits stay in the overlay and never
            touch it). When omitted, the sandbox starts as an empty
            in-memory filesystem.
        timeout (float): Per-snippet execution budget in seconds for
            ``run_python_code`` / ``run_python_file``. Defaults to 30.
        tools (list): Additional :class:`Tool` instances (or plain async
            functions) to expose alongside the built-in tools. Names must
            not start with ``_`` or collide with built-ins.
        sandbox (Sandbox): Optional ready-made sandbox to operate on instead
            of building one from ``workdir`` — e.g. a :meth:`Sandbox.fork`
            of another agent's filesystem. When given, ``workdir`` is used
            only for the default instructions text.
        max_subagent_depth (int): When ``> 0``, the agent gains
            ``spawn_subagents`` / ``merge_subagent`` / ``discard_subagent``
            tools, letting the LM run subagents in parallel — each on an
            isolated :meth:`Sandbox.fork` of the filesystem whose changes
            only land on an explicit ``merge_subagent``. The value caps
            nesting: ``1`` (the recommended setting) lets this agent spawn
            subagents that cannot themselves spawn; ``2`` allows one more
            level, and so on. Defaults to ``0`` (subagents disabled —
            backward-compatible). Requires a fork-capable sandbox
            (``MontySandbox`` is).

            Subagent forks here are **filesystem branches** (each gets a
            fresh interpreter), so across parallel subagents you can fold
            back **all** their file changes. Folding back Python REPL state
            (variables/functions/imports) across subagents is a
            :class:`RecursiveLanguageModelAgent` feature — and limited to one
            subagent there, because Monty serializes the REPL namespace only
            as a whole (it can't union parallel namespaces). That is a
            backend constraint, not a design shortcut.
        schema (dict): JSON schema for the final answer.
        data_model (DataModel): DataModel for the final answer.
            Mutually exclusive with ``schema``.
        language_model (LanguageModel): The language model that drives
            the agent loop.
        prompt_template (str): Forwarded to the tool-call generator.
        examples (list): Few-shot examples for the tool-call generator.
        instructions (str): Override the default system instructions.
            When omitted, the default is built from the workdir.
        final_instructions (str): Instructions for the final-answer
            generator. Defaults to ``instructions``.
        temperature (float): LM sampling temperature. Defaults to 0.0.
        use_inputs_schema (bool): Include the input schema in the prompt.
        use_outputs_schema (bool): Include the output schema in the prompt.
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When ``True``, the tool-call
            generator emits a ``thinking`` field per round.
        autonomous (bool): When ``True`` (default), the agent runs the
            tool loop end-to-end. When ``False``, returns one step at a
            time for human-in-the-loop workflows.
        return_inputs_with_trajectory (bool): When ``True`` (default),
            the full message trajectory is included alongside the final
            answer.
        max_iterations (int): Maximum number of tool-call rounds.
            Defaults to 10 (coding tasks tend to need more rounds than
            RAG / SQL).
        streaming (bool): Stream the final answer when no ``schema`` is
            set. Defaults to ``False``.
        name (str): Module name.
        description (str): Module description.
    """

    def __init__(
        self,
        *,
        workdir: Optional[str] = None,
        timeout: float = 30.0,
        tools: Optional[List] = None,
        sandbox=None,
        max_subagent_depth: int = 0,
        _subagent_depth: int = 0,
        schema=None,
        data_model=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions: Optional[str] = None,
        final_instructions: Optional[str] = None,
        temperature: float = 0.0,
        use_inputs_schema: bool = False,
        use_outputs_schema: bool = False,
        reasoning_effort: Optional[str] = None,
        use_chain_of_thought: bool = False,
        autonomous: bool = True,
        return_inputs_with_trajectory: bool = True,
        max_iterations: int = 10,
        streaming: bool = False,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(name=name, description=description)

        # `workdir` is optional: when omitted the sandbox is a pure
        # in-memory filesystem. A provided path must exist and be a dir.
        if workdir:
            resolved_workdir = Path(workdir).resolve()
            if not resolved_workdir.exists():
                raise ValueError(f"workdir does not exist: {workdir}")
            if not resolved_workdir.is_dir():
                raise ValueError(f"workdir is not a directory: {workdir}")
            self.workdir = str(resolved_workdir)
        else:
            self.workdir = None

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"`timeout` must be a positive number, got {timeout!r}")
        self.timeout = float(timeout)

        if not isinstance(max_subagent_depth, int) or max_subagent_depth < 0:
            raise ValueError(
                "`max_subagent_depth` must be a non-negative int, got "
                f"{max_subagent_depth!r}"
            )
        self.max_subagent_depth = max_subagent_depth
        self._subagent_depth = _subagent_depth
        # Subagent delegation is offered only while we may still go one level
        # deeper, so the deepest subagents can't fan out endlessly.
        self._subagents_enabled = self._subagent_depth < self.max_subagent_depth

        self.language_model = _get_lm(language_model)

        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema

        if instructions is None:
            instructions = get_default_instructions(self.workdir)
        if self._subagents_enabled:
            instructions = instructions + "\n\n" + get_subagent_tools_guidance()
        self.instructions = instructions
        self.final_instructions = final_instructions

        self.prompt_template = prompt_template
        self.examples = examples
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        self.autonomous = autonomous
        self.return_inputs_with_trajectory = return_inputs_with_trajectory
        self.max_iterations = max_iterations
        self.streaming = streaming

        # The sandbox IS the filesystem the tools operate on: reads fall
        # through to the workdir, writes/edits/code stay host-safe in the
        # overlay. Inspect via `self.sandbox.changes()` / `.journal()`. A
        # caller (or this agent, when spawning a subagent) may supply a
        # ready-made sandbox — e.g. a `fork()` of a parent's filesystem.
        self.sandbox = (
            sandbox
            if sandbox is not None
            else MontySandbox(workdir=self.workdir, timeout=self.timeout)
        )
        # Subagent branches (forks) awaiting the parent's review, keyed by
        # handle; reset per `call()`, populated by `spawn_subagents`.
        self._subagents: Dict[str, object] = {}
        self._subagent_counter = 0
        # Overlay path the full (data) inputs are written to, resolved once on
        # first use so it can't shadow a workdir file (see `_materialize_inputs`).
        self._inputs_path: Optional[str] = None
        builtin_fns = [
            self.sandbox.read_file,
            self.sandbox.list_files,
            self.sandbox.search_files,
            self.sandbox.write_file,
            self.sandbox.edit_file,
            self.sandbox.run_python_code,
            self.sandbox.run_python_file,
        ]
        if self._subagents_enabled:
            builtin_fns += [
                self.spawn_subagents,
                self.merge_subagent,
                self.discard_subagent,
            ]
        builtin_tools = [Tool(fn) for fn in builtin_fns]
        builtin_names = {t.name for t in builtin_tools}

        self.extra_tools = list(tools) if tools else []
        merged_tools = list(builtin_tools)
        for extra in self.extra_tools:
            extra_tool = extra if isinstance(extra, Tool) else Tool(extra)
            if extra_tool.name in builtin_names:
                raise ValueError(
                    f"Tool name {extra_tool.name!r} collides with a built-in "
                    f"deep-agent tool. Rename the additional tool."
                )
            merged_tools.append(extra_tool)
        # Leading-underscore check is centralized in FunctionCallingAgent.

        self.agent = FunctionCallingAgent(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            final_instructions=self.final_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            reasoning_effort=self.reasoning_effort,
            use_chain_of_thought=self.use_chain_of_thought,
            tools=merged_tools,
            autonomous=self.autonomous,
            return_inputs_with_trajectory=self.return_inputs_with_trajectory,
            max_iterations=self.max_iterations,
            streaming=self.streaming,
            name="agent_" + self.name,
        )

    async def _materialize_inputs(self, inputs):
        """Replace data inputs with a metadata summary, full values on disk.

        A *pure* ``ChatMessages`` conversation passes through untouched. Any
        other input — including a data model that merely carries a ``messages``
        field alongside data — is treated as data: the full JSON is written to a
        collision-free file in the overlay and the LM is handed only an
        :class:`InputsSummary` naming that file, keeping large inputs out of the
        prompt while leaving the complete values reachable via the file tools.
        """
        if not inputs or is_strictly_chat_messages(inputs):
            return inputs
        inputs_json = inputs.get_json()
        if self._inputs_path is None:
            # Resolved while the overlay is still empty, so it reflects only the
            # workdir and never shadows a file the caller mounted.
            self._inputs_path = await unique_inputs_path(self.sandbox)
        await self.sandbox.write_file(
            self._inputs_path,
            json.dumps(inputs_json, indent=2, ensure_ascii=False),
        )
        return summarize_inputs(inputs_json, inputs_file=self._inputs_path)

    async def call(self, inputs, training=False):
        # Subagent branches are scoped to a single turn: a handle from a
        # previous call must not survive into the next one.
        self._subagents = {}
        self._subagent_counter = 0
        inputs = await self._materialize_inputs(inputs)
        return await self.agent(inputs, training=training)

    @staticmethod
    def _coerce_task(task) -> str:
        """Normalize one ``spawn_subagents`` task entry to an instruction string."""
        if isinstance(task, dict):
            return (
                task.get("task")
                or task.get("instructions")
                or task.get("prompt")
                or json.dumps(task, ensure_ascii=False)
            )
        return str(task)

    async def spawn_subagents(self, tasks: List[str]) -> dict:
        """Run subagents in parallel, each on an isolated branch of the filesystem.

        Each task is handed to a fresh subagent working on its own
        copy-on-write fork of the *current* filesystem: it can read every
        file you see now and freely write, edit or delete, but its changes
        are isolated and do NOT affect your filesystem. Subagents run
        concurrently. Nothing is applied automatically — review each
        returned ``diff`` and then call ``merge_subagent(handle)`` to fold
        the changes you want into your filesystem (or
        ``discard_subagent(handle)`` to drop a branch).

        Args:
            tasks (list): One instruction string per subagent describing
                what that subagent should accomplish.

        Returns:
            dict: ``subagents`` — a list of ``{handle, task, result, diff}``
            (``diff`` is the subagent's pending ``{written, deleted}``
            changes), or ``{handle, task, error}`` for a subagent that
            failed; plus a top-level ``error`` when ``tasks`` is empty.
        """
        prompts = [self._coerce_task(t) for t in (tasks or [])]
        if not prompts:
            return {"error": "no tasks provided"}

        # Local imports: agent modules avoid importing `Program` at module
        # scope to sidestep package-init import cycles.
        from synalinks.src.modules.core.input_module import Input
        from synalinks.src.programs.program import Program

        async def run_one(index: int, prompt: str):
            fork = self.sandbox.fork(name=f"{self.name}_sub{index}")
            subagent = DeepAgent(
                sandbox=fork,
                language_model=self.language_model,
                tools=self.extra_tools,
                instructions=get_subagent_instructions(),
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                use_chain_of_thought=self.use_chain_of_thought,
                max_iterations=self.max_iterations,
                max_subagent_depth=self.max_subagent_depth,
                _subagent_depth=self._subagent_depth + 1,
                return_inputs_with_trajectory=False,
                autonomous=True,
                name=f"{self.name}_sub{index}",
            )
            # Run the subagent through a Program — the canonical execution
            # path. A direct eager call would re-run the agent's
            # `compute_output_spec` on concrete inputs (extra throwaway LM
            # calls); building once with a symbolic Input keeps that step
            # LM-free, so the subagent costs the same as a normal DeepAgent.
            inputs = Input(data_model=ChatMessages)
            outputs = await subagent(inputs)
            program = Program(
                inputs=inputs,
                outputs=outputs,
                name=f"{self.name}_sub{index}_program",
            )
            messages = ChatMessages(
                messages=[ChatMessage(role=ChatRole.USER, content=prompt)]
            )
            output = await program(messages)
            return fork, _final_answer_text(output)

        results = await asyncio.gather(
            *(run_one(i, p) for i, p in enumerate(prompts)),
            return_exceptions=True,
        )

        report = []
        for prompt, result in zip(prompts, results):
            handle = f"subagent_{self._subagent_counter}"
            self._subagent_counter += 1
            if isinstance(result, Exception):
                report.append(
                    {
                        "handle": handle,
                        "task": prompt,
                        "error": f"{type(result).__name__}: {result}",
                    }
                )
                continue
            fork, answer = result
            self._subagents[handle] = fork
            report.append(
                {
                    "handle": handle,
                    "task": prompt,
                    "result": answer,
                    "diff": fork.diff(),
                }
            )
        return {"subagents": report}

    async def merge_subagent(
        self, handle: str, paths: Optional[List[str]] = None, force: bool = False
    ) -> dict:
        """Apply a subagent's filesystem changes onto your own filesystem.

        Folds the writes and deletions a subagent made on its branch into
        your filesystem. The handle stays valid afterwards, so you can merge
        a different subset later. A path you also changed since spawning is a
        conflict: it is **refused** (reported under ``conflicts`` /
        ``skipped`` and left as-is) unless you pass ``force=True``, which
        applies the subagent's version (last writer wins).

        Args:
            handle (str): A handle returned by ``spawn_subagents``.
            paths (list): Optional subset of virtual paths to merge; omit to
                merge all of the subagent's changes.
            force (bool): Apply conflicting paths instead of refusing them.
                Defaults to false.

        Returns:
            dict: ``{written, deleted, conflicts, skipped}`` virtual paths, or
            ``error`` for an unknown handle.
        """
        fork = self._subagents.get(handle)
        if fork is None:
            return {"error": f"unknown subagent handle: {handle!r}"}
        return self.sandbox.merge(fork, paths=paths, force=force)

    async def discard_subagent(self, handle: str) -> dict:
        """Drop a subagent's branch without applying any of its changes.

        Args:
            handle (str): A handle returned by ``spawn_subagents``.

        Returns:
            dict: ``{discarded: handle}``, or ``error`` for an unknown handle.
        """
        if self._subagents.pop(handle, None) is None:
            return {"error": f"unknown subagent handle: {handle!r}"}
        return {"discarded": handle}

    async def compute_output_spec(self, inputs, training=False):
        # Mirror the runtime shape: data inputs become an InputsSummary; only a
        # pure ChatMessages conversation is passed through.
        if inputs and not is_strictly_chat_messages(inputs):
            inputs = SymbolicDataModel(
                schema=InputsSummary.get_schema(),
                name="inputs_summary_" + self.name,
            )
        return await self.agent.compute_output_spec(inputs, training=training)

    def get_config(self):
        config = {
            "workdir": self.workdir,
            "timeout": self.timeout,
            "max_subagent_depth": self.max_subagent_depth,
            "schema": self.schema,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "final_instructions": self.final_instructions,
            "temperature": self.temperature,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "reasoning_effort": self.reasoning_effort,
            "use_chain_of_thought": self.use_chain_of_thought,
            "autonomous": self.autonomous,
            "return_inputs_with_trajectory": self.return_inputs_with_trajectory,
            "max_iterations": self.max_iterations,
            "streaming": self.streaming,
            "name": self.name,
            "description": self.description,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        tools_config = {
            "tools": [
                serialization_lib.serialize_synalinks_object(
                    t if isinstance(t, Tool) else Tool(t)
                )
                for t in self.extra_tools
            ]
        }
        return {**config, **language_model_config, **tools_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        tools = [
            serialization_lib.deserialize_synalinks_object(t)
            for t in config.pop("tools", [])
        ]
        return cls(
            language_model=language_model,
            tools=tools,
            **config,
        )
