# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import json
from typing import Dict
from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_strictly_chat_messages
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.agents.utils.agents_utils import InputsSummary
from synalinks.src.modules.agents.utils.agents_utils import resolve_workdir
from synalinks.src.modules.agents.utils.agents_utils import summarize_inputs
from synalinks.src.modules.agents.utils.agents_utils import unique_inputs_path
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox
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
edit_file, run_bash

Plan:
1. Use `list_files` to discover files (glob, e.g. `**/*.py`).
2. Use `search_files` to grep file contents by regex across a glob.
3. Use `read_file` to read a file; it returns the requested lines with
   1-based `start_line` / `end_line`. Page through long files with
   `offset` / `limit` (raise `offset` to read further in).
4. Use `edit_file` for surgical changes (preferred over `write_file`).
5. Use `run_bash` to run shell commands against the filesystem — pipes,
   redirects, globs, `&&`, loops, and `python3` (e.g. `python3 script.py`
   after `write_file`-ing it). `python3` is a real interpreter with the
   full standard library, third-party packages and network.
6. Once you have the answer, stop calling tools and respond.

Notes:
- The filesystem is host-safe: edits and new files land in the sandbox's
  mounted filesystem and never modify the real workspace on disk.
- If the conversation includes an `InputsSummary`, you only see field
  previews and sizes — the full, untruncated input values are saved as the
  JSON file named in its `inputs_file` field. Read it with the `read_file`
  tool (or `run_bash` `cat`) rather than retyping values from the preview.
""".strip()


def get_subagent_tools_guidance() -> str:
    """Guidance appended to the instructions when subagents are enabled."""
    return """
You can delegate work to parallel subagents, each on its own isolated
branch (a copy-on-write fork) of the filesystem:
- `spawn_subagents(tasks)`: launch one subagent per task string. Each
  subagent sees the files you see *now* and may freely read/write/edit/
  delete them, but its changes stay on its own branch — they never touch
  your filesystem. Subagents run concurrently, so use this to parallelize
  independent exploration or edits. Returns a `handle` and a `patch`
  (its pending changes as a git-style unified diff — the actual line-level
  edits) plus a structured `diff` summary per subagent.
- `merge_subagent(handle, paths=None, force=False)`: after reviewing a
  subagent's `patch`, fold its changes into your filesystem (pass `paths` to
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
write_file, edit_file, run_bash.

Plan:
1. Explore with `list_files` / `search_files` / `read_file`.
2. Make the changes your task requires with `edit_file` / `write_file`,
   or run code with `run_bash` (e.g. `python3 script.py`).
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
class DeepAgent(FunctionCallingAgent):
    """A coding agent whose tools are a sandboxed copy of a workdir.

    DeepAgent is a thin specialization of `FunctionCallingAgent`
    that mounts the workdir in a `MirageSandbox` and exposes the
    sandbox's tool methods to the LM:

    - ``read_file``: read a file by 1-based line range (paginated).
    - ``list_files``: list files matching a glob.
    - ``search_files``: glob for files and grep their contents (regex).
    - ``write_file``: create/overwrite a file.
    - ``edit_file``: exact-string replacement.
    - ``run_bash``: run a shell command (pipes, redirects, globs, loops and
      ``python3``) against the mounted filesystem.

    Every tool is backed by the sandbox's mounted filesystem, seeded from
    ``workdir``, so the agent is **host-safe by construction**: writes, edits
    and code execution land in the mount and can never modify the real
    ``workdir`` or reach the host — so there is nothing to gate, and all tools
    are always available. Inspect what the agent did through ``agent.sandbox``
    — ``changes()`` / ``diff()`` — and persist any of it yourself if desired.

    The constructor mirrors `FunctionCallingAgent` — every
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
        schema (dict): JSON schema for the final answer.
        data_model (DataModel): DataModel for the final answer.
            Mutually exclusive with ``schema``.
        language_model (LanguageModel): The language model that drives
            the agent loop.
        sub_language_model (LanguageModel): Optional. The language model
            that drives spawned subagents (typically a cheaper model).
            Defaults to ``language_model`` when omitted.
        prompt_template (str): Forwarded to the tool-call generator.
        examples (list): Few-shot examples for the tool-call generator.
        instructions (str): Override the default system instructions.
            When omitted, the default is built from the workdir.
        final_instructions (str): Instructions for the final-answer
            generator. Defaults to ``instructions``.
        temperature (float): LM sampling temperature. Defaults to None (the model's own default applies).
        max_tokens (int): Optional. Maximum number of tokens to generate.
            Default None (the model's own default; caps generation length).
        top_p (float): Optional. Nucleus sampling probability. Default None
            (the model's own default).
        top_k (int): Optional. Top-k sampling cutoff. Default None (the
            model's own default).
        use_inputs_schema (bool): Include the input schema in the prompt.
        use_outputs_schema (bool): Include the output schema in the prompt.
        reasoning_effort (str): Forwarded to the generators (for
            reasoning-capable LMs).
        use_chain_of_thought (bool): When ``True``, the tool-call
            generator emits a ``thinking`` field per round.
        tools (list): Additional `Tool` instances (or plain async
            functions) to expose alongside the built-in tools. Names must
            not start with ``_`` or collide with built-ins.
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
        timeout (float): Per-command execution budget in seconds for
            ``run_bash``. Defaults to 30.
        workdir (str): Optional working directory the agent operates on.
            When given it must exist and its files seed the sandbox
            filesystem (the LM's writes/edits stay there and never touch
            the real directory). When omitted, the sandbox starts as an
            empty in-memory filesystem.
        sandbox (Sandbox): Optional ready-made sandbox to operate on instead
            of building one from ``workdir`` — e.g. a `Sandbox.fork`
            of another agent's filesystem. When given, ``workdir`` is used
            only for the default instructions text.
        skills (list): Optional. Folder paths (Agent Skill roots) whose skills
            are listed for the agent as an ``<available_skills>`` context message
            (see `FunctionCallingAgent`). The skill files must also be reachable
            from the agent's sandbox (e.g. under ``workdir``) for their bodies to
            be read on demand. Defaults to ``None``.
        max_subagent_depth (int): When ``> 0``, the agent gains
            ``spawn_subagents`` / ``merge_subagent`` / ``discard_subagent``
            tools, letting the LM run subagents in parallel — each on an
            isolated `Sandbox.fork` of the filesystem whose changes
            only land on an explicit ``merge_subagent``. The value caps
            nesting: ``1`` (the recommended setting) lets this agent spawn
            subagents that cannot themselves spawn; ``2`` allows one more
            level, and so on. Defaults to ``0`` (subagents disabled —
            backward-compatible). Requires a fork-capable sandbox
            (``MirageSandbox`` is).

            Subagent forks here are **filesystem branches** (each gets a
            fresh interpreter), so across parallel subagents you can fold
            back **all** their file changes. Folding back Python REPL state
            (variables/functions/imports) across subagents is a
            `RecursiveLanguageModelAgent` feature — and limited to one
            subagent there, because the REPL namespace serializes only
            as a whole (it can't union parallel namespaces). That is a
            backend constraint, not a design shortcut.
        name (str): Module name.
        description (str): Module description.
    """

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        language_model=None,
        sub_language_model=None,
        prompt_template=None,
        examples=None,
        instructions: Optional[str] = None,
        final_instructions: Optional[str] = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        use_inputs_schema: bool = False,
        use_outputs_schema: bool = False,
        reasoning_effort: Optional[str] = None,
        use_chain_of_thought: bool = False,
        tools: Optional[List] = None,
        autonomous: bool = True,
        return_inputs_with_trajectory: bool = True,
        max_iterations: int = 10,
        streaming: bool = False,
        timeout: float = 30.0,
        workdir: Optional[str] = None,
        skills=None,
        sandbox=None,
        max_subagent_depth: int = 0,
        _subagent_depth: int = 0,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        # `workdir` is optional: when omitted the sandbox is a pure in-memory
        # filesystem. Resolve it first — the sandbox and default instructions
        # are derived from it, and they must exist before `super().__init__()`
        # (which calls `_get_builtin_tools`).
        self.workdir = resolve_workdir(workdir)

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

        # `sub_language_model` drives spawned subagents (typically a cheaper
        # model). Defaults to the primary LM when omitted; ``get(None)`` would
        # raise, so resolve only when a value is given.
        self.sub_language_model = (
            _get_lm(sub_language_model)
            if sub_language_model is not None
            else _get_lm(language_model)
        )

        # The sandbox IS the filesystem the tools operate on: it is seeded
        # from the workdir and writes/edits/shell stay host-safe in the
        # mount. Inspect via `self.sandbox.changes()` / `.diff()`. A caller
        # (or this agent, when spawning a subagent) may supply a ready-made
        # sandbox — e.g. a `fork()` of a parent's filesystem. Built before
        # `super().__init__()` because `_get_builtin_tools` binds its file tools.
        self.sandbox = (
            sandbox
            if sandbox is not None
            else MirageSandbox(workdir=self.workdir, timeout=self.timeout)
        )
        # Subagent branches (forks) awaiting the parent's review, keyed by
        # handle; reset per `call()`, populated by `spawn_subagents`.
        self._subagents: Dict[str, object] = {}
        self._subagent_counter = 0
        # Overlay path the full (data) inputs are written to, resolved once on
        # first use so it can't shadow a workdir file (see `_materialize_inputs`).
        self._inputs_path: Optional[str] = None

        if instructions is None:
            instructions = get_default_instructions(self.workdir)
        if self._subagents_enabled:
            instructions = instructions + "\n\n" + get_subagent_tools_guidance()

        super().__init__(
            schema=schema,
            data_model=data_model,
            language_model=language_model,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            final_instructions=final_instructions,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            use_inputs_schema=use_inputs_schema,
            use_outputs_schema=use_outputs_schema,
            reasoning_effort=reasoning_effort,
            use_chain_of_thought=use_chain_of_thought,
            tools=tools,
            autonomous=autonomous,
            return_inputs_with_trajectory=return_inputs_with_trajectory,
            max_iterations=max_iterations,
            streaming=streaming,
            workdir=workdir,
            skills=skills,
            name=name,
            description=description,
        )

    def _get_builtin_tools(self):
        builtin_fns = [
            self.sandbox.read_file,
            self.sandbox.list_files,
            self.sandbox.search_files,
            self.sandbox.write_file,
            self.sandbox.edit_file,
            self.sandbox.run_bash,
        ]
        if self._subagents_enabled:
            builtin_fns += [
                self.spawn_subagents,
                self.merge_subagent,
                self.discard_subagent,
            ]
        return [Tool(fn) for fn in builtin_fns]

    def _builtin_tool_kind(self):
        return "deep-agent"

    async def _materialize_inputs(self, inputs):
        """Replace data inputs with a metadata summary, full values on disk.

        A *pure* ``ChatMessages`` conversation passes through untouched. Any
        other input — including a data model that merely carries a ``messages``
        field alongside data — is treated as data: the full JSON is written to a
        collision-free file in the overlay and the LM is handed only an
        `InputsSummary` naming that file, keeping large inputs out of the
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
        return await super().call(inputs, training=training)

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
        returned ``patch`` and then call ``merge_subagent(handle)`` to fold
        the changes you want into your filesystem (or
        ``discard_subagent(handle)`` to drop a branch).

        Args:
            tasks (list): One instruction string per subagent describing
                what that subagent should accomplish.

        Returns:
            dict: ``subagents`` — a list of
            ``{handle, task, result, diff, patch}`` per subagent, where
            ``patch`` is the subagent's pending changes as a git-style unified
            diff (the actual line-level edits) and ``diff`` is the structured
            ``{written, deleted}`` summary of changed paths; or
            ``{handle, task, error}`` for a subagent that failed; plus a
            top-level ``error`` when ``tasks`` is empty.
        """
        prompts = [self._coerce_task(t) for t in (tasks or [])]
        if not prompts:
            return {"error": "no tasks provided"}

        # Local imports: agent modules avoid importing `Program` at module
        # scope to sidestep package-init import cycles.
        from synalinks.src.modules.core.input_module import Input
        from synalinks.src.programs.program import Program

        async def run_one(index: int, prompt: str):
            # Subagents inherit the parent's confinement (``confine=None``):
            # when this agent's sandbox is confined, the subagent is confined to
            # its OWN fork (host hidden, network cut, isolated filesystem, and
            # the parent's egress/mount/seccomp posture); when the parent runs
            # unconfined, so does the subagent.
            fork = self.sandbox.fork(name=f"{self.name}_sub{index}", confine=None)
            subagent = DeepAgent(
                sandbox=fork,
                language_model=self.sub_language_model,
                sub_language_model=self.sub_language_model,
                tools=self.extra_tools,
                instructions=get_subagent_instructions(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
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
                    "patch": fork.patch(),
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
        return await super().compute_output_spec(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "timeout": self.timeout,
                "max_subagent_depth": self.max_subagent_depth,
                "sub_language_model": serialization_lib.serialize_synalinks_object(
                    self.sub_language_model,
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if config.get("sub_language_model") is not None:
            config["sub_language_model"] = serialization_lib.deserialize_synalinks_object(
                config.pop("sub_language_model")
            )
        else:
            config.pop("sub_language_model", None)
        return super().from_config(config)
